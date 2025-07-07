use crate::ntypes;
use axum::{
    Json,
    extract::Request,
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::MethodRouter,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use std::env;
use supabase_auth::models::AuthClient;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticatedUser {
    pub id: String,
    pub email: String,
}

#[derive(Debug)]
pub struct AuthError {
    pub message: String,
    pub status_code: StatusCode,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let body = Json(ntypes::Response {
            status: self.status_code.as_u16() as i16,
            data: Some(serde_json::json!({
                "error": "Authentication failed",
                "message": self.message
            })),
        });
        (self.status_code, body).into_response()
    }
}

pub async fn auth_middleware(
    headers: HeaderMap,
    mut req: Request,
    next: Next,
) -> Result<Response, AuthError> {
    if let Some(api_key) = extract_api_key(&headers) {
        match validate_api_key(&api_key).await {
            Ok(authenticated_user) => {
                req.extensions_mut().insert(authenticated_user);
                return Ok(next.run(req).await);
            }
            Err(e) => return Err(e),
        }
    }

    if let Some(auth_header) = headers.get("authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                let auth_client = AuthClient::new_from_env().map_err(|e| AuthError {
                    message: format!("Failed to create auth client: {e}"),
                    status_code: StatusCode::INTERNAL_SERVER_ERROR,
                })?;

                match auth_client.get_user(token).await {
                    Ok(user) => {
                        let authenticated_user = AuthenticatedUser {
                            id: user.id.to_string(),
                            email: user.email,
                        };
                        req.extensions_mut().insert(authenticated_user);
                        return Ok(next.run(req).await);
                    }
                    Err(_) => {
                        return Err(AuthError {
                            message: "Invalid access token".to_string(),
                            status_code: StatusCode::UNAUTHORIZED,
                        });
                    }
                }
            }
        }
    }

    Err(AuthError {
        message: "Missing authentication credentials".to_string(),
        status_code: StatusCode::UNAUTHORIZED,
    })
}

async fn create_db_pool() -> Result<PgPool, sqlx::Error> {
    let password = env::var("SUPABASE_PASSWORD")
        .map_err(|_| sqlx::Error::Configuration("SUPABASE_PASSWORD not set".into()))?;

    let connection_string = format!(
        "postgresql://postgres.hecctslcfhdrpnwovwbc:{password}@aws-0-us-east-1.compute-1.amazonaws.com:5432/postgres"
    );

    sqlx::postgres::PgPoolOptions::new()
        .connect(&connection_string)
        .await
}

fn extract_api_key(headers: &HeaderMap) -> Option<String> {
    if let Some(key) = headers.get("x-api-key") {
        if let Ok(key_str) = key.to_str() {
            return Some(key_str.to_string());
        }
    }

    None
}

async fn validate_api_key(api_key: &str) -> Result<AuthenticatedUser, AuthError> {
    let pool = create_db_pool().await.map_err(|e| AuthError {
        message: format!("Database connection failed: {e}"),
        status_code: StatusCode::INTERNAL_SERVER_ERROR,
    })?;

    let mut hasher = Sha256::new();
    hasher.update(api_key.as_bytes());
    let key_hash = format!("{:x}", hasher.finalize());

    let record = sqlx::query_as::<_, ntypes::ApiKeyRecord>(
        r#"
        SELECT
            ak.id::text,
            ak.user_id::text,
            ak.name,
            ak.key_hash,
            ak.created_at,
            ak.last_used,
            ak.revoked,
            u.email as user_email
        FROM api_keys ak
        JOIN auth.users u ON ak.user_id = u.id
        WHERE ak.key_hash = $1 AND ak.revoked = false
        "#,
    )
    .bind(&key_hash)
    .fetch_optional(&pool)
    .await
    .map_err(|e| AuthError {
        message: format!("Database query failed: {e}"),
        status_code: StatusCode::INTERNAL_SERVER_ERROR,
    })?;

    match record {
        Some(api_key_record) => {
            let _ = sqlx::query("UPDATE api_keys SET last_used = NOW() WHERE id = $1")
                .bind(uuid::Uuid::parse_str(&api_key_record.id).unwrap())
                .execute(&pool)
                .await;

            Ok(AuthenticatedUser {
                id: api_key_record.user_id,
                email: api_key_record.user_email,
            })
        }
        None => Err(AuthError {
            message: "Invalid or revoked API key".to_string(),
            status_code: StatusCode::UNAUTHORIZED,
        }),
    }
}

pub fn protected_route<T>(method_router: MethodRouter<T>) -> MethodRouter<T>
where
    T: Clone + Send + Sync + 'static,
{
    method_router.layer(middleware::from_fn(auth_middleware))
}
