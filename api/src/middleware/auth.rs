use crate::state;
use crate::types;
use axum::{
    Json,
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::MethodRouter,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use supabase_auth::models::AuthClient;
use tracing::{debug, error, info, warn};

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
        let body = Json(types::Response {
            status: self.status_code.as_u16() as i16,
            data: Some(serde_json::json!({
                "error": "Authentication failed",
                "message": self.message
            })),
        });
        (self.status_code, body).into_response()
    }
}

pub fn protected_route<T>(
    method_router: MethodRouter<T>,
    app_state: &state::AppState,
) -> MethodRouter<T>
where
    T: Clone + Send + Sync + 'static,
{
    method_router.layer(middleware::from_fn_with_state(
        app_state.clone(),
        auth_middleware,
    ))
}

pub async fn auth_middleware(
    headers: HeaderMap,
    State(app_state): State<state::AppState>,
    mut req: Request,
    next: Next,
) -> Result<Response, AuthError> {
    let uri = req.uri().clone();
    let method = req.method().clone();
    debug!("Auth middleware: {} {}", method, uri);

    /*
    First check for API key
    */
    if let Some(api_key) = extract_api_key(&headers) {
        debug!("Found API key in request headers");
        match validate_api_key(app_state.db_pool, &api_key).await {
            Ok(authenticated_user) => {
                info!(
                    "API key authentication successful for user: {}",
                    authenticated_user.email
                );
                req.extensions_mut().insert(authenticated_user);
                return Ok(next.run(req).await);
            }
            Err(e) => {
                warn!("API key authentication failed: {}", e.message);
                return Err(e);
            }
        }
    }

    /*
    Then check for access token
    */
    if let Some(auth_header) = headers.get("authorization") {
        debug!("Found authorization header");
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                debug!("Extracted bearer token from authorization header");
                let auth_client = AuthClient::new_from_env().map_err(|e| {
                    error!("Failed to create auth client: {}", e);
                    AuthError {
                        message: format!("Failed to create auth client: {e}"),
                        status_code: StatusCode::INTERNAL_SERVER_ERROR,
                    }
                })?;

                match auth_client.get_user(token).await {
                    Ok(user) => {
                        let authenticated_user = AuthenticatedUser {
                            id: user.id.to_string(),
                            email: user.email.clone(),
                        };
                        info!(
                            "Bearer token authentication successful for user: {}",
                            user.email
                        );
                        req.extensions_mut().insert(authenticated_user);
                        return Ok(next.run(req).await);
                    }
                    Err(e) => {
                        warn!("Bearer token authentication failed: {}", e);
                        return Err(AuthError {
                            message: "Invalid access token".to_string(),
                            status_code: StatusCode::UNAUTHORIZED,
                        });
                    }
                }
            } else {
                debug!("Authorization header does not contain Bearer token");
            }
        } else {
            warn!("Authorization header contains invalid UTF-8");
        }
    } else {
        debug!("No authorization header found");
    }

    warn!(
        "Authentication failed: no valid API key or bearer token found for {} {}",
        method, uri
    );
    Err(AuthError {
        message: "Missing authentication credentials".to_string(),
        status_code: StatusCode::UNAUTHORIZED,
    })
}

fn extract_api_key(headers: &HeaderMap) -> Option<String> {
    if let Some(key) = headers.get("x-api-key") {
        if let Ok(key_str) = key.to_str() {
            return Some(key_str.to_string());
        }
    }

    None
}

async fn validate_api_key(
    pool: sqlx::PgPool,
    api_key: &str,
) -> Result<AuthenticatedUser, AuthError> {
    debug!("Validating API key (length: {})", api_key.len());
    let mut hasher = Sha256::new();
    hasher.update(api_key.as_bytes());
    let key_hash = format!("{:x}", hasher.finalize());
    debug!("Generated hash for API key validation");
    let record = sqlx::query_as::<_, types::ApiKeyRecord>(
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
    .map_err(|e| {
        error!("Database error during API key validation: {}", e);
        AuthError {
            message: format!("Database query failed: {e}"),
            status_code: StatusCode::INTERNAL_SERVER_ERROR,
        }
    })?;

    match record {
        Some(api_key_record) => {
            debug!(
                "API key found in database for user: {}",
                api_key_record.user_email
            );
            let update_result = sqlx::query("UPDATE api_keys SET last_used = NOW() WHERE id = $1")
                .bind(uuid::Uuid::parse_str(&api_key_record.id).unwrap())
                .execute(&pool)
                .await;

            if let Err(e) = update_result {
                warn!("Failed to update API key last_used timestamp: {}", e);
            } else {
                debug!("Updated last_used timestamp for API key");
            }

            Ok(AuthenticatedUser {
                id: api_key_record.user_id,
                email: api_key_record.user_email,
            })
        }
        None => {
            debug!("API key not found in database or is revoked");
            Err(AuthError {
                message: "Invalid or revoked API key".to_string(),
                status_code: StatusCode::UNAUTHORIZED,
            })
        }
    }
}
