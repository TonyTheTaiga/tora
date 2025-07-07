use crate::ntypes;
use axum::{
    Json,
    extract::Request,
    http::{HeaderMap, StatusCode, header::SET_COOKIE},
    middleware::{self, Next},
    response::{IntoResponse, Redirect, Response},
    routing::MethodRouter,
};
use axum_extra::extract::CookieJar;
use axum_extra::extract::cookie::{Cookie, SameSite};
use base64::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use std::env;
use std::time::{SystemTime, UNIX_EPOCH};
use supabase_auth::models::{AuthClient, User};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticatedUser {
    pub id: String,
    pub email: String,
    pub refresh_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TokenPayload {
    pub access_token: String,
    pub token_type: String,
    pub expires_in: u64,
    pub expires_at: i64,
    pub refresh_token: String,
    pub user: User,
}

impl AuthenticatedUser {
    pub fn new(user: User, refresh_token: String) -> Self {
        Self {
            id: user.id.to_string(),
            email: user.email,
            refresh_token,
        }
    }
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
    jar: CookieJar,
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

    if let Some(cookie) = jar.get("tora_auth_token") {
        let token_b64 = cookie.value();
        let auth_client = AuthClient::new_from_env().map_err(|e| AuthError {
            message: format!("Failed to create auth client: {e}"),
            status_code: StatusCode::INTERNAL_SERVER_ERROR,
        })?;

        match decode_and_validate_token(token_b64, &auth_client).await {
            Ok(payload) => {
                let authenticated_user =
                    AuthenticatedUser::new(payload.user.clone(), payload.refresh_token.clone());
                req.extensions_mut().insert(authenticated_user);

                let current_time = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                if current_time as i64 >= payload.expires_at - 300 {
                    let new_payload = serde_json::json!({
                        "access_token": payload.access_token,
                        "token_type": "bearer",
                        "expires_in": payload.expires_in,
                        "expires_at": payload.expires_at,
                        "refresh_token": payload.refresh_token,
                        "user": payload.user
                    });

                    let payload_json = serde_json::to_string(&new_payload).unwrap();
                    let payload_base64 = BASE64_STANDARD.encode(payload_json.as_bytes());

                    let is_production = std::env::var("RUST_ENV")
                        .unwrap_or_else(|_| "development".to_string())
                        == "production";
                    let cookie = Cookie::build(("tora_auth_token", payload_base64))
                        .http_only(true)
                        .secure(is_production)
                        .same_site(SameSite::Lax)
                        .path("/");

                    let mut response = next.run(req).await;
                    response
                        .headers_mut()
                        .insert(SET_COOKIE, cookie.to_string().parse().unwrap());
                    return Ok(response);
                }

                return Ok(next.run(req).await);
            }
            Err(auth_error) => {
                if auth_error.status_code == StatusCode::UNAUTHORIZED {
                    return Err(auth_error);
                }
                return Err(auth_error);
            }
        }
    }

    Err(AuthError {
        message: "Missing authentication credentials".to_string(),
        status_code: StatusCode::UNAUTHORIZED,
    })
}

async fn decode_and_validate_token(
    token_b64: &str,
    auth_client: &AuthClient,
) -> Result<TokenPayload, AuthError> {
    let token_json = BASE64_STANDARD.decode(token_b64).map_err(|e| AuthError {
        message: format!("Invalid token format: {e}"),
        status_code: StatusCode::UNAUTHORIZED,
    })?;

    let token_str = String::from_utf8(token_json).map_err(|e| AuthError {
        message: format!("Invalid token encoding: {e}"),
        status_code: StatusCode::UNAUTHORIZED,
    })?;

    let mut payload: TokenPayload = serde_json::from_str(&token_str).map_err(|e| AuthError {
        message: format!("Invalid token payload: {e}"),
        status_code: StatusCode::UNAUTHORIZED,
    })?;

    let current_time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();

    if current_time as i64 >= payload.expires_at {
        match auth_client.refresh_session(&payload.refresh_token).await {
            Ok(new_session) => {
                payload.access_token = new_session.access_token;
                payload.expires_at = new_session.expires_at as i64;
                payload.expires_in = new_session.expires_in as u64;
                payload.refresh_token = new_session.refresh_token;
                payload.user = new_session.user;
                Ok(payload)
            }
            Err(e) => Err(AuthError {
                message: format!("Failed to refresh token: {e}"),
                status_code: StatusCode::UNAUTHORIZED,
            }),
        }
    } else {
        let user = auth_client
            .get_user(&payload.access_token)
            .await
            .map_err(|e| AuthError {
                message: format!("Invalid access token: {e}"),
                status_code: StatusCode::UNAUTHORIZED,
            })?;
        payload.user = user;
        Ok(payload)
    }
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

    // if let Some(auth) = headers.get("authorization") {
    //     if let Ok(auth_str) = auth.to_str() {
    //         if let Some(key) = auth_str.strip_prefix("ApiKey ") {
    //             return Some(key.to_string());
    //         }
    //         if let Some(key) = auth_str.strip_prefix("Bearer ") {
    //             // Only if it starts with "ak_" (API key prefix)
    //             if key.starts_with("ak_") {
    //                 return Some(key.to_string());
    //             }
    //         }
    //     }
    // }

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
                refresh_token: String::new(), // Empty for API key auth
            })
        }
        None => Err(AuthError {
            message: "Invalid or revoked API key".to_string(),
            status_code: StatusCode::UNAUTHORIZED,
        }),
    }
}

pub async fn ui_auth_middleware(
    jar: CookieJar,
    req: Request,
    next: Next,
) -> Result<Response, Response> {
    let path = req.uri().path();
    println!("path to navigate to {path}");

    let protected_paths = ["/settings", "/app"];
    if protected_paths.iter().any(|p| path.starts_with(p)) {
        if let Some(cookie) = jar.get("tora_auth_token") {
            if let Ok(auth_client) = AuthClient::new_from_env() {
                if let Ok(_payload) = decode_and_validate_token(cookie.value(), &auth_client).await
                {
                    return Ok(next.run(req).await);
                }
            }
        }
        return Ok(Redirect::to("/login").into_response());
    }

    Ok(next.run(req).await)
}

pub async fn redirect_if_authenticated_middleware(
    jar: CookieJar,
    req: Request,
    next: Next,
) -> Result<Response, Response> {
    let path = req.uri().path();

    let auth_paths = ["/", "/login", "/signup"];
    if auth_paths.iter().any(|p| path.starts_with(p)) {
        if let Some(cookie) = jar.get("tora_auth_token") {
            if let Ok(auth_client) = AuthClient::new_from_env() {
                if let Ok(_payload) = decode_and_validate_token(cookie.value(), &auth_client).await
                {
                    return Ok(Redirect::to("/workspaces").into_response());
                }
            }
        }
    }

    Ok(next.run(req).await)
}

pub fn protected_route<T>(method_router: MethodRouter<T>) -> MethodRouter<T>
where
    T: Clone + Send + Sync + 'static,
{
    method_router.layer(middleware::from_fn(auth_middleware))
}
