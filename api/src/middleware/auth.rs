use crate::state;
use crate::types;
use axum::{
    Json,
    extract::{Request, State},
    http::{HeaderMap, Method, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use http::HeaderValue;
use http::header::WWW_AUTHENTICATE;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use once_cell::sync::Lazy;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

const JWKS_TTL_SECS: u64 = 600;

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
        let mut res = (self.status_code, body).into_response();
        if self.status_code == StatusCode::UNAUTHORIZED {
            // Instruct clients about the required auth scheme
            res.headers_mut().insert(
                WWW_AUTHENTICATE,
                HeaderValue::from_static("Bearer realm=\"api\""),
            );
        }
        res
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JwtClaims {
    sub: String,
    #[serde(default)]
    email: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    exp: Option<u64>,
    #[serde(default)]
    iss: Option<String>,
    #[serde(default)]
    aud: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct Jwk {
    kid: Option<String>,
    kty: String,
    // #[serde(default)]
    // alg: Option<String>,
    #[serde(default)]
    n: Option<String>,
    #[serde(default)]
    e: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct Jwks {
    keys: Vec<Jwk>,
}

#[derive(Default)]
struct JwksCache {
    fetched_at: Option<Instant>,
    keys: HashMap<String, Arc<DecodingKey>>,
}

static JWKS_CACHE: Lazy<RwLock<JwksCache>> = Lazy::new(|| RwLock::new(JwksCache::default()));
static HTTP: Lazy<HttpClient> = Lazy::new(|| {
    HttpClient::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .expect("http client")
});

pub async fn auth_middleware(
    headers: HeaderMap,
    State(app_state): State<state::AppState>,
    mut req: Request,
    next: Next,
) -> Result<Response, AuthError> {
    let uri = req.uri().clone();
    let method = req.method().clone();
    debug!("Auth middleware: {} {}", method, uri);

    if should_skip_auth(&method) {
        debug!("Skipping auth for {} {}", method, uri);
        return Ok(next.run(req).await);
    }

    /*
    First check for API key
    */
    if let Some(api_user) = authenticate_api_key(&app_state.db_pool, &headers).await? {
        req.extensions_mut().insert(api_user);
        return Ok(next.run(req).await);
    }

    /*
    Then check for access token
    */
    if let Some(token) = bearer_token_from_headers(&headers) {
        match verify_jwt(&token, &app_state.settings).await {
            Ok(authenticated_user) => {
                info!(
                    "JWT authentication successful for user: {}",
                    authenticated_user.email
                );
                req.extensions_mut().insert(authenticated_user);
                return Ok(next.run(req).await);
            }
            Err(e) => {
                warn!("JWT authentication failed: {}", e.message);
                return Err(e);
            }
        }
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

fn should_skip_auth(method: &Method) -> bool {
    *method == Method::OPTIONS || *method == Method::HEAD
}

fn bearer_token_from_headers(headers: &HeaderMap) -> Option<String> {
    if let Some(auth_header) = headers.get("authorization") {
        debug!("Found authorization header");
        match auth_header.to_str() {
            Ok(auth_str) => {
                let mut parts = auth_str.split_whitespace();
                let scheme = parts.next();
                let token = parts.next();
                if matches!(scheme, Some(s) if s.eq_ignore_ascii_case("Bearer")) && token.is_some()
                {
                    debug!("Extracted bearer token from authorization header");
                    return Some(token.unwrap().to_string());
                } else {
                    debug!("Authorization header does not contain Bearer token");
                }
            }
            Err(_) => warn!("Authorization header contains invalid UTF-8"),
        }
    } else {
        debug!("No authorization header found");
    }
    None
}

async fn verify_jwt(
    token: &str,
    settings: &crate::settings::Settings,
) -> Result<AuthenticatedUser, AuthError> {
    let header = decode_header(token).map_err(|e| AuthError {
        message: format!("Invalid token header: {e}"),
        status_code: StatusCode::UNAUTHORIZED,
    })?;

    info!("JWT header alg={:?} kid={:?}", header.alg, header.kid);
    match header.alg {
        Algorithm::HS256 => {
            let mut validation = Validation::new(Algorithm::HS256);
            validation.validate_aud = false; // don't enforce audience
            let _ = validation.required_spec_claims.remove("aud");
            let token_data = decode::<JwtClaims>(
                token,
                &DecodingKey::from_secret(settings.supabase_jwt_secret.as_bytes()),
                &validation,
            )
            .map_err(|e| {
                error!("JWT decode failed: {}", e);
                AuthError {
                    message: "Invalid access token".to_string(),
                    status_code: StatusCode::UNAUTHORIZED,
                }
            })?;
            let JwtClaims {
                sub,
                email,
                iss,
                aud,
                ..
            } = token_data.claims;
            info!(
                "JWT claims verified (HS256): sub={}, iss={:?}, aud={:?}",
                sub, iss, aud
            );
            let email = email.ok_or(AuthError {
                message: "Token missing required email claim".to_string(),
                status_code: StatusCode::UNAUTHORIZED,
            })?;
            Ok(AuthenticatedUser { id: sub, email })
        }
        Algorithm::RS256 => {
            let kid = header.kid.as_deref();
            let key = get_jwk_key(settings.supabase_url.as_str(), kid).await?;
            let mut validation = Validation::new(Algorithm::RS256);
            validation.validate_aud = false; // don't enforce audience
            let _ = validation.required_spec_claims.remove("aud");
            let token_data = decode::<JwtClaims>(token, &key, &validation).map_err(|e| {
                error!("JWT decode failed: {}", e);
                AuthError {
                    message: "Invalid access token".to_string(),
                    status_code: StatusCode::UNAUTHORIZED,
                }
            })?;
            let JwtClaims {
                sub,
                email,
                iss,
                aud,
                ..
            } = token_data.claims;
            info!(
                "JWT claims verified (RS256): sub={}, iss={:?}, aud={:?}",
                sub, iss, aud
            );
            let email = email.ok_or(AuthError {
                message: "Token missing required email claim".to_string(),
                status_code: StatusCode::UNAUTHORIZED,
            })?;
            Ok(AuthenticatedUser { id: sub, email })
        }
        other => Err(AuthError {
            message: format!("Unsupported JWT algorithm: {other:?}"),
            status_code: StatusCode::UNAUTHORIZED,
        }),
    }
}

async fn get_jwk_key(supabase_url: &str, kid: Option<&str>) -> Result<Arc<DecodingKey>, AuthError> {
    {
        let cache = JWKS_CACHE.read().await;
        let fresh = cache
            .fetched_at
            .map(|t| t.elapsed() < Duration::from_secs(JWKS_TTL_SECS))
            .unwrap_or(false);

        if fresh {
            if let Some(k) = kid {
                if let Some(entry) = cache.keys.get(k) {
                    info!("Using cached JWK for kid={}", k);
                    return Ok(entry.clone());
                }
            } else if cache.keys.len() == 1 {
                let dk = cache.keys.values().next().unwrap().clone();
                info!("Using single cached JWK (no kid)");
                return Ok(dk);
            }
        }
    }

    let jwks_url = format!("{}/auth/v1/jwks", supabase_url.trim_end_matches('/'));
    info!("Fetching JWKS: {}", jwks_url);
    let resp = HTTP.get(&jwks_url).send().await.map_err(|e| AuthError {
        message: format!("Failed to fetch JWKS: {e}"),
        status_code: StatusCode::INTERNAL_SERVER_ERROR,
    })?;
    if !resp.status().is_success() {
        return Err(AuthError {
            message: format!("JWKS HTTP error: {}", resp.status()),
            status_code: StatusCode::INTERNAL_SERVER_ERROR,
        });
    }
    let jwks: Jwks = resp.json().await.map_err(|e| AuthError {
        message: format!("Invalid JWKS payload: {e}"),
        status_code: StatusCode::INTERNAL_SERVER_ERROR,
    })?;

    let map: HashMap<String, Arc<DecodingKey>> = jwks
        .keys
        .into_iter()
        .filter_map(|k| {
            if k.kty != "RSA" {
                return None;
            }
            let (Some(n), Some(e), Some(kid)) = (k.n, k.e, k.kid) else {
                return None;
            };
            match DecodingKey::from_rsa_components(&n, &e) {
                Ok(dk) => Some((kid, Arc::new(dk))),
                Err(err) => {
                    warn!("Skipping invalid JWK '{}': {}", kid, err);
                    None
                }
            }
        })
        .collect();

    {
        let mut cache = JWKS_CACHE.write().await;
        let count = map.len();
        cache.keys = map;
        cache.fetched_at = Some(Instant::now());
        info!("JWKS cache updated: {} keys", count);
    }

    let cache = JWKS_CACHE.read().await;
    if let Some(k) = kid {
        if let Some(entry) = cache.keys.get(k) {
            info!("Using refreshed JWK for kid={}", k);
            return Ok(entry.clone());
        }
        return Err(AuthError {
            message: format!("Requested JWK kid '{k}' not found"),
            status_code: StatusCode::UNAUTHORIZED,
        });
    }
    if cache.keys.len() == 1 {
        let dk = cache.keys.values().next().unwrap().clone();
        info!("Using the single JWK from cache (no kid)");
        return Ok(dk);
    }

    Err(AuthError {
        message: "No suitable JWK found".to_string(),
        status_code: StatusCode::UNAUTHORIZED,
    })
}

async fn authenticate_api_key(
    pool: &sqlx::PgPool,
    headers: &HeaderMap,
) -> Result<Option<AuthenticatedUser>, AuthError> {
    if let Some(api_key) = extract_api_key(headers) {
        debug!("Found API key in request headers");
        match validate_api_key(pool.clone(), &api_key).await {
            Ok(authenticated_user) => {
                info!(
                    "API key authentication successful for user: {}",
                    authenticated_user.email
                );
                Ok(Some(authenticated_user))
            }
            Err(e) => {
                warn!("API key authentication failed: {}", e.message);
                Err(e)
            }
        }
    } else {
        Ok(None)
    }
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
            if let Ok(id_uuid) = uuid::Uuid::parse_str(&api_key_record.id) {
                match sqlx::query("UPDATE api_keys SET last_used = NOW() WHERE id = $1")
                    .bind(id_uuid)
                    .execute(&pool)
                    .await
                {
                    Ok(_) => debug!("Updated last_used timestamp for API key"),
                    Err(e) => warn!("Failed to update API key last_used timestamp: {}", e),
                }
            } else {
                warn!("Invalid UUID in api_keys.id when updating last_used");
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
