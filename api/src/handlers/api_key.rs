use axum::{
    Extension, Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes::{ApiKey, CreateApiKeyRequest, Response};
use sqlx::PgPool;
use uuid::Uuid;
use sha2::{Sha256, Digest};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use rand::Rng;

pub async fn create_api_key(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Json(request): Json<CreateApiKeyRequest>,
) -> impl IntoResponse {
    let user_uuid = match Uuid::parse_str(&user.id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid user ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Generate a secure API key
    let mut rng = rand::thread_rng();
    let key_bytes: [u8; 32] = rng.gen();
    let key_value = format!("tora_{}", BASE64.encode(key_bytes));
    
    // Hash the key for storage
    let mut hasher = Sha256::new();
    hasher.update(key_value.as_bytes());
    let key_hash = format!("{:x}", hasher.finalize());

    let mut tx = match pool.begin().await {
        Ok(tx) => tx,
        Err(e) => {
            eprintln!("Failed to begin transaction: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to create API key".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Insert the API key
    let result = sqlx::query_as::<_, (String, chrono::DateTime<chrono::Utc>)>(
        "INSERT INTO api_keys (user_id, name, key_hash) VALUES ($1, $2, $3) RETURNING id::text, created_at",
    )
    .bind(user_uuid)
    .bind(&request.name)
    .bind(&key_hash)
    .fetch_one(&mut *tx)
    .await;

    match result {
        Ok((id, created_at)) => {
            if let Err(e) = tx.commit().await {
                eprintln!("Failed to commit transaction: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(Response {
                        status: 500,
                        data: Some("Failed to create API key".to_string()),
                    }),
                )
                    .into_response();
            }

            let api_key = ApiKey {
                id,
                name: request.name,
                created_at: created_at.format("%b %d").to_string(),
                revoked: false,
                key: Some(key_value), // Only returned on creation
            };

            (
                StatusCode::CREATED,
                Json(Response {
                    status: 201,
                    data: Some(api_key),
                }),
            )
                .into_response()
        }
        Err(e) => {
            eprintln!("Failed to create API key: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to create API key".to_string()),
                }),
            )
                .into_response()
        }
    }
}

pub async fn list_api_keys(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
) -> impl IntoResponse {
    let user_uuid = match Uuid::parse_str(&user.id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid user ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    let result = sqlx::query_as::<_, (String, String, chrono::DateTime<chrono::Utc>, bool)>(
        "SELECT id::text, name, created_at, revoked FROM api_keys WHERE user_id = $1 ORDER BY created_at DESC",
    )
    .bind(user_uuid)
    .fetch_all(&pool)
    .await;

    match result {
        Ok(rows) => {
            let api_keys: Vec<ApiKey> = rows
                .into_iter()
                .map(|(id, name, created_at, revoked)| ApiKey {
                    id,
                    name,
                    created_at: created_at.format("%b %d").to_string(),
                    revoked,
                    key: None, // Never include actual key in list
                })
                .collect();

            Json(Response {
                status: 200,
                data: Some(api_keys),
            })
                .into_response()
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to fetch API keys".to_string()),
                }),
            )
                .into_response()
        }
    }
}

pub async fn revoke_api_key(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Path(key_id): Path<String>,
) -> impl IntoResponse {
    let user_uuid = match Uuid::parse_str(&user.id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid user ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    let key_uuid = match Uuid::parse_str(&key_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid API key ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    let result = sqlx::query(
        "UPDATE api_keys SET revoked = true WHERE id = $1 AND user_id = $2",
    )
    .bind(key_uuid)
    .bind(user_uuid)
    .execute(&pool)
    .await;

    match result {
        Ok(_) => {
            Json(Response {
                status: 200,
                data: Some("API key revoked successfully"),
            })
                .into_response()
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to revoke API key".to_string()),
                }),
            )
                .into_response()
        }
    }
}