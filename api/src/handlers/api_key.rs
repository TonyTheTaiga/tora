use crate::middleware::auth::AuthenticatedUser;
use crate::types::{ApiKey, CreateApiKeyRequest, Response};
use axum::{
    Extension, Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use rand::distributions::Alphanumeric;
use rand::{Rng, thread_rng};
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use uuid::Uuid;

pub async fn create_api_key(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Json(request): Json<CreateApiKeyRequest>,
) -> impl IntoResponse {
    let key_value: String = thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();
    let full_key = format!("tora_sk_{key_value}");

    let mut hasher = Sha256::new();
    hasher.update(&full_key);
    let key_hash = format!("{:x}", hasher.finalize());

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

    let result = sqlx::query_as::<_, ApiKey>(
        "INSERT INTO api_keys (user_id, name, key_hash) VALUES ($1, $2, $3) RETURNING id::text, name, created_at, revoked, NULL as key"
    )
    .bind(user_uuid)
    .bind(&request.name)
    .bind(&key_hash)
    .fetch_one(&pool)
    .await;

    match result {
        Ok(mut api_key) => {
            api_key.key = Some(full_key);
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
            eprintln!("Failed to create API key: {e}");
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

    let result = sqlx::query_as::<_, ApiKey>(
        r#"
        SELECT id::text, name, created_at, revoked, NULL as key
        FROM api_keys
        WHERE user_id = $1
        ORDER BY created_at DESC
        "#,
    )
    .bind(user_uuid)
    .fetch_all(&pool)
    .await;

    match result {
        Ok(api_keys) => Json(Response {
            status: 200,
            data: Some(api_keys),
        })
        .into_response(),
        Err(e) => {
            eprintln!("Database error: {e}");
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
        r#"
        UPDATE api_keys
        SET revoked = TRUE
        WHERE id = $1 AND user_id = $2
        "#,
    )
    .bind(key_uuid)
    .bind(user_uuid)
    .execute(&pool)
    .await;

    match result {
        Ok(query_result) => {
            if query_result.rows_affected() == 0 {
                (
                    StatusCode::NOT_FOUND,
                    Json(Response {
                        status: 404,
                        data: Some("API key not found or not owned by user".to_string()),
                    }),
                )
                    .into_response()
            } else {
                Json(Response {
                    status: 200,
                    data: Some("API key revoked successfully"),
                })
                .into_response()
            }
        }
        Err(e) => {
            eprintln!("Database error: {e}");
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
