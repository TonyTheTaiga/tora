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
    let key_value = format!("tora_key_{}", Uuid::new_v4().to_string().replace("-", "")[0..32]);
    
    // Create the API key in the database
    let api_key_result = sqlx::query_as::<_, (String, String, chrono::DateTime<chrono::Utc>)>(
        r#"
        INSERT INTO api_key (user_id, name, key_hash, revoked)
        VALUES ($1, $2, $3, false)
        RETURNING id::text, name, created_at
        "#,
    )
    .bind(user_uuid)
    .bind(&request.name)
    .bind(&key_value) // In a real implementation, this should be hashed
    .fetch_one(&pool)
    .await;

    match api_key_result {
        Ok((id, name, created_at)) => {
            let api_key = ApiKey {
                id,
                name,
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
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to create API key".to_string()),
                }),
            )
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

    let api_keys_result = sqlx::query_as::<_, (String, String, chrono::DateTime<chrono::Utc>, bool)>(
        r#"
        SELECT id::text, name, created_at, revoked
        FROM api_key
        WHERE user_id = $1
        ORDER BY created_at DESC
        "#,
    )
    .bind(user_uuid)
    .fetch_all(&pool)
    .await;

    match api_keys_result {
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

    // Check if the API key belongs to the user
    let ownership_check = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM api_key WHERE id = $1 AND user_id = $2",
    )
    .bind(key_uuid)
    .bind(user_uuid)
    .fetch_one(&pool)
    .await;

    match ownership_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::NOT_FOUND,
                Json(Response {
                    status: 404,
                    data: Some("API key not found".to_string()),
                }),
            )
                .into_response();
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to check API key ownership".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    // Revoke the API key
    let revoke_result = sqlx::query(
        "UPDATE api_key SET revoked = true WHERE id = $1",
    )
    .bind(key_uuid)
    .execute(&pool)
    .await;

    match revoke_result {
        Ok(_) => Json(Response {
            status: 200,
            data: Some("API key revoked successfully"),
        }),
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