use crate::handlers::{AppError, AppResult, parse_uuid};
use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use crate::types::{ApiKey, CreateApiKeyRequest, Response};
use axum::{
    Extension, Json,
    extract::{Path, State},
    response::IntoResponse,
};
use rand::{Rng, distr::Alphanumeric, rng};
use sha2::{Digest, Sha256};

pub async fn create_api_key(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Json(request): Json<CreateApiKeyRequest>,
) -> AppResult<impl IntoResponse> {
    let key_value: String = rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect();
    let full_key = format!("tora_sk_{key_value}");
    let mut hasher = Sha256::new();
    hasher.update(&full_key);
    let key_hash = format!("{:x}", hasher.finalize());
    let user_uuid = parse_uuid(&user.id, "user_id")?;
    let result = sqlx::query_as::<_, ApiKey>(
        "INSERT INTO api_keys (user_id, name, key_hash) VALUES ($1, $2, $3) RETURNING id::text, name, created_at, revoked, NULL as key"
    )
    .bind(user_uuid)
    .bind(&request.name)
    .bind(&key_hash)
    .fetch_one(&app_state.db_pool)
    .await?;

    let mut api_key = result;
    api_key.key = Some(full_key);
    Ok((
        axum::http::StatusCode::CREATED,
        Json(Response {
            status: 201,
            data: Some(api_key),
        }),
    )
        .into_response())
}

pub async fn list_api_keys(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
) -> AppResult<impl IntoResponse> {
    let user_uuid = parse_uuid(&user.id, "user_id")?;

    let result = sqlx::query_as::<_, ApiKey>(
        r#"
        SELECT id::text, name, created_at, revoked, NULL as key
        FROM api_keys
        WHERE user_id = $1
        ORDER BY created_at DESC
        "#,
    )
    .bind(user_uuid)
    .fetch_all(&app_state.db_pool)
    .await?;

    Ok(Json(Response {
        status: 200,
        data: Some(result),
    })
    .into_response())
}

pub async fn revoke_api_key(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(key_id): Path<String>,
) -> AppResult<impl IntoResponse> {
    let user_uuid = parse_uuid(&user.id, "user_id")?;
    let key_uuid = parse_uuid(&key_id, "key_id")?;

    let result = sqlx::query(
        r#"
        UPDATE api_keys
        SET revoked = TRUE
        WHERE id = $1 AND user_id = $2
        "#,
    )
    .bind(key_uuid)
    .bind(user_uuid)
    .execute(&app_state.db_pool)
    .await?;

    if result.rows_affected() == 0 {
        return Err(AppError::NotFound(
            "API key not found or not owned by user".to_string(),
        ));
    }

    Ok(Json(Response {
        status: 200,
        data: Some("API key revoked successfully"),
    })
    .into_response())
}
