use axum::{
    Extension, Json,
    extract::Path,
    http::StatusCode,
    response::IntoResponse,
};
use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes::{ApiKey, CreateApiKeyRequest, Response};

pub async fn create_api_key(
    Extension(_user): Extension<AuthenticatedUser>,
    Json(request): Json<CreateApiKeyRequest>,
) -> impl IntoResponse {
    // Generate a mock API key
    let key_value = format!("tora_key_{}", uuid::Uuid::new_v4().to_string().replace("-", "")[0..16].to_string());
    
    let api_key = ApiKey {
        id: format!("key_{}", uuid::Uuid::new_v4()),
        name: request.name,
        created_at: chrono::Utc::now().format("%b %d").to_string(),
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

pub async fn list_api_keys(
    Extension(_user): Extension<AuthenticatedUser>,
) -> impl IntoResponse {
    // Mock data - will be replaced with database query
    let api_keys = vec![
        ApiKey {
            id: "key_1".to_string(),
            name: "Development Key".to_string(),
            created_at: "Dec 15".to_string(),
            revoked: false,
            key: None, // Never include actual key in list
        },
        ApiKey {
            id: "key_2".to_string(),
            name: "Production Key".to_string(),
            created_at: "Dec 10".to_string(),
            revoked: true,
            key: None,
        },
    ];

    Json(Response {
        status: 200,
        data: Some(api_keys),
    })
}

pub async fn revoke_api_key(
    Extension(_user): Extension<AuthenticatedUser>,
    Path(key_id): Path<String>,
) -> impl IntoResponse {
    // Mock - will be replaced with database update
    println!("Revoking API key: {}", key_id);

    Json(Response {
        status: 200,
        data: Some("API key revoked successfully"),
    })
}