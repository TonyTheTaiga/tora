use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response as AxumResponse},
};
use serde::{Deserialize, Serialize};

// ============================================================================
// Generic API Response Types
// ============================================================================

#[derive(Serialize, Deserialize)]
pub struct Ping {
    pub msg: String,
}

#[derive(Serialize, Deserialize)]
pub struct Response<T> {
    pub status: i16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
}

// ============================================================================
// Authentication-related Request/Response Types
// ============================================================================

#[derive(Deserialize, Serialize)]
pub struct CreateUser {
    pub email: String,
    pub password: String,
}

#[derive(Deserialize, Serialize)]
pub struct ConfirmQueryParams {
    pub token_hash: String,
    pub confirm_type: String,
}

#[derive(Deserialize, Debug)]
pub struct LoginParams {
    pub email: String,
    pub password: String,
}

#[derive(Deserialize, Debug)]
pub struct RefreshTokenRequest {
    pub refresh_token: String,
}

#[derive(Serialize)]
pub struct AuthStatus {
    pub authenticated: bool,
    pub user: Option<crate::types::core::UserInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticatedUser {
    pub id: String,
    pub email: String,
}

// ============================================================================
// Request Types for Various Endpoints
// ============================================================================

#[derive(Deserialize)]
pub struct CreateWorkspaceRequest {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct CreateExperimentRequest {
    pub name: String,
    pub description: String,
    pub workspace_id: String,
    pub tags: Option<Vec<String>>,
    pub hyperparams: Option<Vec<serde_json::Value>>,
}

#[derive(Deserialize)]
pub struct UpdateExperimentRequest {
    pub id: String,
    pub name: String,
    pub description: String,
    pub tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct ListExperimentsQuery {
    pub workspace: Option<String>,
}

#[derive(Deserialize)]
pub struct CreateMetricRequest {
    pub name: String,
    pub value: f64,
    pub step: Option<i64>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct BatchCreateMetricsRequest {
    pub metrics: Vec<CreateMetricRequest>,
}

#[derive(Deserialize)]
pub struct CreateApiKeyRequest {
    pub name: String,
}

#[derive(Deserialize)]
pub struct CreateInvitationRequest {
    #[serde(rename = "workspaceId")]
    pub workspace_id: String,
    pub email: String,
    #[serde(rename = "roleId")]
    pub role_id: String,
}

#[derive(Deserialize)]
pub struct InvitationActionQuery {
    #[serde(rename = "invitationId")]
    pub invitation_id: String,
    pub action: String, // "accept" or "deny"
}

// ============================================================================
// Error Types
// ============================================================================

#[derive(Debug, Clone, Serialize)]
pub struct ErrorResponse {
    pub status: u16,
    pub error: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<String>,
}

#[derive(Debug)]
pub enum AppError {
    BadRequest(String),
    Unauthorized(String),
    Forbidden(String),
    NotFound(String),
    Conflict(String),
    UnprocessableEntity(String),
    Internal(String),
    Database(sqlx::Error),
    Validation(String),
    InvalidUuid(String),
    AuthenticationFailed(String),
}

#[derive(Debug)]
pub struct AuthError {
    pub message: String,
    pub status_code: StatusCode,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> AxumResponse {
        let body = Json(Response {
            status: self.status_code.as_u16() as i16,
            data: Some(serde_json::json!({
                "error": "Authentication failed",
                "message": self.message
            })),
        });
        (self.status_code, body).into_response()
    }
}

pub type AppResult<T> = Result<T, AppError>;
