use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde::Serialize;
use std::fmt;

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

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AppError::BadRequest(msg) => write!(f, "Bad request: {msg}"),
            AppError::Unauthorized(msg) => write!(f, "Unauthorized: {msg}"),
            AppError::Forbidden(msg) => write!(f, "Forbidden: {msg}"),
            AppError::NotFound(msg) => write!(f, "Not found: {msg}"),
            AppError::Conflict(msg) => write!(f, "Conflict: {msg}"),
            AppError::UnprocessableEntity(msg) => write!(f, "Unprocessable entity: {msg}"),
            AppError::Internal(msg) => write!(f, "Internal server error: {msg}"),
            AppError::Database(err) => write!(f, "Database error: {err}"),
            AppError::Validation(msg) => write!(f, "Validation error: {msg}"),
            AppError::InvalidUuid(msg) => write!(f, "Invalid UUID: {msg}"),
            AppError::AuthenticationFailed(msg) => write!(f, "Authentication failed: {msg}"),
        }
    }
}

impl std::error::Error for AppError {}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        let (status, error_type, message, details) = match self {
            AppError::BadRequest(msg) => (StatusCode::BAD_REQUEST, "BAD_REQUEST", msg, None),
            AppError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, "UNAUTHORIZED", msg, None),
            AppError::Forbidden(msg) => (StatusCode::FORBIDDEN, "FORBIDDEN", msg, None),
            AppError::NotFound(msg) => (StatusCode::NOT_FOUND, "NOT_FOUND", msg, None),
            AppError::Conflict(msg) => (StatusCode::CONFLICT, "CONFLICT", msg, None),
            AppError::UnprocessableEntity(msg) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "UNPROCESSABLE_ENTITY",
                msg,
                None,
            ),
            AppError::Internal(msg) => {
                tracing::error!("Internal server error: {}", msg);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "INTERNAL_SERVER_ERROR",
                    "An internal server error occurred".to_string(),
                    Some(msg),
                )
            }
            AppError::Database(err) => {
                tracing::error!("Database error: {}", err);
                match err {
                    sqlx::Error::RowNotFound => (
                        StatusCode::NOT_FOUND,
                        "NOT_FOUND",
                        "Resource not found".to_string(),
                        None,
                    ),
                    sqlx::Error::Database(ref db_err) => {
                        if db_err.is_unique_violation() {
                            (
                                StatusCode::CONFLICT,
                                "CONFLICT",
                                "Resource already exists".to_string(),
                                None,
                            )
                        } else if db_err.is_foreign_key_violation() {
                            (
                                StatusCode::BAD_REQUEST,
                                "BAD_REQUEST",
                                "Invalid reference to related resource".to_string(),
                                None,
                            )
                        } else {
                            (
                                StatusCode::INTERNAL_SERVER_ERROR,
                                "INTERNAL_SERVER_ERROR",
                                "Database operation failed".to_string(),
                                Some(err.to_string()),
                            )
                        }
                    }
                    _ => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        "INTERNAL_SERVER_ERROR",
                        "Database operation failed".to_string(),
                        Some(err.to_string()),
                    ),
                }
            }
            AppError::Validation(msg) => (StatusCode::BAD_REQUEST, "VALIDATION_ERROR", msg, None),
            AppError::InvalidUuid(field) => (
                StatusCode::BAD_REQUEST,
                "INVALID_UUID",
                format!("Invalid UUID format for field: {field}"),
                None,
            ),
            AppError::AuthenticationFailed(msg) => {
                (StatusCode::UNAUTHORIZED, "AUTHENTICATION_FAILED", msg, None)
            }
        };

        let error_response = ErrorResponse {
            status: status.as_u16(),
            error: error_type.to_string(),
            message,
            details,
        };

        (status, Json(error_response)).into_response()
    }
}

impl From<sqlx::Error> for AppError {
    fn from(err: sqlx::Error) -> Self {
        AppError::Database(err)
    }
}

impl From<uuid::Error> for AppError {
    fn from(_: uuid::Error) -> Self {
        AppError::InvalidUuid("Invalid UUID format".to_string())
    }
}

pub type AppResult<T> = Result<T, AppError>;

pub fn parse_uuid(uuid_str: &str, field_name: &str) -> AppResult<uuid::Uuid> {
    uuid::Uuid::parse_str(uuid_str).map_err(|_| AppError::InvalidUuid(field_name.to_string()))
}
