use axum::{
    extract::Json,
    http::{HeaderMap, StatusCode, header::AUTHORIZATION},
    response::IntoResponse,
};
use serde::Deserialize;
use serde_json::json;
use supabase_auth::models::{AuthClient, LogoutScope, UpdatedUser};

use crate::ntypes;

fn client() -> Result<AuthClient, supabase_auth::error::Error> {
    AuthClient::new_from_env()
}

pub async fn sign_up(
    Json(payload): Json<ntypes::Credentials>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let c = client().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;
    match c
        .sign_up_with_email_and_password(&payload.email, &payload.password, None)
        .await
    {
        Ok(res) => {
            let value = match res {
                supabase_auth::models::EmailSignUpResult::SessionResult(s) => {
                    serde_json::to_value(s).unwrap()
                }
                supabase_auth::models::EmailSignUpResult::ConfirmationResult(_) => {
                    json!({"confirmation": true})
                }
            };
            Ok(Json(value))
        }
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

pub async fn login(
    Json(payload): Json<ntypes::Credentials>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let c = client().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;
    match c.login_with_email(&payload.email, &payload.password).await {
        Ok(session) => Ok(Json(serde_json::to_value(session).unwrap())),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

fn bearer_token(headers: &HeaderMap) -> Option<String> {
    headers
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|s| s.to_string())
}

pub async fn logout(headers: HeaderMap) -> impl IntoResponse {
    if let Some(token) = bearer_token(&headers) {
        match client() {
            Ok(c) => match c.logout(Some(LogoutScope::Global), &token).await {
                Ok(_) => (StatusCode::OK, Json(json!({"success": true}))),
                Err(e) => (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": e.to_string()})),
                ),
            },
            Err(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            ),
        }
    } else {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "missing bearer token"})),
        )
    }
}

pub async fn get_user(
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if let Some(token) = bearer_token(&headers) {
        match client() {
            Ok(c) => match c.get_user(&token).await {
                Ok(user) => Ok(Json(serde_json::to_value(user).unwrap())),
                Err(e) => Err((
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": e.to_string()})),
                )),
            },
            Err(e) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )),
        }
    } else {
        Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "missing bearer token"})),
        ))
    }
}

pub async fn update_user(
    headers: HeaderMap,
    Json(payload): Json<UpdatedUser>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if let Some(token) = bearer_token(&headers) {
        match client() {
            Ok(c) => match c.update_user(payload, &token).await {
                Ok(user) => Ok(Json(serde_json::to_value(user).unwrap())),
                Err(e) => Err((
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": e.to_string()})),
                )),
            },
            Err(e) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )),
        }
    } else {
        Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "missing bearer token"})),
        ))
    }
}

#[derive(Deserialize)]
pub struct ResetRequest {
    pub email: String,
}

pub async fn reset_password(
    Json(req): Json<ResetRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let c = client().map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;
    match c.reset_password_for_email(&req.email, None).await {
        Ok(_) => Ok(Json(json!({"success": true}))),
        Err(e) => Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": e.to_string()})),
        )),
    }
}
