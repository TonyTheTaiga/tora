use crate::handlers::{api_key, invitation, workspace};
use crate::middleware::auth::AuthenticatedUser;
use crate::settings::Settings;
use crate::state::AppState;
use crate::types;
use axum::{
    Extension, Json,
    body::to_bytes,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use supabase_auth::models::{AuthClient, VerifyOtpParams, VerifyTokenHashParams};

fn create_client(settings: &Settings) -> AuthClient {
    AuthClient::new(
        &settings.supabase_url,
        &settings.supabase_api_key,
        &settings.supabase_jwt_secret,
    )
}

pub async fn create_user(
    State(app_state): State<AppState>,
    Json(payload): Json<types::CreateUser>,
) -> impl IntoResponse {
    let auth_client = create_client(&app_state.settings);
    match auth_client
        .sign_up_with_email_and_password(&payload.email, &payload.password, None)
        .await
    {
        Ok(_response) => Json(types::Response {
            status: 201,
            data: Some(serde_json::json!({
                "message": "User created successfully",
                "email": payload.email
            })),
        }),
        Err(err) => Json(types::Response {
            status: 400,
            data: Some(serde_json::json!({
                "error": "Failed to create user",
                "message": err.to_string()
            })),
        }),
    }
}

pub async fn confirm_create(
    State(app_state): State<AppState>,
    Query(payload): Query<types::ConfirmQueryParams>,
) -> impl IntoResponse {
    let auth_client = create_client(&app_state.settings);
    let params = VerifyTokenHashParams {
        token_hash: payload.token_hash.clone(),
        otp_type: supabase_auth::models::OtpType::Email,
    };
    match auth_client
        .verify_otp(VerifyOtpParams::TokenHash(params))
        .await
    {
        Ok(_response) => Json(types::Response {
            status: 200,
            data: Some("Sucecss".into()),
        }),
        Err(err) => Json(types::Response {
            status: 400,
            data: Some(serde_json::json!({
                "error" : "Failed to confirm signup",
                "message": err.to_string()
            })),
        }),
    }
}

pub async fn login(
    State(app_state): State<AppState>,
    Json(payload): Json<types::LoginParams>,
) -> impl IntoResponse {
    let auth_client = create_client(&app_state.settings);
    let session = auth_client
        .login_with_email(&payload.email, &payload.password)
        .await
        .expect("Failed to login user! Double check password and email.");

    let session_payload = serde_json::json!({
        "access_token": session.access_token,
        "token_type": "bearer",
        "expires_in": session.expires_in,
        "expires_at": session.expires_at,
        "refresh_token": session.refresh_token,
        "user": {
            "id": session.user.id,
            "email": session.user.email,
        }
    });

    Json(types::Response {
        status: 200,
        data: Some(session_payload),
    })
}

pub async fn refresh_token(
    State(app_state): State<AppState>,
    Json(payload): Json<types::RefreshTokenRequest>,
) -> impl IntoResponse {
    let auth_client = create_client(&app_state.settings);
    match auth_client.refresh_session(&payload.refresh_token).await {
        Ok(session) => {
            let session_payload = serde_json::json!({
                "access_token": session.access_token,
                "token_type": "bearer",
                "expires_in": session.expires_in,
                "expires_at": session.expires_at,
                "refresh_token": session.refresh_token,
                "user": {
                    "id": session.user.id,
                    "email": session.user.email,
                }
            });

            Json(types::Response {
                status: 200,
                data: Some(session_payload),
            })
            .into_response()
        }
        Err(_) => (
            StatusCode::UNAUTHORIZED,
            Json(types::Response::<&str> {
                status: 401,
                data: Some("Failed to refresh token"),
            }),
        )
            .into_response(),
    }
}

pub async fn get_settings(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
) -> Json<types::SettingsData> {
    let workspaces_response =
        workspace::list_workspaces(Extension(user.clone()), State(app_state.clone()))
            .await
            .into_response();
    let workspaces: Vec<types::Workspace> = if workspaces_response.status().is_success() {
        let body = to_bytes(workspaces_response.into_body(), usize::MAX)
            .await
            .unwrap();
        let response_data: types::Response<Vec<types::Workspace>> =
            serde_json::from_slice(&body).unwrap();
        response_data.data.unwrap_or_default()
    } else {
        vec![]
    };

    let api_keys_response =
        api_key::list_api_keys(Extension(user.clone()), State(app_state.clone()))
            .await
            .into_response();
    let api_keys: Vec<types::ApiKey> = if api_keys_response.status().is_success() {
        let body = to_bytes(api_keys_response.into_body(), usize::MAX)
            .await
            .unwrap();
        let response_data: types::Response<Vec<types::ApiKey>> =
            serde_json::from_slice(&body).unwrap();
        response_data.data.unwrap_or_default()
    } else {
        vec![]
    };

    let invitations_response =
        invitation::list_invitations(Extension(user.clone()), State(app_state.clone()))
            .await
            .into_response();
    let invitations: Vec<types::WorkspaceInvitation> = if invitations_response.status().is_success()
    {
        let body = to_bytes(invitations_response.into_body(), usize::MAX)
            .await
            .unwrap();
        let response_data: types::Response<Vec<types::WorkspaceInvitation>> =
            serde_json::from_slice(&body).unwrap();
        response_data.data.unwrap_or_default()
    } else {
        vec![]
    };

    Json(types::SettingsData {
        user: types::UserInfo {
            id: user.id,
            email: user.email,
        },
        workspaces,
        api_keys,
        invitations,
    })
}
