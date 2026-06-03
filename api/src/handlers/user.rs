use crate::handlers::{AppError, AppResult, parse_uuid};
use crate::middleware::auth::AuthenticatedUser;
use crate::settings::Settings;
use crate::state::AppState;
use crate::types;
use axum::{
    Extension, Json,
    extract::{Query, State},
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
) -> AppResult<impl IntoResponse> {
    let auth_client = create_client(&app_state.settings);
    auth_client
        .sign_up_with_email_and_password(&payload.email, &payload.password, None)
        .await
        .map_err(|e| AppError::BadRequest(format!("Failed to create user: {e}")))?;

    Ok(Json(types::Response {
        status: 201,
        data: Some(serde_json::json!({
            "message": "User created successfully",
            "email": payload.email
        })),
    }))
}

pub async fn confirm_create(
    State(app_state): State<AppState>,
    Query(payload): Query<types::ConfirmQueryParams>,
) -> AppResult<impl IntoResponse> {
    let auth_client = create_client(&app_state.settings);
    let params = VerifyTokenHashParams {
        token_hash: payload.token_hash.clone(),
        otp_type: supabase_auth::models::OtpType::Email,
    };
    auth_client
        .verify_otp(VerifyOtpParams::TokenHash(params))
        .await
        .map_err(|e| AppError::BadRequest(format!("Failed to confirm signup: {e}")))?;

    Ok(Json(types::Response::<String> {
        status: 200,
        data: Some("Sucecss".into()),
    }))
}

pub async fn login(
    State(app_state): State<AppState>,
    Json(payload): Json<types::LoginParams>,
) -> AppResult<impl IntoResponse> {
    let auth_client = create_client(&app_state.settings);
    let session = auth_client
        .login_with_email(&payload.email, &payload.password)
        .await
        .map_err(|_| AppError::AuthenticationFailed("Invalid email or password".to_string()))?;

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

    Ok(Json(types::Response {
        status: 200,
        data: Some(session_payload),
    }))
}

pub async fn refresh_token(
    State(app_state): State<AppState>,
    Json(payload): Json<types::RefreshTokenRequest>,
) -> AppResult<impl IntoResponse> {
    let auth_client = create_client(&app_state.settings);
    let session = auth_client
        .refresh_session(&payload.refresh_token)
        .await
        .map_err(|_| AppError::AuthenticationFailed("Failed to refresh token".to_string()))?;

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

    Ok(Json(types::Response {
        status: 200,
        data: Some(session_payload),
    })
    .into_response())
}

pub async fn get_settings(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
) -> Json<types::SettingsData> {
    let user_uuid = parse_uuid(&user.id, "user_id").unwrap();
    let pool = &app_state.db_pool;

    let (workspaces_result, api_keys_result, invitations_result) = tokio::join!(
        sqlx::query_as::<
            _,
            (
                String,
                String,
                Option<String>,
                chrono::DateTime<chrono::Utc>,
                String
            ),
        >(
            r#"
            SELECT w.id::text, w.name, w.description, w.created_at, wr.name as role
            FROM workspace w
            JOIN user_workspaces uw ON w.id = uw.workspace_id
            JOIN workspace_role wr ON uw.role_id = wr.id
            WHERE uw.user_id = $1
            ORDER BY w.created_at DESC
            "#,
        )
        .bind(user_uuid)
        .fetch_all(pool),
        sqlx::query_as::<_, types::ApiKey>(
            r#"
            SELECT id::text, name, created_at, revoked, NULL as key
            FROM api_keys
            WHERE user_id = $1
            ORDER BY created_at DESC
            "#,
        )
        .bind(user_uuid)
        .fetch_all(pool),
        sqlx::query_as::<_, types::WorkspaceInvitation>(
            r#"
            SELECT
                wi.id::text,
                w.id::text as workspace_id,
                w.name as workspace_name,
                u_to.email as email,
                wr.name as role,
                u_from.email as from,
                wi.created_at
            FROM workspace_invitations wi
            JOIN workspace w ON wi.workspace_id = w.id
            JOIN workspace_role wr ON wi.role_id = wr.id
            JOIN auth.users u_to ON wi.to = u_to.id
            JOIN auth.users u_from ON wi.from = u_from.id
            WHERE wi.to = $1 AND wi.status = 'PENDING'
            ORDER BY wi.created_at DESC
            "#,
        )
        .bind(user_uuid)
        .fetch_all(pool),
    );

    let workspaces = workspaces_result
        .unwrap_or_default()
        .into_iter()
        .map(
            |(id, name, description, created_at, role)| types::Workspace {
                id,
                name,
                description,
                created_at,
                role,
            },
        )
        .collect();

    let api_keys = api_keys_result.unwrap_or_default();
    let invitations = invitations_result.unwrap_or_default();

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
