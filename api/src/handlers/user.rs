use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes;
use axum::{
    Extension, Json,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Redirect},
};
use supabase_auth::models::{AuthClient, VerifyOtpParams, VerifyTokenHashParams};
use sqlx::PgPool;

fn create_client() -> AuthClient {
    AuthClient::new_from_env().unwrap()
}

pub async fn create_user(Json(payload): Json<ntypes::CreateUser>) -> impl IntoResponse {
    let auth_client = create_client();
    match auth_client
        .sign_up_with_email_and_password(&payload.email, &payload.password, None)
        .await
    {
        Ok(_response) => Json(ntypes::Response {
            status: 201,
            data: Some(serde_json::json!({
                "message": "User created successfully",
                "email": payload.email
            })),
        }),
        Err(err) => Json(ntypes::Response {
            status: 400,
            data: Some(serde_json::json!({
                "error": "Failed to create user",
                "message": err.to_string()
            })),
        }),
    }
}

pub async fn confirm_create(Query(payload): Query<ntypes::ConfirmQueryParams>) -> Redirect {
    let auth_client = create_client();
    let params = VerifyTokenHashParams {
        token_hash: payload.token_hash,
        otp_type: supabase_auth::models::OtpType::Email,
    };

    auth_client
        .verify_otp(VerifyOtpParams::TokenHash(params))
        .await
        .expect("Failed to verify_otp!");

    Redirect::permanent(
        &std::env::var("REDIRECT_URL_CONFIRM").expect("REDIRECT_URL_CONFIRM not set!"),
    )
}

pub async fn login(Json(payload): Json<ntypes::LoginParams>) -> impl IntoResponse {
    let auth_client = create_client();
    let session = auth_client
        .login_with_email(&payload.email, &payload.password)
        .await
        .expect("Failed to login user! Double check password and email.");

    // Return tokens in response body for SSR
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

    Json(ntypes::Response {
        status: 200,
        data: Some(session_payload),
    })
}

pub async fn refresh_token(Json(payload): Json<ntypes::RefreshTokenRequest>) -> impl IntoResponse {
    let auth_client = create_client();

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

            Json(ntypes::Response {
                status: 200,
                data: Some(session_payload),
            })
            .into_response()
        }
        Err(_) => (
            StatusCode::UNAUTHORIZED,
            Json(ntypes::Response::<&str> {
                status: 401,
                data: None,
            }),
        )
            .into_response(),
    }
}

pub async fn get_settings(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
) -> impl IntoResponse {
    use crate::repos::workspace::Workspace;
    use crate::ntypes::{ApiKey, WorkspaceInvitation, SettingsData, UserInfo, Response};
    use sqlx::PgPool;
    use uuid::Uuid;

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

    // Fetch user's workspaces
    let workspaces_result = sqlx::query_as::<_, (String, String, Option<String>, chrono::DateTime<chrono::Utc>, String)>(
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
    .fetch_all(&pool)
    .await;

    // Fetch user's API keys
    let api_keys_result = sqlx::query_as::<_, (String, String, chrono::DateTime<chrono::Utc>, bool)>(
        "SELECT id::text, name, created_at, revoked FROM api_keys WHERE user_id = $1 ORDER BY created_at DESC",
    )
    .bind(user_uuid)
    .fetch_all(&pool)
    .await;

    // Fetch pending invitations for the user
    let invitations_result = sqlx::query_as::<_, (String, String, String, String, chrono::DateTime<chrono::Utc>)>(
        r#"
        SELECT wi.id::text, w.name as workspace_name, u.email as from_email, wr.name as role_name, wi.created_at
        FROM workspace_invitations wi
        JOIN workspace w ON wi.workspace_id = w.id
        JOIN auth.users u ON wi."from" = u.id
        JOIN workspace_role wr ON wi.role_id = wr.id
        WHERE wi."to" = $1 AND wi.status = 'pending'
        ORDER BY wi.created_at DESC
        "#,
    )
    .bind(user_uuid)
    .fetch_all(&pool)
    .await;

    match (workspaces_result, api_keys_result, invitations_result) {
        (Ok(workspace_rows), Ok(api_key_rows), Ok(invitation_rows)) => {
            let workspaces: Vec<Workspace> = workspace_rows
                .into_iter()
                .map(|(id, name, description, created_at, role)| Workspace {
                    id,
                    name,
                    description,
                    created_at,
                    role,
                })
                .collect();

            let api_keys: Vec<ApiKey> = api_key_rows
                .into_iter()
                .map(|(id, name, created_at, revoked)| ApiKey {
                    id,
                    name,
                    created_at: created_at.format("%b %d").to_string(),
                    revoked,
                    key: None, // Never include actual key in settings
                })
                .collect();

            let invitations: Vec<WorkspaceInvitation> = invitation_rows
                .into_iter()
                .map(|(id, workspace_name, from_email, role_name, created_at)| {
                    WorkspaceInvitation {
                        id,
                        workspace_id: workspace_name,
                        email: user.email.clone(),
                        role: role_name,
                        from: from_email,
                        created_at,
                    }
                })
                .collect();

            let settings_data = SettingsData {
                user: UserInfo {
                    id: user.id,
                    email: user.email,
                },
                workspaces,
                api_keys,
                invitations,
            };

            Json(Response {
                status: 200,
                data: Some(settings_data),
            })
                .into_response()
        }
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to fetch settings data".to_string()),
                }),
            )
                .into_response()
        }
    }
}
