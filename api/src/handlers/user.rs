use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes;
use axum::{
    Extension, Json,
    extract::Query,
    http::StatusCode,
    response::{IntoResponse, Redirect},
    State,
};
use supabase_auth::models::{AuthClient, VerifyOtpParams, VerifyTokenHashParams};

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
    State(pool): State<sqlx::PgPool>,
) -> impl IntoResponse {
    let user_uuid = match uuid::Uuid::parse_str(&user.id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(ntypes::Response {
                    status: 400,
                    data: Some("Invalid user ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Fetch user's workspaces
    let workspaces_result = sqlx::query_as::<
        _,
        (
            String,
            String,
            Option<String>,
            chrono::DateTime<chrono::Utc>,
            String,
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
    .fetch_all(&pool)
    .await;

    let workspaces = match workspaces_result {
        Ok(rows) => rows
            .into_iter()
            .map(|(id, name, description, created_at, role)| crate::repos::workspace::Workspace {
                id,
                name,
                description,
                created_at,
                role,
            })
            .collect(),
        Err(e) => {
            eprintln!("Database error fetching workspaces: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ntypes::Response {
                    status: 500,
                    data: Some("Failed to fetch workspaces".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Fetch user's API keys
    let api_keys_result = sqlx::query_as::<
        _,
        (String, String, chrono::DateTime<chrono::Utc>, bool),
    >(
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

    let api_keys = match api_keys_result {
        Ok(rows) => rows
            .into_iter()
            .map(|(id, name, created_at, revoked)| ntypes::ApiKey {
                id,
                name,
                created_at: created_at.format("%b %d").to_string(),
                revoked,
                key: None, // Never return the actual key
            })
            .collect(),
        Err(e) => {
            eprintln!("Database error fetching API keys: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ntypes::Response {
                    status: 500,
                    data: Some("Failed to fetch API keys".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Fetch pending invitations for the user
    let invitations_result = sqlx::query_as::<
        _,
        (
            String,
            String,
            String,
            String,
            String,
            chrono::DateTime<chrono::Utc>,
        ),
    >(
        r#"
        SELECT 
            wi.id::text,
            w.name as workspace_name,
            wi.email,
            wr.name as role,
            u.email as from_email,
            wi.created_at
        FROM workspace_invitation wi
        JOIN workspace w ON wi.workspace_id = w.id
        JOIN workspace_role wr ON wi.role_id = wr.id
        JOIN "user" u ON wi.invited_by = u.id
        WHERE wi.email = $1 AND wi.status = 'PENDING'
        ORDER BY wi.created_at DESC
        "#,
    )
    .bind(&user.email)
    .fetch_all(&pool)
    .await;

    let invitations = match invitations_result {
        Ok(rows) => rows
            .into_iter()
            .map(|(id, workspace_name, email, role, from_email, created_at)| {
                ntypes::WorkspaceInvitation {
                    id,
                    workspace_id: workspace_name,
                    email,
                    role,
                    from: from_email,
                    created_at,
                }
            })
            .collect(),
        Err(e) => {
            eprintln!("Database error fetching invitations: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ntypes::Response {
                    status: 500,
                    data: Some("Failed to fetch invitations".to_string()),
                }),
            )
                .into_response();
        }
    };

    Json(ntypes::Response {
        status: 200,
        data: Some(ntypes::SettingsData {
            user: ntypes::UserInfo {
                id: user.id,
                email: user.email,
            },
            workspaces,
            api_keys,
            invitations,
        }),
    })
}
