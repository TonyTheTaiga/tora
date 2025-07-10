use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes;
use axum::{
    Extension, Json,
    body::to_bytes,
    extract::{Query, State},
    http::StatusCode,
    response::{IntoResponse, Redirect},
};
use sqlx::PgPool;
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
    State(pool): State<PgPool>,
) -> Json<ntypes::SettingsData> {
    use crate::handlers::{api_key, invitation};
    use crate::repos::workspace;

    println!("user: {user:?}");

    let workspaces_response =
        workspace::list_workspaces(Extension(user.clone()), State(pool.clone()))
            .await
            .into_response();
    let workspaces: Vec<workspace::Workspace> = if workspaces_response.status().is_success() {
        let body = to_bytes(workspaces_response.into_body(), usize::MAX)
            .await
            .unwrap();
        let response_data: ntypes::Response<Vec<workspace::Workspace>> =
            serde_json::from_slice(&body).unwrap();
        response_data.data.unwrap_or_default()
    } else {
        vec![]
    };

    let api_keys_response = api_key::list_api_keys(Extension(user.clone()), State(pool.clone()))
        .await
        .into_response();
    let api_keys: Vec<ntypes::ApiKey> = if api_keys_response.status().is_success() {
        let body = to_bytes(api_keys_response.into_body(), usize::MAX)
            .await
            .unwrap();
        let response_data: ntypes::Response<Vec<ntypes::ApiKey>> =
            serde_json::from_slice(&body).unwrap();
        response_data.data.unwrap_or_default()
    } else {
        vec![]
    };

    let invitations_response =
        invitation::list_invitations(Extension(user.clone()), State(pool.clone()))
            .await
            .into_response();
    let invitations: Vec<ntypes::WorkspaceInvitation> =
        if invitations_response.status().is_success() {
            let body = to_bytes(invitations_response.into_body(), usize::MAX)
                .await
                .unwrap();
            let response_data: ntypes::Response<Vec<ntypes::WorkspaceInvitation>> =
                serde_json::from_slice(&body).unwrap();
            response_data.data.unwrap_or_default()
        } else {
            vec![]
        };

    Json(ntypes::SettingsData {
        user: ntypes::UserInfo {
            id: user.id,
            email: user.email,
        },
        workspaces,
        api_keys,
        invitations,
    })
}
