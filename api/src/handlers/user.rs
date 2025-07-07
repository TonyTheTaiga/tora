use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes;
use axum::{
    Extension, Json,
    extract::Query,
    http::{HeaderMap, StatusCode, header::SET_COOKIE},
    response::{IntoResponse, Redirect},
};
use axum_extra::extract::cookie::{Cookie, SameSite};
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
        "user": session.user
    });

    Json(ntypes::Response {
        status: 200,
        data: Some(session_payload),
    })
}

pub async fn logout(Extension(user): Extension<AuthenticatedUser>) -> impl IntoResponse {
    let auth_client = create_client();

    match auth_client.logout(None, &user.refresh_token).await {
        Ok(_) => {
            let is_production = std::env::var("RUST_ENV")
                .unwrap_or_else(|_| "development".to_string())
                == "production";
            let cookie = Cookie::build(("tora_auth_token", ""))
                .http_only(true)
                .secure(is_production)
                .same_site(SameSite::Lax)
                .path("/")
                .max_age(time::Duration::seconds(0));

            let mut headers = HeaderMap::new();
            headers.insert(SET_COOKIE, cookie.to_string().parse().unwrap());

            let jbody = Json(ntypes::Response::<&str> {
                status: 200,
                data: Some("Logged out successfully"),
            });
            (StatusCode::OK, headers, jbody).into_response()
        }
        Err(err) => {
            let jbody = Json(ntypes::Response {
                status: 500,
                data: Some(serde_json::json!({
                    "error": "Logout failed",
                    "message": err.to_string()
                })),
            });
            (StatusCode::INTERNAL_SERVER_ERROR, jbody).into_response()
        }
    }
}

pub async fn auth_status(
    Extension(user): Extension<AuthenticatedUser>,
) -> Json<ntypes::AuthStatus> {
    Json(ntypes::AuthStatus {
        authenticated: true,
        user: Some(ntypes::UserInfo {
            id: user.id,
            email: user.email,
        }),
    })
}

pub async fn get_settings(
    Extension(user): Extension<AuthenticatedUser>,
) -> Json<ntypes::SettingsData> {
    use crate::repos::workspace::Workspace;

    // Mock data - will be replaced with database queries
    let workspaces = vec![
        Workspace {
            id: "ws_1".to_string(),
            name: "ML Research".to_string(),
            description: Some("Machine learning experiments and research".to_string()),
            created_at: chrono::Utc::now(),
            role: "OWNER".to_string(),
        },
        Workspace {
            id: "ws_2".to_string(),
            name: "NLP Project".to_string(),
            description: Some("Natural language processing experiments".to_string()),
            created_at: chrono::Utc::now(),
            role: "ADMIN".to_string(),
        },
    ];

    let api_keys = vec![
        ntypes::ApiKey {
            id: "key_1".to_string(),
            name: "Development Key".to_string(),
            created_at: "Dec 15".to_string(),
            revoked: false,
            key: None,
        },
        ntypes::ApiKey {
            id: "key_2".to_string(),
            name: "Production Key".to_string(),
            created_at: "Dec 10".to_string(),
            revoked: true,
            key: None,
        },
    ];

    let invitations = vec![ntypes::WorkspaceInvitation {
        id: "inv_1".to_string(),
        workspace_id: "Data Science Team".to_string(),
        email: user.email.clone(),
        role: "ADMIN".to_string(),
        from: "john@example.com".to_string(),
        created_at: chrono::Utc::now(),
    }];

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
