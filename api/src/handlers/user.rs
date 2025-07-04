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
    let is_production =
        std::env::var("RUST_ENV").unwrap_or_else(|_| "development".to_string()) == "production";
    let cookie = Cookie::build(("tora_auth_token", session.refresh_token.clone()))
        .http_only(true)
        .secure(is_production)
        .same_site(SameSite::Lax)
        .path("/");

    let mut headers = HeaderMap::new();
    headers.insert(SET_COOKIE, cookie.to_string().parse().unwrap());
    let jbody = Json(ntypes::Response::<&str> {
        status: 200,
        data: None,
    });
    (StatusCode::OK, headers, jbody).into_response()
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
