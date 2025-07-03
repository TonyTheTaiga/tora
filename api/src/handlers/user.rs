use crate::ntypes;
use axum::{
    Json,
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
    let cookie = Cookie::build(("tora_auth_token", session.refresh_token.clone()))
        .http_only(true)
        .secure(false)
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
