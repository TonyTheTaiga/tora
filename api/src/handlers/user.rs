use crate::ntypes;
use axum::{Json, extract::Query, response::IntoResponse, response::Redirect};
use supabase_auth::models::AuthClient;

pub async fn create_user(Json(payload): Json<ntypes::CreateUser>) -> impl IntoResponse {
    let auth_client = AuthClient::new_from_env().unwrap();
    match auth_client
        .sign_up_with_email_and_password(&payload.email, &payload.password, None)
        .await
    {
        Ok(_response) => Json(ntypes::Response {
            status: 201,
            data: serde_json::json!({
                "message": "User created successfully",
                "email": payload.email
            }),
        }),
        Err(err) => Json(ntypes::Response {
            status: 400,
            data: serde_json::json!({
                "error": "Failed to create user",
                "message": err.to_string()
            }),
        }),
    }
}

pub async fn confirm_create(Query(payload): Query<ntypes::ConfirmQueryParams>) -> Redirect {
    println!("{}", payload.confirm_type);
    Redirect::permanent("/")
}
