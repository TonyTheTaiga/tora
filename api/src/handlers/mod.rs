use axum::{
    Router,
    routing::{get, post, put},
};

mod ping;
mod user;

pub fn api_routes() -> Router {
    Router::new()
        .route("/workspaces", get(crate::repos::workspace::list_workspaces))
        .route("/ping", post(ping::ping))
        .route("/signup", post(user::sign_up))
        .route("/login", post(user::login))
        .route("/logout", post(user::logout))
        .route("/user", get(user::get_user))
        .route("/user", put(user::update_user))
        .route("/reset-password", post(user::reset_password))
}
