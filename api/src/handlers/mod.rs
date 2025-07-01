use axum::{Router, routing::{post, get}};

mod ping;
mod user;

pub fn api_routes() -> Router {
    Router::new()
        .route("/workspaces", get(crate::repos::workspace::list_workspaces))
        .route("/ping", post(ping::ping))
        .route("/signup", post(user::create_user))
}
