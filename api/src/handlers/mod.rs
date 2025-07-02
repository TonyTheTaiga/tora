use axum::{
    Router,
    routing::{get, post},
};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

mod ping;
mod user;

pub fn api_routes() -> Router {
    Router::new()
        .route("/workspaces", get(crate::repos::workspace::list_workspaces))
        .route("/ping", post(ping::ping))
        .route("/signup", post(user::create_user))
        .route("/signup/confirm", get(user::confirm_create))
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive()),
        )
}
