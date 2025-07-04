use axum::{
    Router,
    routing::{get, post},
};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use crate::middleware::auth::protected_route;

mod ping;
mod user;

pub fn api_routes() -> Router {
    let protected_routes = Router::new()
        .route("/workspaces", protected_route(get(crate::repos::workspace::list_workspaces)))
        .route("/ping", protected_route(post(ping::ping)))
        .route("/logout", protected_route(post(user::logout)))
        .route("/auth/status", protected_route(get(user::auth_status)));

    let public_routes = Router::new()
        .route("/signup", post(user::create_user))
        .route("/signup/confirm", get(user::confirm_create))
        .route("/login", post(user::login));

    Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(CorsLayer::permissive()),
        )
}
