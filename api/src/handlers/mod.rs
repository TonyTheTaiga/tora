use axum::{
    Router,
    routing::{get, post, put, delete},
};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use crate::middleware::auth::protected_route;

mod ping;
mod user;
mod experiment;
mod metric;

pub fn api_routes() -> Router {
    let protected_routes = Router::new()
        // Workspaces
        .route("/workspaces", protected_route(get(crate::repos::workspace::list_workspaces)))
        .route("/workspaces", protected_route(post(crate::repos::workspace::create_workspace)))
        .route("/workspaces/{id}", protected_route(get(crate::repos::workspace::get_workspace)))
        .route("/workspaces/{id}/members", protected_route(get(crate::repos::workspace::get_workspace_members)))
        
        // Experiments
        .route("/experiments", protected_route(get(experiment::list_experiments)))
        .route("/experiments", protected_route(post(experiment::create_experiment)))
        .route("/experiments/{id}", protected_route(get(experiment::get_experiment)))
        .route("/experiments/{id}", protected_route(put(experiment::update_experiment)))
        .route("/experiments/{id}", protected_route(delete(experiment::delete_experiment)))
        
        // Metrics
        .route("/experiments/{id}/metrics", protected_route(get(metric::get_metrics)))
        .route("/experiments/{id}/metrics", protected_route(post(metric::create_metric)))
        .route("/experiments/{id}/metrics/batch", protected_route(post(metric::batch_create_metrics)))
        .route("/experiments/{id}/metrics/csv", protected_route(get(metric::export_metrics_csv)))
        
        // Other protected routes
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
