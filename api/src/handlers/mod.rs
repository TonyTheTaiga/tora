use crate::middleware::auth::protected_route;
use axum::{
    Router,
    http::{HeaderValue, Method},
    routing::{delete, get, post, put},
};
use http::header::{AUTHORIZATION, CONTENT_TYPE};
use std::env;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

mod api_key;
mod dashboard;
mod experiment;
mod invitation;
mod metric;
mod role;
mod user;
mod workspace;

pub fn api_routes(pool: &sqlx::PgPool) -> Router<sqlx::PgPool> {
    let protected_routes = Router::new()
        // Workspaces
        .route(
            "/workspaces",
            protected_route(get(workspace::list_workspaces), pool),
        )
        .route(
            "/workspaces",
            protected_route(post(workspace::create_workspace), pool),
        )
        .route(
            "/workspaces/{id}",
            protected_route(get(workspace::get_workspace), pool),
        )
        .route(
            "/workspaces/{id}",
            protected_route(delete(workspace::delete_workspace), pool),
        )
        .route(
            "/workspaces/{id}/leave",
            protected_route(post(workspace::leave_workspace), pool),
        )
        .route(
            "/workspaces/{id}/members",
            protected_route(get(workspace::get_workspace_members), pool),
        )
        .route(
            "/workspaces/{id}/experiments",
            protected_route(get(experiment::list_workspace_experiments), pool),
        )
        // Experiments
        .route(
            "/experiments",
            protected_route(get(experiment::list_experiments), pool),
        )
        .route(
            "/experiments",
            protected_route(post(experiment::create_experiment), pool),
        )
        .route(
            "/experiments/{id}",
            protected_route(get(experiment::get_experiment), pool),
        )
        .route(
            "/experiments/{id}",
            protected_route(put(experiment::update_experiment), pool),
        )
        .route(
            "/experiments/{id}",
            protected_route(delete(experiment::delete_experiment), pool),
        )
        // Metrics
        .route(
            "/experiments/{id}/metrics",
            protected_route(get(metric::get_metrics), pool),
        )
        .route(
            "/experiments/{id}/metrics",
            protected_route(post(metric::create_metric), pool),
        )
        .route(
            "/experiments/{experiment_id}/metrics/batch",
            protected_route(post(metric::batch_create_metrics), pool),
        )
        .route(
            "/experiments/{id}/metrics/csv",
            protected_route(get(metric::export_metrics_csv), pool),
        )
        // Dashboard
        .route(
            "/dashboard/overview",
            protected_route(get(dashboard::get_dashboard_overview), pool),
        )
        // Settings and user management
        .route("/settings", protected_route(get(user::get_settings), pool))
        .route("/workspace-roles", get(role::list_workspace_roles))
        // API Keys
        .route(
            "/api-keys",
            protected_route(get(api_key::list_api_keys), pool),
        )
        .route(
            "/api-keys",
            protected_route(post(api_key::create_api_key), pool),
        )
        .route(
            "/api-keys/{id}",
            protected_route(delete(api_key::revoke_api_key), pool),
        )
        .route(
            "/workspace-invitations",
            protected_route(post(invitation::create_invitation), pool),
        )
        .route(
            "/workspace-invitations",
            protected_route(get(invitation::list_invitations), pool),
        )
        .route(
            "/workspaces/any/invitations",
            protected_route(put(invitation::respond_to_invitation), pool),
        );

    let public_routes = Router::new()
        .route("/signup", post(user::create_user))
        .route("/signup/confirm", get(user::confirm_create))
        .route("/login", post(user::login))
        .route("/refresh", post(user::refresh_token));

    Router::new()
        .merge(protected_routes)
        .merge(public_routes)
        .layer(
            ServiceBuilder::new()
                .layer(TraceLayer::new_for_http())
                .layer(
                    CorsLayer::new()
                        .allow_origin(
                            env::var("FRONTEND_URL")
                                .expect("FRONTEND_URL not set!")
                                .parse::<HeaderValue>()
                                .unwrap(),
                        )
                        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
                        .allow_headers([AUTHORIZATION, CONTENT_TYPE])
                        .allow_credentials(true),
                ),
        )
}
