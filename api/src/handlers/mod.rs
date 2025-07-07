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
mod experiment;
mod invitation;
mod metric;
mod user;

pub fn api_routes() -> Router<sqlx::PgPool> {
    let protected_routes = Router::new()
        // Workspaces
        .route(
            "/workspaces",
            protected_route(get(crate::repos::workspace::list_workspaces)),
        )
        .route(
            "/workspaces",
            protected_route(post(crate::repos::workspace::create_workspace)),
        )
        .route(
            "/workspaces/{id}",
            protected_route(get(crate::repos::workspace::get_workspace)),
        )
        .route(
            "/workspaces/{id}",
            protected_route(delete(crate::repos::workspace::delete_workspace)),
        )
        .route(
            "/workspaces/{id}/leave",
            protected_route(post(crate::repos::workspace::leave_workspace)),
        )
        .route(
            "/workspaces/{id}/members",
            protected_route(get(crate::repos::workspace::get_workspace_members)),
        )
        .route(
            "/workspaces/{id}/experiments",
            protected_route(get(experiment::list_workspace_experiments)),
        )
        // Experiments
        .route(
            "/experiments",
            protected_route(get(experiment::list_experiments)),
        )
        .route(
            "/experiments",
            protected_route(post(experiment::create_experiment)),
        )
        .route(
            "/experiments/{id}",
            protected_route(get(experiment::get_experiment)),
        )
        .route(
            "/experiments/{id}",
            protected_route(put(experiment::update_experiment)),
        )
        .route(
            "/experiments/{id}",
            protected_route(delete(experiment::delete_experiment)),
        )
        // Metrics
        .route(
            "/experiments/{id}/metrics",
            protected_route(get(metric::get_metrics)),
        )
        .route(
            "/experiments/{id}/metrics",
            protected_route(post(metric::create_metric)),
        )
        .route(
            "/experiments/{id}/metrics/batch",
            protected_route(post(metric::batch_create_metrics)),
        )
        .route(
            "/experiments/{id}/metrics/csv",
            protected_route(get(metric::export_metrics_csv)),
        )
        // Settings and user management
        .route("/settings", protected_route(get(user::get_settings)))
        // API Keys
        .route("/api-keys", protected_route(get(api_key::list_api_keys)))
        .route("/api-keys", protected_route(post(api_key::create_api_key)))
        .route(
            "/api-keys/{id}",
            protected_route(delete(api_key::revoke_api_key)),
        )
        // Workspace Invitations
        .route(
            "/workspace-invitations",
            protected_route(post(invitation::create_invitation)),
        )
        .route(
            "/workspace-invitations",
            protected_route(get(invitation::list_invitations)),
        )
        .route(
            "/workspaces/any/invitations",
            protected_route(put(invitation::respond_to_invitation)),
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
