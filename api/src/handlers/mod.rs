use crate::middleware::auth::protected_route;
use crate::state::AppState;
use axum::{
    Router,
    http::{HeaderValue, Method},
    routing::{delete, get, post, put},
};
use http::header::{AUTHORIZATION, CONTENT_TYPE};
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

mod api_key;
mod experiment;
mod invitation;
mod metric;
pub mod result;
mod role;
mod user;
mod workspace;
pub use result::{AppError, AppResult, parse_uuid};

pub fn api_routes(app_state: &AppState) -> Router<AppState> {
    let protected_routes = Router::new()
        .route(
            "/workspaces",
            protected_route(get(workspace::list_workspaces), app_state),
        )
        .route(
            "/workspaces",
            protected_route(post(workspace::create_workspace), app_state),
        )
        .route(
            "/workspaces/{id}",
            protected_route(get(workspace::get_workspace), app_state),
        )
        .route(
            "/workspaces/{id}",
            protected_route(delete(workspace::delete_workspace), app_state),
        )
        .route(
            "/workspaces/{id}/leave",
            protected_route(post(workspace::leave_workspace), app_state),
        )
        .route(
            "/workspaces/{id}/members",
            protected_route(get(workspace::get_workspace_members), app_state),
        )
        .route(
            "/workspaces/{id}/experiments",
            protected_route(get(experiment::list_workspace_experiments), app_state),
        )
        .route(
            "/experiments",
            protected_route(get(experiment::list_experiments), app_state),
        )
        .route(
            "/experiments",
            protected_route(post(experiment::create_experiment), app_state),
        )
        .route(
            "/experiments/{id}",
            protected_route(get(experiment::get_experiment), app_state),
        )
        .route(
            "/experiments/{id}",
            protected_route(put(experiment::update_experiment), app_state),
        )
        .route(
            "/experiments/{id}",
            protected_route(delete(experiment::delete_experiment), app_state),
        )
        .route(
            "/experiments/batch",
            protected_route(post(experiment::get_experiments_batch), app_state),
        )
        .route(
            "/experiments/{id}/metrics",
            protected_route(get(metric::get_metrics), app_state),
        )
        .route(
            "/experiments/{id}/metrics",
            protected_route(post(metric::create_metric), app_state),
        )
        .route(
            "/experiments/{experiment_id}/metrics/batch",
            protected_route(post(metric::batch_create_metrics), app_state),
        )
        .route(
            "/experiments/{id}/metrics/csv",
            protected_route(get(metric::export_metrics_csv), app_state),
        )
        // Settings and user management
        .route(
            "/settings",
            protected_route(get(user::get_settings), app_state),
        )
        .route("/workspace-roles", get(role::list_workspace_roles))
        .route(
            "/api-keys",
            protected_route(get(api_key::list_api_keys), app_state),
        )
        .route(
            "/api-keys",
            protected_route(post(api_key::create_api_key), app_state),
        )
        .route(
            "/api-keys/{id}",
            protected_route(delete(api_key::revoke_api_key), app_state),
        )
        .route(
            "/workspace-invitations",
            protected_route(post(invitation::create_invitation), app_state),
        )
        .route(
            "/workspace-invitations",
            protected_route(get(invitation::list_invitations), app_state),
        )
        .route(
            "/workspaces/any/invitations",
            protected_route(put(invitation::respond_to_invitation), app_state),
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
                .layer(
                    TraceLayer::new_for_http()
                        .make_span_with(|request: &axum::extract::Request| {
                            tracing::info_span!(
                                "http",
                                method = %request.method(),
                                uri = %request.uri(),
                            )
                        })
                        .on_response(
                            |response: &axum::response::Response,
                             latency: std::time::Duration,
                             span: &tracing::Span| {
                                span.in_scope(|| {
                                    tracing::info!(
                                        status = %response.status(),
                                        latency_ms = latency.as_millis(),
                                    );
                                });
                            },
                        ),
                )
                .layer(
                    CorsLayer::new()
                        .allow_origin(
                            app_state
                                .settings
                                .frontend_url
                                .parse::<HeaderValue>()
                                .unwrap(),
                        )
                        .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
                        .allow_headers([AUTHORIZATION, CONTENT_TYPE])
                        .allow_credentials(true),
                ),
        )
}
