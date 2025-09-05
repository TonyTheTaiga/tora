use crate::state::AppState;
use axum::{
    Router,
    routing::{any, delete, get, post, put},
};

mod api_key;
mod experiment;
mod health;
mod invitation;
mod log;
pub mod result;
mod role;
mod stream;
mod user;
mod workspace;
pub use result::{AppError, AppResult, parse_uuid};

pub fn build_public_routes() -> Router<AppState> {
    Router::new()
        .route("/health", get(health::health))
        .route("/signup", post(user::create_user))
        .route("/signup/confirm", get(user::confirm_create))
        .route("/login", post(user::login))
        .route("/refresh", post(user::refresh_token))
        .route("/workspace-roles", get(role::list_workspace_roles))
        .route(
            "/experiments/{experiment_id}/logs/stream",
            // Streaming is validated via token + origin check
            any(stream::stream_logs),
        )
}

pub fn build_private_routes() -> Router<AppState> {
    let workspaces = Router::new()
        .route(
            "/",
            get(workspace::list_workspaces).post(workspace::create_workspace),
        )
        .route(
            "/{id}",
            get(workspace::get_workspace).delete(workspace::delete_workspace),
        )
        .route("/{id}/leave", post(workspace::leave_workspace))
        .route("/{id}/members", get(workspace::get_workspace_members))
        .route(
            "/{id}/experiments",
            get(experiment::list_workspace_experiments),
        );

    let experiments = Router::new()
        .route(
            "/",
            get(experiment::list_experiments).post(experiment::create_experiment),
        )
        .route("/batch", post(experiment::get_experiments_batch))
        .route(
            "/{id}",
            get(experiment::get_experiment)
                .put(experiment::update_experiment)
                .delete(experiment::delete_experiment),
        )
        .route("/{id}/logs", get(log::get_logs).post(log::create_log))
        .route("/{experiment_id}/logs/batch", post(log::batch_create_logs))
        .route("/{id}/metrics", get(log::get_metrics))
        .route("/{id}/results", get(log::get_results))
        .route("/{id}/logs/csv", get(log::export_logs_csv))
        .route("/{id}/logs/stream-token", post(stream::create_stream_token));

    Router::new()
        .nest("/workspaces", workspaces)
        .nest("/experiments", experiments)
        .route(
            "/api-keys",
            get(api_key::list_api_keys).post(api_key::create_api_key),
        )
        .route("/api-keys/{id}", delete(api_key::revoke_api_key))
        .route(
            "/workspace-invitations",
            get(invitation::list_invitations).post(invitation::create_invitation),
        )
        .route(
            "/workspaces/any/invitations",
            put(invitation::respond_to_invitation),
        )
        .route("/settings", get(user::get_settings))
}
