use serde::Deserialize;
use axum::Extension;
use crate::middleware::auth::AuthenticatedUser;

#[derive(Deserialize)]
struct Workspace {
    name: String,
}

pub async fn list_workspaces(Extension(user): Extension<AuthenticatedUser>) -> String {
    format!("list workspaces for user: {}", user.email)
}

