use axum::{Json, Extension};
use crate::ntypes;
use crate::middleware::auth::AuthenticatedUser;

pub async fn ping(
    Extension(user): Extension<AuthenticatedUser>,
    Json(payload): Json<ntypes::Ping>,
) -> Json<ntypes::Ping> {
    Json(ntypes::Ping {
        msg: format!("pong: {} (authenticated as: {})", payload.msg, user.email),
    })
}
