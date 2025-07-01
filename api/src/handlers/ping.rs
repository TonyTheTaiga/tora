use axum::Json;
use crate::ntypes;

pub async fn ping(Json(payload): Json<ntypes::Ping>) -> Json<ntypes::Ping> {
    Json(ntypes::Ping {
        msg: format!("pong: {}", payload.msg),
    })
}
