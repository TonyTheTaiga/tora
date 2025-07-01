use axum::{response::IntoResponse, Json};
use crate::ntypes;

pub async fn create_user(Json(payload): Json<ntypes::CreateUser>) -> impl IntoResponse {
    println!("{:?}", payload.email);
    println!("{:?}", payload.password);

    Json(ntypes::Response {
        status: 200,
        data: payload,
    })
}
