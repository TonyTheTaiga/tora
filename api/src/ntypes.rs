use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Ping {
    pub msg: String,
}

#[derive(Serialize)]
pub struct Response<T> {
    pub status: i16,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
}

#[derive(Deserialize, Serialize)]
pub struct CreateUser {
    pub email: String,
    pub password: String,
}

#[derive(Deserialize, Serialize)]
pub struct ConfirmQueryParams {
    pub token_hash: String,
    pub confirm_type: String,
}

#[derive(Deserialize, Debug)]
pub struct LoginParams {
    pub email: String,
    pub password: String,
}
