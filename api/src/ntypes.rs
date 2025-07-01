use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Ping {
    pub msg: String,
}

#[derive(Serialize)]
pub struct Response<T> {
    pub status: i16,
    pub data: T,
}

#[derive(Deserialize, Serialize)]
pub struct CreateUser {
    pub email: String,
    pub password: String,
}
