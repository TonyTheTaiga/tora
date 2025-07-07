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

#[derive(Serialize)]
pub struct AuthStatus {
    pub authenticated: bool,
    pub user: Option<UserInfo>,
}

#[derive(Serialize)]
pub struct UserInfo {
    pub id: String,
    pub email: String,
}

#[derive(sqlx::FromRow)]
#[allow(dead_code)]
pub struct ApiKeyRecord {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub key_hash: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_used: chrono::DateTime<chrono::Utc>,
    pub revoked: bool,
    // User info from JOIN
    pub user_email: String,
}
