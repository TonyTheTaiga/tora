use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Ping {
    pub msg: String,
}

#[derive(Serialize, Deserialize)]
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

#[derive(Deserialize, Debug)]
pub struct RefreshTokenRequest {
    pub refresh_token: String,
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

#[derive(sqlx::FromRow, Debug)]
#[allow(dead_code)]
pub struct ApiKeyRecord {
    pub id: String,
    pub user_id: String,
    pub name: String,
    pub key_hash: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_used: chrono::DateTime<chrono::Utc>,
    pub revoked: bool,
    pub user_email: String,
}

#[derive(Serialize, Deserialize, sqlx::FromRow, Debug)]
pub struct ApiKey {
    pub id: String,
    pub name: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub revoked: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[sqlx(skip)]
    pub key: Option<String>,
}

#[derive(Deserialize)]
pub struct CreateApiKeyRequest {
    pub name: String,
}

#[derive(Serialize, Deserialize, sqlx::FromRow)]
pub struct WorkspaceInvitation {
    pub id: String,
    pub workspace_id: String,
    pub email: String,
    pub role: String,
    pub from: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Deserialize)]
pub struct CreateInvitationRequest {
    #[serde(rename = "workspaceId")]
    pub workspace_id: String,
    pub email: String,
    #[serde(rename = "roleId")]
    pub role_id: String,
}

#[derive(Serialize)]
pub struct SettingsData {
    pub user: UserInfo,
    pub workspaces: Vec<crate::handlers::Workspace>,
    #[serde(rename = "apiKeys")]
    pub api_keys: Vec<ApiKey>,
    pub invitations: Vec<WorkspaceInvitation>,
}

#[derive(Serialize, Deserialize, sqlx::FromRow)]
pub struct WorkspaceRole {
    pub id: String,
    pub name: String,
}
