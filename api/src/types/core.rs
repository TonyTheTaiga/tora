use chrono;
use serde::{Deserialize, Serialize};
use sqlx;

// ============================================================================
// User-related types
// ============================================================================

#[derive(Serialize)]
pub struct UserInfo {
    pub id: String,
    pub email: String,
}

// ============================================================================
// Workspace-related types
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct Workspace {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    #[serde(rename = "createdAt")]
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub role: String,
}

#[derive(Serialize)]
pub struct WorkspaceMember {
    pub id: String,
    pub email: String,
    pub role: String,
    pub joined_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Serialize, Deserialize, sqlx::FromRow)]
pub struct WorkspaceRole {
    pub id: String,
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct WorkspaceSummary {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub role: String,
    pub experiment_count: i64,
    pub recent_experiment_count: i64,
}

// ============================================================================
// Experiment-related types
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub hyperparams: Vec<serde_json::Value>,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub available_metrics: Vec<String>,
    pub workspace_id: Option<String>,
    pub url: String,
}

#[derive(Deserialize, Serialize, Debug)]
pub enum HyperparamValue {
    Float(f32),
    Integer(i64),
    String(String),
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Hyperparam {
    pub key: String,
    pub value: HyperparamValue,
}

// ============================================================================
// Metric-related types
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct Metric {
    pub id: i64,
    pub experiment_id: String,
    pub name: String,
    pub value: f64,
    pub step: Option<i64>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

// ============================================================================
// API Key-related types
// ============================================================================

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

// ============================================================================
// Invitation-related types
// ============================================================================

#[derive(Serialize, Deserialize, sqlx::FromRow)]
pub struct WorkspaceInvitation {
    pub id: String,
    #[serde(rename = "workspaceId")]
    pub workspace_id: String,
    #[serde(rename = "workspaceName")]
    pub workspace_name: String,
    pub email: String,
    pub role: String,
    pub from: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

// ============================================================================
// Dashboard-related types
// ============================================================================

#[derive(Serialize, Deserialize, Debug)]
pub struct DashboardOverview {
    pub workspaces: Vec<WorkspaceSummary>,
    pub recent_experiments: Vec<Experiment>,
}

#[derive(Serialize)]
pub struct SettingsData {
    pub user: UserInfo,
    pub workspaces: Vec<Workspace>,
    #[serde(rename = "apiKeys")]
    pub api_keys: Vec<ApiKey>,
    pub invitations: Vec<WorkspaceInvitation>,
}
