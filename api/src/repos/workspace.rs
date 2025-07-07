use axum::{
    Extension, Json,
    extract::Path,
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes::Response;

#[derive(Serialize, Deserialize, Debug)]
pub struct Workspace {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub role: String,
}

#[derive(Deserialize)]
pub struct CreateWorkspaceRequest {
    #[serde(rename = "workspace-name")]
    pub name: String,
    #[serde(rename = "workspace-description")]
    pub description: Option<String>,
}

#[derive(Serialize)]
pub struct WorkspaceMember {
    pub id: String,
    pub email: String,
    pub role: String,
    pub joined_at: chrono::DateTime<chrono::Utc>,
}

pub async fn list_workspaces(Extension(_user): Extension<AuthenticatedUser>) -> impl IntoResponse {
    // Mock data - will be replaced with database queries
    let workspaces = vec![
        Workspace {
            id: "ws_1".to_string(),
            name: "ML Research".to_string(),
            description: Some("Machine learning experiments and research".to_string()),
            created_at: chrono::Utc::now(),
            role: "OWNER".to_string(),
        },
        Workspace {
            id: "ws_2".to_string(),
            name: "NLP Project".to_string(),
            description: Some("Natural language processing experiments".to_string()),
            created_at: chrono::Utc::now(),
            role: "ADMIN".to_string(),
        },
    ];

    Json(Response {
        status: 200,
        data: Some(workspaces),
    })
}

pub async fn create_workspace(
    Extension(_user): Extension<AuthenticatedUser>,
    Json(request): Json<CreateWorkspaceRequest>,
) -> impl IntoResponse {
    let workspace = Workspace {
        id: format!("ws_{}", uuid::Uuid::new_v4()),
        name: request.name,
        description: request.description,
        created_at: chrono::Utc::now(),
        role: "OWNER".to_string(),
    };

    (
        StatusCode::CREATED,
        Json(Response {
            status: 201,
            data: Some(workspace),
        }),
    )
}

pub async fn get_workspace(
    Extension(_user): Extension<AuthenticatedUser>,
    Path(workspace_id): Path<String>,
) -> impl IntoResponse {
    // Mock data - will be replaced with database query
    let workspace = Workspace {
        id: workspace_id,
        name: "Test Workspace".to_string(),
        description: Some("A test workspace".to_string()),
        created_at: chrono::Utc::now(),
        role: "OWNER".to_string(),
    };

    Json(Response {
        status: 200,
        data: Some(workspace),
    })
}

pub async fn get_workspace_members(
    Extension(_user): Extension<AuthenticatedUser>,
    Path(_workspace_id): Path<String>,
) -> impl IntoResponse {
    // Mock data - will be replaced with database query
    let members = vec![
        WorkspaceMember {
            id: "user_1".to_string(),
            email: "user@example.com".to_string(),
            role: "OWNER".to_string(),
            joined_at: chrono::Utc::now(),
        }
    ];

    Json(Response {
        status: 200,
        data: Some(members),
    })
}

pub async fn delete_workspace(
    Extension(_user): Extension<AuthenticatedUser>,
    Path(workspace_id): Path<String>,
) -> impl IntoResponse {
    // Mock - will be replaced with database deletion
    println!("Deleting workspace: {}", workspace_id);

    Json(Response {
        status: 200,
        data: Some("Workspace deleted successfully"),
    })
}

pub async fn leave_workspace(
    Extension(_user): Extension<AuthenticatedUser>,
    Path(workspace_id): Path<String>,
) -> impl IntoResponse {
    // Mock - will be replaced with database update to remove user from workspace
    println!("User leaving workspace: {}", workspace_id);

    Json(Response {
        status: 200,
        data: Some("Left workspace successfully"),
    })
}

