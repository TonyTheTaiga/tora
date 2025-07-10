use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes::Response;
use axum::{
    Extension, Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

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

#[derive(Deserialize)]
pub struct UpdateWorkspaceRequest {
    #[serde(rename = "workspace-name")]
    pub name: Option<String>,
    #[serde(rename = "workspace-description")]
    pub description: Option<String>,
}

#[derive(Serialize)]
pub struct WorkspaceRole {
    pub id: String,
    pub name: String,
}

#[derive(Deserialize)]
pub struct UpdateMemberRoleRequest {
    pub role_id: String,
}

pub async fn list_workspaces(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
) -> impl IntoResponse {
    let user_uuid = match Uuid::parse_str(&user.id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid user ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    let result = sqlx::query_as::<
        _,
        (
            String,
            String,
            Option<String>,
            chrono::DateTime<chrono::Utc>,
            String,
        ),
    >(
        r#"
        SELECT w.id::text, w.name, w.description, w.created_at, wr.name as role
        FROM workspace w
        JOIN user_workspaces uw ON w.id = uw.workspace_id
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE uw.user_id = $1
        ORDER BY w.created_at DESC
        "#,
    )
    .bind(user_uuid)
    .fetch_all(&pool)
    .await;

    match result {
        Ok(rows) => {
            let workspaces: Vec<Workspace> = rows
                .into_iter()
                .map(|(id, name, description, created_at, role)| Workspace {
                    id,
                    name,
                    description,
                    created_at,
                    role,
                })
                .collect();

            Json(Response {
                status: 200,
                data: Some(workspaces),
            })
            .into_response()
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to fetch workspaces".to_string()),
                }),
            )
                .into_response()
        }
    }
}

pub async fn create_workspace(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Json(request): Json<CreateWorkspaceRequest>,
) -> impl IntoResponse {
    let mut tx = match pool.begin().await {
        Ok(tx) => tx,
        Err(e) => {
            eprintln!("Failed to begin transaction: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to create workspace".to_string()),
                }),
            )
                .into_response();
        }
    };

    let workspace_result = sqlx::query_as::<_, (String, String, Option<String>, chrono::DateTime<chrono::Utc>)>(
        "INSERT INTO workspace (name, description) VALUES ($1, $2) RETURNING id::text, name, description, created_at",
    )
    .bind(&request.name)
    .bind(&request.description)
    .fetch_one(&mut *tx)
    .await;

    let (workspace_id, name, description, created_at) = match workspace_result {
        Ok(row) => row,
        Err(e) => {
            eprintln!("Failed to create workspace: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to create workspace".to_string()),
                }),
            )
                .into_response();
        }
    };

    let owner_role_result =
        sqlx::query_as::<_, (String,)>("SELECT id::text FROM workspace_role WHERE name = 'OWNER'")
            .fetch_one(&mut *tx)
            .await;

    let (owner_role_id,) = match owner_role_result {
        Ok(row) => row,
        Err(e) => {
            eprintln!("Failed to get OWNER role: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to create workspace".to_string()),
                }),
            )
                .into_response();
        }
    };

    let user_uuid = match Uuid::parse_str(&user.id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid user ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    let user_workspace_result = sqlx::query(
        "INSERT INTO user_workspaces (user_id, workspace_id, role_id) VALUES ($1, $2, $3)",
    )
    .bind(user_uuid)
    .bind(Uuid::parse_str(&workspace_id).unwrap())
    .bind(Uuid::parse_str(&owner_role_id).unwrap())
    .execute(&mut *tx)
    .await;

    if let Err(e) = user_workspace_result {
        eprintln!("Failed to add user to workspace: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(Response {
                status: 500,
                data: Some("Failed to create workspace".to_string()),
            }),
        )
            .into_response();
    }

    if let Err(e) = tx.commit().await {
        eprintln!("Failed to commit transaction: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(Response {
                status: 500,
                data: Some("Failed to create workspace".to_string()),
            }),
        )
            .into_response();
    }

    let workspace = Workspace {
        id: workspace_id,
        name,
        description,
        created_at,
        role: "OWNER".to_string(),
    };

    (
        StatusCode::CREATED,
        Json(Response {
            status: 201,
            data: Some(workspace),
        }),
    )
        .into_response()
}

pub async fn get_workspace(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Path(workspace_id): Path<String>,
) -> impl IntoResponse {
    let workspace_uuid = match Uuid::parse_str(&workspace_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid workspace ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    let result = sqlx::query_as::<
        _,
        (
            String,
            String,
            Option<String>,
            chrono::DateTime<chrono::Utc>,
            String,
        ),
    >(
        r#"
        SELECT w.id::text, w.name, w.description, w.created_at, wr.name as role
        FROM workspace w
        JOIN user_workspaces uw ON w.id = uw.workspace_id
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE w.id = $1 AND uw.user_id = $2
        "#,
    )
    .bind(workspace_uuid)
    .bind(Uuid::parse_str(&user.id).unwrap())
    .fetch_one(&pool)
    .await;

    match result {
        Ok((id, name, description, created_at, role)) => {
            let workspace = Workspace {
                id,
                name,
                description,
                created_at,
                role,
            };

            Json(Response {
                status: 200,
                data: Some(workspace),
            })
            .into_response()
        }
        Err(sqlx::Error::RowNotFound) => (
            StatusCode::NOT_FOUND,
            Json(Response {
                status: 404,
                data: Some("Workspace not found".to_string()),
            }),
        )
            .into_response(),
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to fetch workspace".to_string()),
                }),
            )
                .into_response()
        }
    }
}

pub async fn get_workspace_members(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Path(workspace_id): Path<String>,
) -> impl IntoResponse {
    let workspace_uuid = match Uuid::parse_str(&workspace_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid workspace ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    let access_check = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM user_workspaces WHERE workspace_id = $1 AND user_id = $2",
    )
    .bind(workspace_uuid)
    .bind(Uuid::parse_str(&user.id).unwrap())
    .fetch_one(&pool)
    .await;

    match access_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::FORBIDDEN,
                Json(Response {
                    status: 403,
                    data: Some("Access denied".to_string()),
                }),
            )
                .into_response();
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to check access".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    let result = sqlx::query_as::<_, (String, String, String, chrono::DateTime<chrono::Utc>)>(
        r#"
        SELECT u.id::text, u.email, wr.name as role, uw.created_at as joined_at
        FROM user_workspaces uw
        JOIN auth.users u ON uw.user_id = u.id
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE uw.workspace_id = $1
        ORDER BY uw.created_at
        "#,
    )
    .bind(workspace_uuid)
    .fetch_all(&pool)
    .await;

    match result {
        Ok(rows) => {
            let members: Vec<WorkspaceMember> = rows
                .into_iter()
                .map(|(id, email, role, joined_at)| WorkspaceMember {
                    id,
                    email,
                    role,
                    joined_at,
                })
                .collect();

            Json(Response {
                status: 200,
                data: Some(members),
            })
            .into_response()
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to fetch workspace members".to_string()),
                }),
            )
                .into_response()
        }
    }
}

pub async fn delete_workspace(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Path(workspace_id): Path<String>,
) -> impl IntoResponse {
    let workspace_uuid = match Uuid::parse_str(&workspace_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid workspace ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Check if user is the owner of this workspace
    let owner_check = sqlx::query_as::<_, (i64,)>(
        r#"
        SELECT COUNT(*) FROM user_workspaces uw
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE uw.workspace_id = $1 AND uw.user_id = $2 AND wr.name = 'OWNER'
        "#,
    )
    .bind(workspace_uuid)
    .bind(Uuid::parse_str(&user.id).unwrap())
    .fetch_one(&pool)
    .await;

    match owner_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::FORBIDDEN,
                Json(Response {
                    status: 403,
                    data: Some("Only workspace owners can delete workspaces".to_string()),
                }),
            )
                .into_response();
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to check ownership".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    // Delete the workspace (CASCADE will handle related records)
    let delete_result = sqlx::query("DELETE FROM workspace WHERE id = $1")
        .bind(workspace_uuid)
        .execute(&pool)
        .await;

    match delete_result {
        Ok(_) => Json(Response {
            status: 200,
            data: Some("Workspace deleted successfully"),
        })
        .into_response(),
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to delete workspace".to_string()),
                }),
            )
                .into_response()
        }
    }
}

pub async fn leave_workspace(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Path(workspace_id): Path<String>,
) -> impl IntoResponse {
    let workspace_uuid = match Uuid::parse_str(&workspace_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid workspace ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Check if user is the only owner - prevent leaving if so
    let owner_count = sqlx::query_as::<_, (i64,)>(
        r#"
        SELECT COUNT(*) FROM user_workspaces uw
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE uw.workspace_id = $1 AND wr.name = 'OWNER'
        "#,
    )
    .bind(workspace_uuid)
    .fetch_one(&pool)
    .await;

    let is_owner = sqlx::query_as::<_, (i64,)>(
        r#"
        SELECT COUNT(*) FROM user_workspaces uw
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE uw.workspace_id = $1 AND uw.user_id = $2 AND wr.name = 'OWNER'
        "#,
    )
    .bind(workspace_uuid)
    .bind(Uuid::parse_str(&user.id).unwrap())
    .fetch_one(&pool)
    .await;

    match (owner_count, is_owner) {
        (Ok((total_owners,)), Ok((user_is_owner,))) => {
            if user_is_owner > 0 && total_owners == 1 {
                return (StatusCode::BAD_REQUEST, Json(Response {
                    status: 400,
                    data: Some("Cannot leave workspace as the only owner. Transfer ownership or delete the workspace.".to_string()),
                })).into_response();
            }
        }
        (Err(e), _) | (_, Err(e)) => {
            eprintln!("Database error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to check ownership".to_string()),
                }),
            )
                .into_response();
        }
    }

    // Remove user from workspace
    let delete_result =
        sqlx::query("DELETE FROM user_workspaces WHERE workspace_id = $1 AND user_id = $2")
            .bind(workspace_uuid)
            .bind(Uuid::parse_str(&user.id).unwrap())
            .execute(&pool)
            .await;

    match delete_result {
        Ok(result) => {
            if result.rows_affected() == 0 {
                (
                    StatusCode::NOT_FOUND,
                    Json(Response {
                        status: 404,
                        data: Some("User not found in workspace".to_string()),
                    }),
                )
                    .into_response()
            } else {
                Json(Response {
                    status: 200,
                    data: Some("Left workspace successfully"),
                })
                .into_response()
            }
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to leave workspace".to_string()),
                }),
            )
                .into_response()
        }
    }
}

pub async fn update_workspace(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Path(workspace_id): Path<String>,
    Json(request): Json<UpdateWorkspaceRequest>,
) -> impl IntoResponse {
    let workspace_uuid = match Uuid::parse_str(&workspace_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid workspace ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Check if user has permission to update this workspace (OWNER or ADMIN)
    let permission_check = sqlx::query_as::<_, (i64,)>(
        r#"
        SELECT COUNT(*) FROM user_workspaces uw
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE uw.workspace_id = $1 AND uw.user_id = $2 AND wr.name IN ('OWNER', 'ADMIN')
        "#,
    )
    .bind(workspace_uuid)
    .bind(Uuid::parse_str(&user.id).unwrap())
    .fetch_one(&pool)
    .await;

    match permission_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::FORBIDDEN,
                Json(Response {
                    status: 403,
                    data: Some("Insufficient permissions to update workspace".to_string()),
                }),
            )
                .into_response();
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to check permissions".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    // Build dynamic update query
    let mut query_parts = Vec::new();
    let mut params: Vec<Box<dyn sqlx::Encode<'_, sqlx::Postgres> + Send + Sync>> = Vec::new();
    let mut param_count = 1;

    if let Some(name) = &request.name {
        query_parts.push(format!("name = ${}", param_count));
        params.push(Box::new(name.clone()));
        param_count += 1;
    }

    if let Some(description) = &request.description {
        query_parts.push(format!("description = ${}", param_count));
        params.push(Box::new(description.clone()));
        param_count += 1;
    }

    if query_parts.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(Response {
                status: 400,
                data: Some("No fields to update".to_string()),
            }),
        )
            .into_response();
    }

    // Add workspace_id as the last parameter
    params.push(Box::new(workspace_uuid));

    let query = format!(
        "UPDATE workspace SET {} WHERE id = ${} RETURNING id::text, name, description, created_at",
        query_parts.join(", "),
        param_count
    );

    let update_result = sqlx::query_as::<_, (String, String, Option<String>, chrono::DateTime<chrono::Utc>)>(&query)
        .bind_all(params)
        .fetch_one(&pool)
        .await;

    match update_result {
        Ok((id, name, description, created_at)) => {
            let workspace = Workspace {
                id,
                name,
                description,
                created_at,
                role: "".to_string(), // We don't need role for update response
            };

            Json(Response {
                status: 200,
                data: Some(workspace),
            })
            .into_response()
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to update workspace".to_string()),
                }),
            )
                .into_response()
        }
    }
}

pub async fn list_workspace_roles(
    Extension(_user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
) -> impl IntoResponse {
    let result = sqlx::query_as::<_, (String, String)>(
        "SELECT id::text, name FROM workspace_role ORDER BY name",
    )
    .fetch_all(&pool)
    .await;

    match result {
        Ok(rows) => {
            let roles: Vec<WorkspaceRole> = rows
                .into_iter()
                .map(|(id, name)| WorkspaceRole { id, name })
                .collect();

            Json(Response {
                status: 200,
                data: Some(roles),
            })
            .into_response()
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to fetch workspace roles".to_string()),
                }),
            )
                .into_response()
        }
    }
}

pub async fn update_member_role(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Path((workspace_id, member_id)): Path<(String, String)>,
    Json(request): Json<UpdateMemberRoleRequest>,
) -> impl IntoResponse {
    let workspace_uuid = match Uuid::parse_str(&workspace_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid workspace ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    let member_uuid = match Uuid::parse_str(&member_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid member ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    let role_uuid = match Uuid::parse_str(&request.role_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid role ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Check if user has permission to update member roles (OWNER or ADMIN)
    let permission_check = sqlx::query_as::<_, (i64,)>(
        r#"
        SELECT COUNT(*) FROM user_workspaces uw
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE uw.workspace_id = $1 AND uw.user_id = $2 AND wr.name IN ('OWNER', 'ADMIN')
        "#,
    )
    .bind(workspace_uuid)
    .bind(Uuid::parse_str(&user.id).unwrap())
    .fetch_one(&pool)
    .await;

    match permission_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::FORBIDDEN,
                Json(Response {
                    status: 403,
                    data: Some("Insufficient permissions to update member roles".to_string()),
                }),
            )
                .into_response();
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to check permissions".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    // Check if the member exists in the workspace
    let member_check = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM user_workspaces WHERE workspace_id = $1 AND user_id = $2",
    )
    .bind(workspace_uuid)
    .bind(member_uuid)
    .fetch_one(&pool)
    .await;

    match member_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::NOT_FOUND,
                Json(Response {
                    status: 404,
                    data: Some("Member not found in workspace".to_string()),
                }),
            )
                .into_response();
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to check member".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    // Check if the role exists
    let role_check = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM workspace_role WHERE id = $1",
    )
    .bind(role_uuid)
    .fetch_one(&pool)
    .await;

    match role_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid role ID".to_string()),
                }),
            )
                .into_response();
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to check role".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    // Update the member's role
    let update_result = sqlx::query(
        "UPDATE user_workspaces SET role_id = $1 WHERE workspace_id = $2 AND user_id = $3",
    )
    .bind(role_uuid)
    .bind(workspace_uuid)
    .bind(member_uuid)
    .execute(&pool)
    .await;

    match update_result {
        Ok(_) => Json(Response {
            status: 200,
            data: Some("Member role updated successfully"),
        })
        .into_response(),
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to update member role".to_string()),
                }),
            )
                .into_response()
        }
    }
}
