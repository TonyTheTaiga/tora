use crate::middleware::auth::AuthenticatedUser;
use crate::types::{CreateWorkspaceRequest, Response, Workspace, WorkspaceMember};
use axum::{
    Extension, Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};

use sqlx::PgPool;
use uuid::Uuid;

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
            eprintln!("Database error: {e}");
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
    println!("starting create workspace...");

    let mut tx = match pool.begin().await {
        Ok(tx) => tx,
        Err(e) => {
            eprintln!("Failed to begin transaction: {e}");
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
            eprintln!("Failed to create workspace: {e}");
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
            eprintln!("Failed to get OWNER role: {e}");
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
        eprintln!("Failed to add user to workspace: {e}");
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
        eprintln!("Failed to commit transaction: {e}");
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
            eprintln!("Database error: {e}");
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
        Ok((0,)) => {
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
            eprintln!("Database error: {e}");
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
            eprintln!("Database error: {e}");
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
        Ok((0,)) => {
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
            eprintln!("Database error: {e}");
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
            eprintln!("Database error: {e}");
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
            eprintln!("Database error: {e}");
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
            eprintln!("Database error: {e}");
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
