use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use crate::types::{
    AppError, AppResult, CreateWorkspaceRequest, Response, Workspace, WorkspaceMember,
};
use axum::{
    Extension, Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use tracing::{debug, info};
use uuid::Uuid;

pub async fn list_workspaces(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
) -> AppResult<impl IntoResponse> {
    info!("Listing workspaces for user: {}", user.email);

    let user_uuid = crate::types::error::parse_uuid(&user.id, "user_id")?;

    debug!(
        "Executing query to fetch workspaces for user: {}",
        user_uuid
    );
    let rows = sqlx::query_as::<
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
    .fetch_all(&app_state.db_pool)
    .await?;

    {
        info!(
            "Successfully fetched {} workspaces for user: {}",
            rows.len(),
            user.email
        );
        let workspaces: Vec<Workspace> = rows
            .into_iter()
            .map(|(id, name, description, created_at, role)| {
                debug!("Workspace: {} ({}), role: {}", name, id, role);
                Workspace {
                    id,
                    name,
                    description,
                    created_at,
                    role,
                }
            })
            .collect();

        Ok(Json(Response {
            status: 200,
            data: Some(workspaces),
        })
        .into_response())
    }
}

pub async fn create_workspace(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Json(request): Json<CreateWorkspaceRequest>,
) -> AppResult<impl IntoResponse> {
    let mut tx = app_state.db_pool.begin().await?;

    let workspace_result = sqlx::query_as::<_, (String, String, Option<String>, chrono::DateTime<chrono::Utc>)>(
        "INSERT INTO workspace (name, description) VALUES ($1, $2) RETURNING id::text, name, description, created_at",
    )
    .bind(&request.name)
    .bind(&request.description)
    .fetch_one(&mut *tx)
    .await?;

    let (workspace_id, name, description, created_at) = workspace_result;

    let owner_role_result =
        sqlx::query_as::<_, (String,)>("SELECT id::text FROM workspace_role WHERE name = 'OWNER'")
            .fetch_one(&mut *tx)
            .await?;

    let (owner_role_id,) = owner_role_result;

    let user_uuid = crate::types::error::parse_uuid(&user.id, "user_id")?;

    sqlx::query("INSERT INTO user_workspaces (user_id, workspace_id, role_id) VALUES ($1, $2, $3)")
        .bind(user_uuid)
        .bind(Uuid::parse_str(&workspace_id).unwrap())
        .bind(Uuid::parse_str(&owner_role_id).unwrap())
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;

    let workspace = Workspace {
        id: workspace_id,
        name,
        description,
        created_at,
        role: "OWNER".to_string(),
    };

    Ok((
        StatusCode::CREATED,
        Json(Response {
            status: 201,
            data: Some(workspace),
        }),
    )
        .into_response())
}

pub async fn get_workspace(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(workspace_id): Path<String>,
) -> AppResult<impl IntoResponse> {
    let workspace_uuid = crate::types::error::parse_uuid(&workspace_id, "workspace_id")?;

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
    .fetch_one(&app_state.db_pool)
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

            Ok(Json(Response {
                status: 200,
                data: Some(workspace),
            })
            .into_response())
        }
        Err(sqlx::Error::RowNotFound) => Err(AppError::NotFound("Workspace not found".to_string())),
        Err(e) => Err(e.into()),
    }
}

pub async fn get_workspace_members(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(workspace_id): Path<String>,
) -> AppResult<impl IntoResponse> {
    let workspace_uuid = crate::types::error::parse_uuid(&workspace_id, "workspace_id")?;

    let access_check = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM user_workspaces WHERE workspace_id = $1 AND user_id = $2",
    )
    .bind(workspace_uuid)
    .bind(Uuid::parse_str(&user.id).unwrap())
    .fetch_one(&app_state.db_pool)
    .await?;

    if access_check.0 == 0 {
        return Err(AppError::Forbidden("Access denied".to_string()));
    }

    let rows = sqlx::query_as::<_, (String, String, String, chrono::DateTime<chrono::Utc>)>(
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
    .fetch_all(&app_state.db_pool)
    .await?;

    {
        let members: Vec<WorkspaceMember> = rows
            .into_iter()
            .map(|(id, email, role, joined_at)| WorkspaceMember {
                id,
                email,
                role,
                joined_at,
            })
            .collect();

        Ok(Json(Response {
            status: 200,
            data: Some(members),
        })
        .into_response())
    }
}

pub async fn delete_workspace(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(workspace_id): Path<String>,
) -> AppResult<impl IntoResponse> {
    let workspace_uuid = crate::types::error::parse_uuid(&workspace_id, "workspace_id")?;

    let owner_check = sqlx::query_as::<_, (i64,)>(
        r#"
        SELECT COUNT(*) FROM user_workspaces uw
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE uw.workspace_id = $1 AND uw.user_id = $2 AND wr.name = 'OWNER'
        "#,
    )
    .bind(workspace_uuid)
    .bind(Uuid::parse_str(&user.id).unwrap())
    .fetch_one(&app_state.db_pool)
    .await?;

    if owner_check.0 == 0 {
        return Err(AppError::Forbidden(
            "Only workspace owners can delete workspaces".to_string(),
        ));
    }

    // Delete the workspace (CASCADE will handle related records)
    let delete_result = sqlx::query("DELETE FROM workspace WHERE id = $1")
        .bind(workspace_uuid)
        .execute(&app_state.db_pool)
        .await;

    match delete_result {
        Ok(_) => Ok(Json(Response {
            status: 200,
            data: Some("Workspace deleted successfully"),
        })
        .into_response()),
        Err(e) => Err(e.into()),
    }
}

pub async fn leave_workspace(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(workspace_id): Path<String>,
) -> AppResult<impl IntoResponse> {
    let workspace_uuid = crate::types::error::parse_uuid(&workspace_id, "workspace_id")?;

    // Check if user is the only owner - prevent leaving if so
    let owner_count = sqlx::query_as::<_, (i64,)>(
        r#"
        SELECT COUNT(*) FROM user_workspaces uw
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE uw.workspace_id = $1 AND wr.name = 'OWNER'
        "#,
    )
    .bind(workspace_uuid)
    .fetch_one(&app_state.db_pool)
    .await?;

    let is_owner = sqlx::query_as::<_, (i64,)>(
        r#"
        SELECT COUNT(*) FROM user_workspaces uw
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE uw.workspace_id = $1 AND uw.user_id = $2 AND wr.name = 'OWNER'
        "#,
    )
    .bind(workspace_uuid)
    .bind(Uuid::parse_str(&user.id).unwrap())
    .fetch_one(&app_state.db_pool)
    .await?;

    let ((total_owners,), (user_is_owner,)) = (owner_count, is_owner);
    if user_is_owner > 0 && total_owners == 1 {
        return Err(AppError::BadRequest(
            "Cannot leave workspace as the only owner. Transfer ownership or delete the workspace."
                .to_string(),
        ));
    }

    let delete_result =
        sqlx::query("DELETE FROM user_workspaces WHERE workspace_id = $1 AND user_id = $2")
            .bind(workspace_uuid)
            .bind(Uuid::parse_str(&user.id).unwrap())
            .execute(&app_state.db_pool)
            .await?;

    if delete_result.rows_affected() == 0 {
        return Err(AppError::NotFound(
            "User not found in workspace".to_string(),
        ));
    }

    Ok(Json(Response {
        status: 200,
        data: Some("Left workspace successfully"),
    })
    .into_response())
}
