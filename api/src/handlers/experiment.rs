use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes::Response;
use axum::{
    Extension, Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

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
}

#[derive(Deserialize)]
pub struct CreateExperimentRequest {
    #[serde(rename = "experiment-name")]
    pub name: String,
    #[serde(rename = "experiment-description")]
    pub description: String,
    #[serde(rename = "workspace-id")]
    pub workspace_id: String,
    pub tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct UpdateExperimentRequest {
    #[serde(rename = "experiment-id")]
    pub id: String,
    #[serde(rename = "experiment-name")]
    pub name: String,
    #[serde(rename = "experiment-description")]
    pub description: String,
    pub tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct ListExperimentsQuery {
    pub workspace: Option<String>,
}

pub async fn list_experiments(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Query(query): Query<ListExperimentsQuery>,
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

    let result = if let Some(workspace_id) = &query.workspace {
        let workspace_uuid = match Uuid::parse_str(workspace_id) {
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

        // List experiments for a specific workspace
        sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>, String)>(
            r#"
            SELECT e.id::text, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at, we.workspace_id::text
            FROM experiment e
            JOIN workspace_experiments we ON e.id = we.experiment_id
            JOIN user_workspaces uw ON we.workspace_id = uw.workspace_id
            WHERE we.workspace_id = $1 AND uw.user_id = $2
            ORDER BY e.created_at DESC
            "#,
        )
        .bind(workspace_uuid)
        .bind(user_uuid)
        .fetch_all(&pool)
        .await
    } else {
        // List all experiments the user has access to
        sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>, String)>(
            r#"
            SELECT DISTINCT e.id::text, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at, we.workspace_id::text
            FROM experiment e
            JOIN workspace_experiments we ON e.id = we.experiment_id
            JOIN user_workspaces uw ON we.workspace_id = uw.workspace_id
            WHERE uw.user_id = $1
            ORDER BY e.created_at DESC
            "#,
        )
        .bind(user_uuid)
        .fetch_all(&pool)
        .await
    };

    match result {
        Ok(rows) => {
            let experiments: Vec<Experiment> = rows
                .into_iter()
                .map(
                    |(id, name, description, hyperparams, tags, created_at, updated_at, workspace_id)| {
                        Experiment {
                            id,
                            name,
                            description,
                            hyperparams: hyperparams.unwrap_or_default(),
                            tags: tags.unwrap_or_default(),
                            created_at,
                            updated_at,
                            available_metrics: vec![], // TODO: Fetch from metrics table
                            workspace_id: Some(workspace_id),
                        }
                    },
                )
                .collect();

            Json(Response {
                status: 200,
                data: Some(experiments),
            })
            .into_response()
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to fetch experiments".to_string()),
                }),
            )
                .into_response()
        }
    }
}

// Create experiment
pub async fn create_experiment(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Json(request): Json<CreateExperimentRequest>,
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

    let workspace_uuid = match Uuid::parse_str(&request.workspace_id) {
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

    let mut tx = match pool.begin().await {
        Ok(tx) => tx,
        Err(e) => {
            eprintln!("Failed to begin transaction: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to create experiment".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Check if user has access to the workspace
    let access_check = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM user_workspaces WHERE workspace_id = $1 AND user_id = $2",
    )
    .bind(workspace_uuid)
    .bind(user_uuid)
    .fetch_one(&mut *tx)
    .await;

    match access_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::FORBIDDEN,
                Json(Response {
                    status: 403,
                    data: Some("Access denied to workspace".to_string()),
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
                    data: Some("Failed to check workspace access".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    // Create the experiment
    let experiment_result = sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>(
        "INSERT INTO experiment (name, description, tags) VALUES ($1, $2, $3) RETURNING id::text, name, description, hyperparams, tags, created_at, updated_at",
    )
    .bind(&request.name)
    .bind(if request.description.is_empty() { None } else { Some(&request.description) })
    .bind(&request.tags.unwrap_or_default())
    .fetch_one(&mut *tx)
    .await;

    let (experiment_id, name, description, hyperparams, tags, created_at, updated_at) =
        match experiment_result {
            Ok(row) => row,
            Err(e) => {
                eprintln!("Failed to create experiment: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(Response {
                        status: 500,
                        data: Some("Failed to create experiment".to_string()),
                    }),
                )
                    .into_response();
            }
        };

    // Add experiment to workspace
    let workspace_experiment_result = sqlx::query(
        "INSERT INTO workspace_experiments (workspace_id, experiment_id) VALUES ($1, $2)",
    )
    .bind(workspace_uuid)
    .bind(Uuid::parse_str(&experiment_id).unwrap())
    .execute(&mut *tx)
    .await;

    if let Err(e) = workspace_experiment_result {
        eprintln!("Failed to add experiment to workspace: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(Response {
                status: 500,
                data: Some("Failed to create experiment".to_string()),
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
                data: Some("Failed to create experiment".to_string()),
            }),
        )
            .into_response();
    }

    let experiment = Experiment {
        id: experiment_id,
        name,
        description,
        hyperparams: hyperparams.unwrap_or_default(),
        tags: tags.unwrap_or_default(),
        created_at,
        updated_at,
        available_metrics: vec![],
        workspace_id: Some(request.workspace_id),
    };

    (
        StatusCode::CREATED,
        Json(Response {
            status: 201,
            data: Some(experiment),
        }),
    )
        .into_response()
}

// Get single experiment
pub async fn get_experiment(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Path(experiment_id): Path<String>,
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

    let experiment_uuid = match Uuid::parse_str(&experiment_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid experiment ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    let result = sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>, String)>(
        r#"
        SELECT e.id::text, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at, we.workspace_id::text
        FROM experiment e
        JOIN workspace_experiments we ON e.id = we.experiment_id
        JOIN user_workspaces uw ON we.workspace_id = uw.workspace_id
        WHERE e.id = $1 AND uw.user_id = $2
        "#,
    )
    .bind(experiment_uuid)
    .bind(user_uuid)
    .fetch_one(&pool)
    .await;

    match result {
        Ok((id, name, description, hyperparams, tags, created_at, updated_at, workspace_id)) => {
            let experiment = Experiment {
                id,
                name,
                description,
                hyperparams: hyperparams.unwrap_or_default(),
                tags: tags.unwrap_or_default(),
                created_at,
                updated_at,
                available_metrics: vec![], // TODO: Fetch from metrics table
                workspace_id: Some(workspace_id),
            };

            Json(Response {
                status: 200,
                data: Some(experiment),
            })
            .into_response()
        }
        Err(sqlx::Error::RowNotFound) => (
            StatusCode::NOT_FOUND,
            Json(Response {
                status: 404,
                data: Some("Experiment not found".to_string()),
            }),
        )
            .into_response(),
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to fetch experiment".to_string()),
                }),
            )
                .into_response()
        }
    }
}

// Update experiment
pub async fn update_experiment(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Json(request): Json<UpdateExperimentRequest>,
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

    let experiment_uuid = match Uuid::parse_str(&request.id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid experiment ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Check if user has access to this experiment
    let access_check = sqlx::query_as::<_, (i64,)>(
        r#"
        SELECT COUNT(*) FROM experiment e
        JOIN workspace_experiments we ON e.id = we.experiment_id
        JOIN user_workspaces uw ON we.workspace_id = uw.workspace_id
        WHERE e.id = $1 AND uw.user_id = $2
        "#,
    )
    .bind(experiment_uuid)
    .bind(user_uuid)
    .fetch_one(&pool)
    .await;

    match access_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::FORBIDDEN,
                Json(Response {
                    status: 403,
                    data: Some("Access denied to experiment".to_string()),
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
                    data: Some("Failed to check experiment access".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    // Update the experiment
    let result = sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>(
        "UPDATE experiment SET name = $1, description = $2, tags = $3, updated_at = NOW() WHERE id = $4 RETURNING id::text, name, description, hyperparams, tags, created_at, updated_at",
    )
    .bind(&request.name)
    .bind(if request.description.is_empty() { None } else { Some(&request.description) })
    .bind(&request.tags.unwrap_or_default())
    .bind(experiment_uuid)
    .fetch_one(&pool)
    .await;

    match result {
        Ok((id, name, description, hyperparams, tags, created_at, updated_at)) => {
            let experiment = Experiment {
                id,
                name,
                description,
                hyperparams: hyperparams.unwrap_or_default(),
                tags: tags.unwrap_or_default(),
                created_at,
                updated_at,
                available_metrics: vec![], // TODO: Fetch from metrics table
                workspace_id: None,        // We don't have workspace_id in the update
            };

            Json(Response {
                status: 200,
                data: Some(experiment),
            })
            .into_response()
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to update experiment".to_string()),
                }),
            )
                .into_response()
        }
    }
}

// Delete experiment
pub async fn delete_experiment(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Path(experiment_id): Path<String>,
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

    let experiment_uuid = match Uuid::parse_str(&experiment_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid experiment ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Check if user has owner/admin access to this experiment
    let access_check = sqlx::query_as::<_, (i64,)>(
        r#"
        SELECT COUNT(*) FROM experiment e
        JOIN workspace_experiments we ON e.id = we.experiment_id
        JOIN user_workspaces uw ON we.workspace_id = uw.workspace_id
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE e.id = $1 AND uw.user_id = $2 AND wr.name IN ('OWNER', 'ADMIN')
        "#,
    )
    .bind(experiment_uuid)
    .bind(user_uuid)
    .fetch_one(&pool)
    .await;

    match access_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::FORBIDDEN,
                Json(Response {
                    status: 403,
                    data: Some(
                        "Access denied - only workspace owners and admins can delete experiments"
                            .to_string(),
                    ),
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
                    data: Some("Failed to check experiment access".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    // Delete the experiment (CASCADE will handle related records)
    let delete_result = sqlx::query("DELETE FROM experiment WHERE id = $1")
        .bind(experiment_uuid)
        .execute(&pool)
        .await;

    match delete_result {
        Ok(result) => {
            if result.rows_affected() == 0 {
                (
                    StatusCode::NOT_FOUND,
                    Json(Response {
                        status: 404,
                        data: Some("Experiment not found".to_string()),
                    }),
                )
                    .into_response()
            } else {
                (
                    StatusCode::NO_CONTENT,
                    Json(Response::<()> {
                        status: 204,
                        data: None,
                    }),
                )
                    .into_response()
            }
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to delete experiment".to_string()),
                }),
            )
                .into_response()
        }
    }
}

// List experiments for a specific workspace
pub async fn list_workspace_experiments(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Path(workspace_id): Path<String>,
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

    // Check if user has access to the workspace
    let access_check = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM user_workspaces WHERE workspace_id = $1 AND user_id = $2",
    )
    .bind(workspace_uuid)
    .bind(user_uuid)
    .fetch_one(&pool)
    .await;

    match access_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::FORBIDDEN,
                Json(Response {
                    status: 403,
                    data: Some("Access denied to workspace".to_string()),
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
                    data: Some("Failed to check workspace access".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    // List experiments for the workspace
    let result = sqlx::query_as::<
        _,
        (
            String,
            String,
            Option<String>,
            Option<Vec<serde_json::Value>>,
            Option<Vec<String>>,
            chrono::DateTime<chrono::Utc>,
            chrono::DateTime<chrono::Utc>,
        ),
    >(
        r#"
        SELECT e.id::text, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at
        FROM experiment e
        JOIN workspace_experiments we ON e.id = we.experiment_id
        WHERE we.workspace_id = $1
        ORDER BY e.created_at DESC
        "#,
    )
    .bind(workspace_uuid)
    .fetch_all(&pool)
    .await;

    match result {
        Ok(rows) => {
            let experiments: Vec<Experiment> = rows
                .into_iter()
                .map(
                    |(id, name, description, hyperparams, tags, created_at, updated_at)| {
                        Experiment {
                            id,
                            name,
                            description,
                            hyperparams: hyperparams.unwrap_or_default(),
                            tags: tags.unwrap_or_default(),
                            created_at,
                            updated_at,
                            available_metrics: vec![], // TODO: Fetch from metrics table
                            workspace_id: Some(workspace_id.clone()),
                        }
                    },
                )
                .collect();

            Json(Response {
                status: 200,
                data: Some(experiments),
            })
            .into_response()
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to fetch experiments".to_string()),
                }),
            )
                .into_response()
        }
    }
}
