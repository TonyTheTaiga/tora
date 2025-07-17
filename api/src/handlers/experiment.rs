use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use crate::types::{
    CreateExperimentRequest, Experiment, ListExperimentsQuery, Response, UpdateExperimentRequest,
};
use axum::{
    Extension, Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use tracing::{debug, error, info};
use uuid::Uuid;

pub async fn create_experiment(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Json(request): Json<CreateExperimentRequest>,
) -> impl IntoResponse {
    info!(
        "Creating experiment '{}' for user: {} in workspace: {}",
        request.name, user.email, request.workspace_id
    );
    debug!("Experiment request: {:?}", request);

    let user_uuid = match Uuid::parse_str(&user.id) {
        Ok(uuid) => {
            debug!("Parsed user UUID: {}", uuid);
            uuid
        }
        Err(e) => {
            error!("Failed to parse user ID '{}': {}", user.id, e);
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

    // TODO: Handler anonymous experiments.
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

    let mut tx = match app_state.db_pool.begin().await {
        Ok(tx) => tx,
        Err(e) => {
            eprintln!("Failed to begin transaction: {e}");
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

    let access_check = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM user_workspaces WHERE workspace_id = $1 AND user_id = $2",
    )
    .bind(workspace_uuid)
    .bind(user_uuid)
    .fetch_one(&mut *tx)
    .await;

    match access_check {
        Ok((0,)) => {
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
            eprintln!("Database error: {e}");
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

    let experiment_result = sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>(
        "INSERT INTO experiment (name, description, tags, hyperparams) VALUES ($1, $2, $3, $4) RETURNING id::text, name, description, hyperparams, tags, created_at, updated_at",
    )
    .bind(&request.name)
    .bind(if request.description.is_empty() { None } else { Some(&request.description) })
    .bind(request.tags.unwrap_or_default())
    .bind(request.hyperparams)
    .fetch_one(&mut *tx)
    .await;

    let (experiment_id, name, description, hyperparams, tags, created_at, updated_at) =
        match experiment_result {
            Ok(row) => row,
            Err(e) => {
                eprintln!("Failed to create experiment: {e}");
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

    let workspace_experiment_result = sqlx::query(
        "INSERT INTO workspace_experiments (workspace_id, experiment_id) VALUES ($1, $2)",
    )
    .bind(workspace_uuid)
    .bind(Uuid::parse_str(&experiment_id).unwrap())
    .execute(&mut *tx)
    .await;

    if let Err(e) = workspace_experiment_result {
        eprintln!("Failed to add experiment to workspace: {e}");
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
        eprintln!("Failed to commit transaction: {e}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(Response {
                status: 500,
                data: Some("Failed to create experiment".to_string()),
            }),
        )
            .into_response();
    }

    let frontend_url = app_state.settings.frontend_url;
    let experiment = Experiment {
        id: experiment_id.to_string(),
        name,
        description,
        hyperparams: hyperparams.unwrap_or_default(),
        tags: tags.unwrap_or_default(),
        created_at,
        updated_at,
        available_metrics: vec![],
        workspace_id: Some(request.workspace_id),
        url: format!("{frontend_url}/experiments/{experiment_id}"),
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

pub async fn list_experiments(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
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

        // List experiments for a specific workspace with metrics summary
        sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>, String, Option<Vec<String>>)>(
            r#"
            SELECT e.id::text, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at, we.workspace_id::text,
                   ARRAY_AGG(DISTINCT m.name) FILTER (WHERE m.name IS NOT NULL) as available_metrics
            FROM experiment e
            JOIN workspace_experiments we ON e.id = we.experiment_id
            JOIN user_workspaces uw ON we.workspace_id = uw.workspace_id
            LEFT JOIN metric m ON e.id = m.experiment_id
            WHERE we.workspace_id = $1 AND uw.user_id = $2
            GROUP BY e.id, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at, we.workspace_id
            ORDER BY e.created_at DESC
            "#,
        )
        .bind(workspace_uuid)
        .bind(user_uuid)
        .fetch_all(&app_state.db_pool)
        .await
    } else {
        sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>, String, Option<Vec<String>>)>(
            r#"
            SELECT DISTINCT e.id::text, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at, we.workspace_id::text,
                   ARRAY_AGG(DISTINCT m.name) FILTER (WHERE m.name IS NOT NULL) as available_metrics
            FROM experiment e
            JOIN workspace_experiments we ON e.id = we.experiment_id
            JOIN user_workspaces uw ON we.workspace_id = uw.workspace_id
            LEFT JOIN metric m ON e.id = m.experiment_id
            WHERE uw.user_id = $1
            GROUP BY e.id, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at, we.workspace_id
            ORDER BY e.created_at DESC
            "#,
        )
        .bind(user_uuid)
        .fetch_all(&app_state.db_pool)
        .await
    };

    let frontend_url = app_state.settings.frontend_url;
    match result {
        Ok(rows) => {
            let experiments: Vec<Experiment> = rows
                .into_iter()
                .map(
                    |(
                        id,
                        name,
                        description,
                        hyperparams,
                        tags,
                        created_at,
                        updated_at,
                        workspace_id,
                        available_metrics,
                    )| {
                        Experiment {
                            id: id.to_string(),
                            name,
                            description,
                            hyperparams: hyperparams.unwrap_or_default(),
                            tags: tags.unwrap_or_default(),
                            created_at,
                            updated_at,
                            available_metrics: available_metrics.unwrap_or_default(),
                            workspace_id: Some(workspace_id),
                            url: format!("{frontend_url}/experiments/{id}"),
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
            eprintln!("Database error: {e}");
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

// Get single experiment
pub async fn get_experiment(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
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

    let result = sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>, String, Option<Vec<String>>)>(
        r#"
        SELECT e.id::text, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at, we.workspace_id::text,
               ARRAY_AGG(DISTINCT m.name) FILTER (WHERE m.name IS NOT NULL) as available_metrics
        FROM experiment e
        JOIN workspace_experiments we ON e.id = we.experiment_id
        JOIN user_workspaces uw ON we.workspace_id = uw.workspace_id
        LEFT JOIN metric m ON e.id = m.experiment_id
        WHERE e.id = $1 AND uw.user_id = $2
        GROUP BY e.id, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at, we.workspace_id
        "#,
    )
    .bind(experiment_uuid)
    .bind(user_uuid)
    .fetch_one(&app_state.db_pool)
    .await;

    let frontend_url = app_state.settings.frontend_url;
    match result {
        Ok((
            id,
            name,
            description,
            hyperparams,
            tags,
            created_at,
            updated_at,
            workspace_id,
            available_metrics,
        )) => {
            let experiment = Experiment {
                id: id.to_string(),
                name,
                description,
                hyperparams: hyperparams.unwrap_or_default(),
                tags: tags.unwrap_or_default(),
                created_at,
                updated_at,
                available_metrics: available_metrics.unwrap_or_default(),
                workspace_id: Some(workspace_id),
                url: format!("{frontend_url}/experiments/{id}"),
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
            eprintln!("Database error: {e}");
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

pub async fn update_experiment(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(id): Path<String>,
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

    let experiment_uuid = match Uuid::parse_str(&id) {
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
    .fetch_one(&app_state.db_pool)
    .await;

    match access_check {
        Ok((0,)) => {
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
            eprintln!("Database error: {e}");
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

    let result = sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>(
        "UPDATE experiment SET name = $1, description = $2, tags = $3, updated_at = NOW() WHERE id = $4 RETURNING id::text, name, description, hyperparams, tags, created_at, updated_at",
    )
    .bind(&request.name)
    .bind(if request.description.is_empty() { None } else { Some(&request.description) })
    .bind(request.tags.unwrap_or_default())
    .bind(experiment_uuid)
    .fetch_one(&app_state.db_pool)
    .await;

    let frontend_url = app_state.settings.frontend_url;
    match result {
        Ok((id, name, description, hyperparams, tags, created_at, updated_at)) => {
            let experiment = Experiment {
                id: id.to_string(),
                name,
                description,
                hyperparams: hyperparams.unwrap_or_default(),
                tags: tags.unwrap_or_default(),
                created_at,
                updated_at,
                available_metrics: vec![], // TODO: Fetch from metrics table
                workspace_id: None,        // We don't have workspace_id in the update
                url: format!("{frontend_url}/experiments/{id}"),
            };

            Json(Response {
                status: 200,
                data: Some(experiment),
            })
            .into_response()
        }
        Err(e) => {
            eprintln!("Database error: {e}");
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
    State(app_state): State<AppState>,
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
    .fetch_one(&app_state.db_pool)
    .await;

    match access_check {
        Ok((0,)) => {
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
            eprintln!("Database error: {e}");
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

    let delete_result = sqlx::query("DELETE FROM experiment WHERE id = $1")
        .bind(experiment_uuid)
        .execute(&app_state.db_pool)
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
            eprintln!("Database error: {e}");
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
    State(app_state): State<AppState>,
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
    .fetch_one(&app_state.db_pool)
    .await;

    match access_check {
        Ok((0,)) => {
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
            eprintln!("Database error: {e}");
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

    // List experiments for the workspace with metrics summary
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
            Option<Vec<String>>,
        ),
    >(
        r#"
        SELECT e.id::text, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at,
               ARRAY_AGG(DISTINCT m.name) FILTER (WHERE m.name IS NOT NULL) as available_metrics
        FROM experiment e
        JOIN workspace_experiments we ON e.id = we.experiment_id
        LEFT JOIN metric m ON e.id = m.experiment_id
        WHERE we.workspace_id = $1
        GROUP BY e.id, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at
        ORDER BY e.created_at DESC
        "#,
    )
    .bind(workspace_uuid)
    .fetch_all(&app_state.db_pool)
    .await;

    let frontend_url = app_state.settings.frontend_url;
    match result {
        Ok(rows) => {
            let experiments: Vec<Experiment> = rows
                .into_iter()
                .map(
                    |(
                        id,
                        name,
                        description,
                        hyperparams,
                        tags,
                        created_at,
                        updated_at,
                        available_metrics,
                    )| {
                        Experiment {
                            id: id.to_string(),
                            name,
                            description,
                            hyperparams: hyperparams.unwrap_or_default(),
                            tags: tags.unwrap_or_default(),
                            created_at,
                            updated_at,
                            available_metrics: available_metrics.unwrap_or_default(),
                            workspace_id: Some(workspace_id.clone()),
                            url: format!("{frontend_url}/experiments/{id}"),
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
            eprintln!("Database error: {e}");
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
