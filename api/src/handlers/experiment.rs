use crate::handlers::{AppError, AppResult, parse_uuid};
use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use crate::types::{
    BatchGetExperimentsRequest, CreateExperimentRequest, Experiment, ListExperimentsQuery,
    Response, UpdateExperimentRequest,
};
use axum::{
    Extension, Json,
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use tracing::{debug, info};
use uuid::Uuid;

pub async fn create_experiment(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Json(request): Json<CreateExperimentRequest>,
) -> AppResult<impl IntoResponse> {
    info!(
        "Creating experiment '{}' for user: {} in workspace: {}",
        request.name, user.email, request.workspace_id
    );
    debug!("Experiment request: {:?}", request);
    let user_uuid = parse_uuid(&user.id, "user_id")?;

    // TODO: Handler anonymous experiments.
    let workspace_uuid = parse_uuid(&request.workspace_id, "workspace_id")?;
    let mut tx = app_state.db_pool.begin().await?;
    let access_check = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM user_workspaces WHERE workspace_id = $1 AND user_id = $2",
    )
    .bind(workspace_uuid)
    .bind(user_uuid)
    .fetch_one(&mut *tx)
    .await?;
    if access_check.0 == 0 {
        return Err(AppError::Forbidden(
            "Access denied to workspace".to_string(),
        ));
    }

    let experiment_result = sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>(
        "INSERT INTO experiment (name, description, tags, hyperparams) VALUES ($1, $2, $3, $4) RETURNING id::text, name, description, hyperparams, tags, created_at, updated_at",
    )
    .bind(&request.name)
    .bind(request.description.unwrap_or_default())
    .bind(request.tags.unwrap_or_default())
    .bind(request.hyperparams)
    .fetch_one(&mut *tx)
    .await?;
    let (experiment_id, name, description, hyperparams, tags, created_at, updated_at) =
        experiment_result;

    sqlx::query("INSERT INTO workspace_experiments (workspace_id, experiment_id) VALUES ($1, $2)")
        .bind(workspace_uuid)
        .bind(Uuid::parse_str(&experiment_id).unwrap())
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;

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

    Ok((
        StatusCode::CREATED,
        Json(Response {
            status: 201,
            data: Some(experiment),
        }),
    )
        .into_response())
}

pub async fn list_experiments(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Query(query): Query<ListExperimentsQuery>,
) -> AppResult<impl IntoResponse> {
    let user_uuid = parse_uuid(&user.id, "user_id")?;

    let result = if let Some(workspace_id) = &query.workspace {
        let workspace_uuid = parse_uuid(workspace_id, "workspace_id")?;

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

            Ok(Json(Response {
                status: 200,
                data: Some(experiments),
            })
            .into_response())
        }
        Err(e) => Err(e.into()),
    }
}

// Get single experiment
pub async fn get_experiment(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(experiment_id): Path<String>,
) -> AppResult<impl IntoResponse> {
    let user_uuid = parse_uuid(&user.id, "user_id")?;
    let experiment_uuid = parse_uuid(&experiment_id, "experiment_id")?;

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

            Ok(Json(Response {
                status: 200,
                data: Some(experiment),
            })
            .into_response())
        }
        Err(sqlx::Error::RowNotFound) => {
            Err(AppError::NotFound("Experiment not found".to_string()))
        }
        Err(e) => Err(e.into()),
    }
}

pub async fn get_experiments_batch(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Json(request): Json<BatchGetExperimentsRequest>,
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

    let mut experiment_ids: Vec<Uuid> = Vec::new();
    for raw_id in &request.ids {
        let experiment_uuid = match Uuid::parse_str(raw_id) {
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
        experiment_ids.push(experiment_uuid);
    }

    let results= sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>, String, Option<Vec<String>>)>(
        r#"
        select e.id::text, e.name, e.description, e.hyperparam, e.tags, e.created_at, e.updated_at, we.workspace_id::text, ARRAY_AGG(DISTINCT m.name) FILTER (WHERE m.name IS NOT NULL)
        from experiment e
        JOIN workspace_experiments we ON e.id = we.experiment_id
        JOIN user_workspaces uw ON we.workspace_id = uw.workspace_id
        LEFT JOIN metric m ON e.id = m.experiment_id
        WHERE e.id = ANY($1::uuid[]) AND uw.user_id = $2
        GROUP BY e.id, e.name, e.description, e.hyperparam, e.tags, e.created_at, e.updated_at, uw.workspace_id
        "#,
    )
    .bind(experiment_ids)
    .bind(user_uuid)
    .fetch_all(&app_state.db_pool)
    .await;

    let frontend_url = app_state.settings.frontend_url;
    match results {
        Ok(experiments) => {
            let experiment_results = experiments
                .into_iter()
                .map(|e| Experiment {
                    id: e.0.clone(),
                    name: e.1,
                    description: e.2,
                    hyperparams: e.3.unwrap_or_default(),
                    tags: e.4.unwrap_or_default(),
                    created_at: e.5,
                    updated_at: e.6,
                    workspace_id: Some(e.7),
                    available_metrics: e.8.unwrap_or_default(),
                    url: format!("{}/experiments/{}", frontend_url, e.0.clone()),
                })
                .collect::<Vec<Experiment>>();

            (
                StatusCode::OK,
                Json(Response {
                    status: 200,
                    data: Some(experiment_results),
                }),
            )
                .into_response()
        }
        Err(sqlx::Error::RowNotFound) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(Response {
                status: 500,
                data: Some("Some of the experiment were not found!".to_string()),
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
) -> AppResult<impl IntoResponse> {
    let user_uuid = parse_uuid(&user.id, "user_id")?;
    let experiment_uuid = parse_uuid(&id, "experiment_id")?;

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
    .await?;
    if access_check.0 == 0 {
        return Err(AppError::Forbidden(
            "Access denied to experiment".to_string(),
        ));
    }

    let result = sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)>(
        "UPDATE experiment SET name = $1, description = $2, tags = $3, updated_at = NOW() WHERE id = $4 RETURNING id::text, name, description, hyperparams, tags, created_at, updated_at",
    )
    .bind(&request.name)
    .bind(request.description.unwrap_or_default())
    .bind(request.tags.unwrap_or_default())
    .bind(experiment_uuid)
    .fetch_one(&app_state.db_pool)
    .await?;
    let frontend_url = app_state.settings.frontend_url;
    let (id, name, description, hyperparams, tags, created_at, updated_at) = result;
    let experiment = Experiment {
        id: id.to_string(),
        name,
        description,
        hyperparams: hyperparams.unwrap_or_default(),
        tags: tags.unwrap_or_default(),
        created_at,
        updated_at,
        available_metrics: vec![],
        workspace_id: None,
        url: format!("{frontend_url}/experiments/{id}"),
    };

    Ok(Json(Response {
        status: 200,
        data: Some(experiment),
    })
    .into_response())
}

// Delete experiment
pub async fn delete_experiment(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(experiment_id): Path<String>,
) -> AppResult<impl IntoResponse> {
    let user_uuid = parse_uuid(&user.id, "user_id")?;
    let experiment_uuid = parse_uuid(&experiment_id, "experiment_id")?;

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
    .await?;
    if access_check.0 == 0 {
        return Err(AppError::Forbidden(
            "Access denied - only workspace owners and admins can delete experiments".to_string(),
        ));
    }

    let delete_result = sqlx::query("DELETE FROM experiment WHERE id = $1")
        .bind(experiment_uuid)
        .execute(&app_state.db_pool)
        .await?;
    if delete_result.rows_affected() == 0 {
        return Err(AppError::NotFound("Experiment not found".to_string()));
    }

    Ok((
        StatusCode::NO_CONTENT,
        Json(Response::<()> {
            status: 204,
            data: None,
        }),
    )
        .into_response())
}

// List experiments for a specific workspace
pub async fn list_workspace_experiments(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(workspace_id): Path<String>,
) -> AppResult<impl IntoResponse> {
    let user_uuid = parse_uuid(&user.id, "user_id")?;
    let workspace_uuid = parse_uuid(&workspace_id, "workspace_id")?;

    // Check if user has access to the workspace
    let access_check = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM user_workspaces WHERE workspace_id = $1 AND user_id = $2",
    )
    .bind(workspace_uuid)
    .bind(user_uuid)
    .fetch_one(&app_state.db_pool)
    .await?;
    if access_check.0 == 0 {
        return Err(AppError::Forbidden(
            "Access denied to workspace".to_string(),
        ));
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

            Ok(Json(Response {
                status: 200,
                data: Some(experiments),
            })
            .into_response())
        }
        Err(e) => Err(e.into()),
    }
}
