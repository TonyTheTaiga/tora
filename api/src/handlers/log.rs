use crate::handlers::{AppError, AppResult, parse_uuid};
use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use crate::types::{BatchCreateMetricsRequest, CreateMetricRequest, Metric, Response};
use axum::{
    Extension, Json,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
};
use tracing::{debug, info};

pub async fn get_logs(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(experiment_id): Path<String>,
) -> AppResult<impl IntoResponse> {
    let user_uuid = parse_uuid(&user.id, "user_id")?;
    let experiment_uuid = parse_uuid(&experiment_id, "experiment_id")?;

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

    let result = sqlx::query_as::<_, (i64, String, String, f64, Option<f64>, Option<serde_json::Value>, chrono::DateTime<chrono::Utc>)>(
        "SELECT id, experiment_id::text, name, value::float8, step::float8, metadata, created_at FROM log WHERE experiment_id = $1 ORDER BY created_at DESC",
    )
    .bind(experiment_uuid)
    .fetch_all(&app_state.db_pool)
    .await?;

    let metrics: Vec<Metric> = result
        .into_iter()
        .map(
            |(id, experiment_id, name, value, step, metadata, created_at)| Metric {
                id,
                experiment_id,
                name,
                value,
                step: step.map(|s| s as i64),
                metadata,
                created_at,
            },
        )
        .collect();

    Ok(Json(Response {
        status: 200,
        data: Some(metrics),
    })
    .into_response())
}

pub async fn create_log(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(experiment_id): Path<String>,
    Json(request): Json<CreateMetricRequest>,
) -> AppResult<impl IntoResponse> {
    let user_uuid = parse_uuid(&user.id, "user_id")?;
    let experiment_uuid = parse_uuid(&experiment_id, "experiment_id")?;

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

    let result = sqlx::query_as::<_, (i64, String, String, f64, Option<f64>, Option<serde_json::Value>, chrono::DateTime<chrono::Utc>)>(
        "INSERT INTO log (experiment_id, name, value, step, metadata) VALUES ($1, $2, $3, $4, $5) RETURNING id, experiment_id::text, name, value::float8, step::float8, metadata, created_at",
    )
    .bind(experiment_uuid)
    .bind(&request.name)
    .bind(request.value)
    .bind(request.step.map(|s| s as f64))
    .bind(&request.metadata)
    .fetch_one(&app_state.db_pool)
    .await?;

    let (id, experiment_id, name, value, step, metadata, created_at) = result;
    let metric = Metric {
        id,
        experiment_id,
        name,
        value,
        step: step.map(|s| s as i64),
        metadata,
        created_at,
    };

    Ok((
        StatusCode::CREATED,
        Json(Response {
            status: 201,
            data: Some(metric),
        }),
    )
        .into_response())
}

pub async fn batch_create_logs(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(experiment_id): Path<String>,
    Json(request): Json<BatchCreateMetricsRequest>,
) -> impl IntoResponse {
    info!(
        "Batch creating {} metrics for experiment {} by user: {}",
        request.metrics.len(),
        experiment_id,
        user.email
    );
    debug!("Batch metrics request: {:?}", request);
    if request.metrics.is_empty() {
        return Ok((
            StatusCode::CREATED,
            Json(Response::<String> {
                status: 201,
                data: None,
            }),
        )
            .into_response());
    }

    let user_uuid = parse_uuid(&user.id, "user_id")?;
    let experiment_uuid = parse_uuid(&experiment_id, "experiment_id")?;

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

    let mut tx = app_state.db_pool.begin().await?;

    let names: Vec<String> = request.metrics.iter().map(|m| m.name.clone()).collect();
    let values: Vec<f64> = request.metrics.iter().map(|m| m.value).collect();
    let steps: Vec<Option<f64>> = request
        .metrics
        .iter()
        .map(|m| m.step.map(|s| s as f64))
        .collect();
    let metadata_values: Vec<Option<serde_json::Value>> =
        request.metrics.iter().map(|m| m.metadata.clone()).collect();
    let result = sqlx::query(
        r#"
        INSERT INTO log (experiment_id, name, value, step, metadata)
        SELECT $1, unnest($2::text[]), unnest($3::float8[]), unnest($4::float8[]), unnest($5::jsonb[])
        "#,
    )
    .bind(experiment_uuid)
    .bind(&names)
    .bind(&values)
    .bind(&steps)
    .bind(&metadata_values)
    .execute(&mut *tx)
    .await?;

    debug!("Successfully inserted {} metrics", result.rows_affected());
    tx.commit().await?;
    info!(
        "Successfully committed {} metrics for experiment {}",
        request.metrics.len(),
        experiment_id
    );

    Ok((
        StatusCode::CREATED,
        Json(Response::<String> {
            status: 201,
            data: None,
        }),
    )
        .into_response())
}

pub async fn export_logs_csv(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(experiment_id): Path<String>,
) -> AppResult<impl IntoResponse> {
    use axum::http::HeaderMap;

    let user_uuid = parse_uuid(&user.id, "user_id")?;
    let experiment_uuid = parse_uuid(&experiment_id, "experiment_id")?;

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

    let result = sqlx::query_as::<_, (i64, String, String, f64, Option<f64>, Option<serde_json::Value>, chrono::DateTime<chrono::Utc>)>(
        "SELECT id, experiment_id::text, name, value::float8, step::float8, metadata, created_at FROM log WHERE experiment_id = $1 ORDER BY created_at ASC",
    )
    .bind(experiment_uuid)
    .fetch_all(&app_state.db_pool)
    .await?;

    let mut csv_data = String::from("id,experiment_id,name,value,step,metadata,created_at\n");

    for (id, exp_id, name, value, step, metadata, created_at) in result {
        let step_str = step
            .map(|s| s.to_string())
            .unwrap_or_else(|| "null".to_string());
        let metadata_str = metadata
            .map(|m| m.to_string())
            .unwrap_or_else(|| "null".to_string());
        csv_data.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            id,
            exp_id,
            name,
            value,
            step_str,
            metadata_str,
            created_at.to_rfc3339()
        ));
    }

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "text/csv".parse().unwrap());
    headers.insert(
        "Content-Disposition",
        format!("attachment; filename=\"metrics_{experiment_id}.csv\"",)
            .parse()
            .unwrap(),
    );

    Ok((StatusCode::OK, headers, csv_data).into_response())
}
