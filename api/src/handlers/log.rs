use crate::handlers::{AppError, AppResult, parse_uuid};
use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use crate::types::{BatchCreateLogsRequest, CreateLogRequest, Log, Response};
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

    let result = sqlx::query_as::<_, (i64, String, String, String, f64, Option<f64>, Option<serde_json::Value>, chrono::DateTime<chrono::Utc>)>(
        "SELECT id, experiment_id::text, msg_id::text, name, value::float8, step::float8, metadata, created_at FROM log WHERE experiment_id = $1 ORDER BY created_at DESC",
    )
    .bind(experiment_uuid)
    .fetch_all(&app_state.db_pool)
    .await?;

    let metrics: Vec<Log> = result
        .into_iter()
        .map(
            |(id, experiment_id, msg_id, name, value, step, metadata, created_at)| Log {
                id,
                experiment_id,
                msg_id,
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
    Json(request): Json<CreateLogRequest>,
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

    let result = sqlx::query_as::<_, (i64, String, String, String, f64, Option<i64>, Option<serde_json::Value>, chrono::DateTime<chrono::Utc>)>(
        "INSERT INTO log (experiment_id, msg_id, name, value, step, metadata) VALUES ($1, $2, $3, $4, $5, $6) RETURNING id, experiment_id::text, msg_id::text, name, value::float8, step::float8, metadata, created_at",
    )
    .bind(experiment_uuid)
    .bind(uuid::Uuid::try_parse(&request.msg_id).expect("Failed to parse UUID"))
    .bind(&request.name)
    .bind(request.value)
    .bind(request.step.map(|s| s as i64))
    .bind(&request.metadata)
    .fetch_one(&app_state.db_pool)
    .await?;

    let (id, experiment_id, msg_id, name, value, step, metadata, created_at) = result;
    let metric = Log {
        id,
        experiment_id,
        msg_id,
        name,
        value,
        step,
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
    Json(request): Json<BatchCreateLogsRequest>,
) -> impl IntoResponse {
    info!(
        "Batch creating {} metrics for experiment {} by user: {}",
        request.logs.len(),
        experiment_id,
        user.email
    );
    debug!("Batch metrics request: {:?}", request);
    if request.logs.is_empty() {
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

    let names: Vec<String> = request.logs.iter().map(|m| m.name.clone()).collect();
    let msg_ids: Vec<uuid::Uuid> = request
        .logs
        .iter()
        .map(|m| uuid::Uuid::try_parse(&m.msg_id).expect("Invalid message ID"))
        .collect();
    let values: Vec<f64> = request.logs.iter().map(|m| m.value).collect();
    let steps: Vec<Option<f64>> = request
        .logs
        .iter()
        .map(|m| m.step.map(|s| s as f64))
        .collect();
    let metadata_values: Vec<Option<serde_json::Value>> =
        request.logs.iter().map(|m| m.metadata.clone()).collect();

    let result = sqlx::query(
        r#"
            INSERT INTO public.log (experiment_id, msg_id, name, value, step, metadata)
            SELECT
              $1, t.msg_id, t.name, t.value, t.step, t.metadata
            FROM unnest(
              $2::uuid[],      -- msg_ids
              $3::text[],      -- names
              $4::float8[],    -- values
              $5::float8[],    -- steps
              $6::jsonb[]      -- metadata
            ) AS t(msg_id, name, value, step, metadata)
            "#,
    )
    .bind(experiment_uuid)
    .bind(&msg_ids)
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
        request.logs.len(),
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
