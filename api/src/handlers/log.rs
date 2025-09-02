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

    let metrics = sqlx::query_as::<_, Log>(
        r#"
            SELECT
                id,
                experiment_id::text,
                msg_id::text,
                name, value::float8,
                step::int8,
                metadata,
                created_at
            FROM log 
            WHERE experiment_id = $1 
            ORDER BY created_at DESC"#,
    )
    .bind(experiment_uuid)
    .fetch_all(&app_state.db_pool)
    .await?;

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

    let msg_uuid = uuid::Uuid::try_parse(&request.msg_id).expect("Failed to parse UUID");
    let (id, experiment_id, msg_id, name, value, step, metadata, created_at) = sqlx::query_as::<
        _,
        (
            i64,
            String,
            String,
            String,
            f64,
            Option<i64>,
            Option<serde_json::Value>,
            chrono::DateTime<chrono::Utc>,
        ),
    >(
        r#"
        WITH ins AS (
          INSERT INTO public.log (experiment_id, msg_id, name, value, step, metadata)
          VALUES ($1, $2, $3, $4, $5, $6)
          RETURNING id, experiment_id, msg_id, name, value, step, metadata, created_at
        ), outbox_ins AS (
          INSERT INTO public.log_outbox (experiment_id, msg_id, payload)
          SELECT
            i.experiment_id,
            i.msg_id,
            jsonb_build_object(
              'name', i.name,
              'value', i.value,
              'step', i.step,
              'type', i.metadata->>'type'
            )
          FROM ins i
          WHERE i.metadata->>'type' = 'metric'
          ON CONFLICT (msg_id) DO NOTHING
          RETURNING 1
        )
        SELECT id,
               experiment_id::text,
               msg_id::text,
               name,
               value::float8,
               step::int8,
               metadata,
               created_at
        FROM ins
        "#,
    )
    .bind(experiment_uuid)
    .bind(msg_uuid)
    .bind(&request.name)
    .bind(request.value)
    .bind(request.step)
    .bind(&request.metadata)
    .fetch_one(&app_state.db_pool)
    .await?;

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

    let names: Vec<String> = request.logs.iter().map(|m| m.name.clone()).collect();
    let msg_ids: Vec<uuid::Uuid> = request
        .logs
        .iter()
        .map(|m| uuid::Uuid::try_parse(&m.msg_id).expect("Invalid message ID"))
        .collect();
    let values: Vec<f64> = request.logs.iter().map(|m| m.value).collect();
    let steps: Vec<Option<i64>> = request.logs.iter().map(|m| m.step).collect();
    let metadata_values: Vec<Option<serde_json::Value>> =
        request.logs.iter().map(|m| m.metadata.clone()).collect();

    let inserted_count = sqlx::query_scalar::<_, i64>(
        r#"
        WITH t AS (
          SELECT * FROM unnest(
            $1::uuid[],
            $2::text[],
            $3::float8[],
            $4::int8[],
            $5::jsonb[]
          ) AS u(msg_id, name, value, step, metadata)
        ), ins AS (
          INSERT INTO public.log (experiment_id, msg_id, name, value, step, metadata)
          SELECT $6, t.msg_id, t.name, t.value, t.step, t.metadata FROM t
          RETURNING experiment_id, msg_id, name, value, step, metadata
        ), outbox AS (
          INSERT INTO public.log_outbox (experiment_id, msg_id, payload)
          SELECT
            $6,
            i.msg_id,
            jsonb_build_object(
              'name', i.name,
              'value', i.value,
              'step', i.step,
              'type', i.metadata->>'type'
            )
          FROM ins i
          WHERE i.metadata->>'type' = 'metric'
          ON CONFLICT (msg_id) DO NOTHING
          RETURNING 1
        )
        SELECT COUNT(*) FROM ins
        "#,
    )
    .bind(&msg_ids)
    .bind(&names)
    .bind(&values)
    .bind(&steps)
    .bind(&metadata_values)
    .bind(experiment_uuid)
    .fetch_one(&app_state.db_pool)
    .await?;

    debug!("Successfully inserted {} metrics", inserted_count);
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

    let result = sqlx::query_as::<_, (i64, String, String, f64, Option<i64>, Option<serde_json::Value>, chrono::DateTime<chrono::Utc>)>(
        "SELECT id, experiment_id::text, name, value::float8, step::int8, metadata, created_at FROM log WHERE experiment_id = $1 ORDER BY created_at ASC",
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
