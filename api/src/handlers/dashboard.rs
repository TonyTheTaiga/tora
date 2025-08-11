use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use crate::types::{AppResult, DashboardOverview, Experiment, Response, WorkspaceSummary};
use axum::{Extension, Json, extract::State, response::IntoResponse};

pub async fn get_dashboard_overview(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
) -> AppResult<impl IntoResponse> {
    let user_uuid = crate::types::error::parse_uuid(&user.id, "user_id")?;

    let workspaces_result = sqlx::query_as::<_, (String, String, Option<String>, chrono::DateTime<chrono::Utc>, String, i64, i64)>(
        r#"
        SELECT w.id::text, w.name, w.description, w.created_at, wr.name as role,
               COALESCE(COUNT(DISTINCT we.experiment_id), 0) as experiment_count,
               COALESCE(COUNT(DISTINCT CASE WHEN e.created_at > NOW() - INTERVAL '7 days' THEN we.experiment_id END), 0) as recent_experiment_count
        FROM user_workspaces uw
        JOIN workspace w ON uw.workspace_id = w.id
        JOIN workspace_role wr ON uw.role_id = wr.id
        LEFT JOIN workspace_experiments we ON w.id = we.workspace_id
        LEFT JOIN experiment e ON we.experiment_id = e.id
        WHERE uw.user_id = $1
        GROUP BY w.id, w.name, w.description, w.created_at, wr.name
        ORDER BY w.created_at DESC
        "#,
    )
    .bind(user_uuid)
    .fetch_all(&app_state.db_pool)
    .await?;

    // Fetch recent experiments with metrics - optimized query
    let experiments_result = sqlx::query_as::<_, (String, String, Option<String>, Option<Vec<serde_json::Value>>, Option<Vec<String>>, chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>, String, Option<Vec<String>>)>(
        r#"
        SELECT DISTINCT e.id::text, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at, we.workspace_id::text,
               ARRAY_AGG(DISTINCT m.name) FILTER (WHERE m.name IS NOT NULL) as available_metrics
        FROM user_workspaces uw
        JOIN workspace_experiments we ON uw.workspace_id = we.workspace_id
        JOIN experiment e ON we.experiment_id = e.id
        LEFT JOIN metric m ON e.id = m.experiment_id
        WHERE uw.user_id = $1
        GROUP BY e.id, e.name, e.description, e.hyperparams, e.tags, e.created_at, e.updated_at, we.workspace_id
        ORDER BY e.created_at DESC
        LIMIT 10
        "#,
    )
    .bind(user_uuid)
    .fetch_all(&app_state.db_pool)
    .await?;

    let workspaces: Vec<WorkspaceSummary> = workspaces_result
        .into_iter()
        .map(
            |(
                id,
                name,
                description,
                created_at,
                role,
                experiment_count,
                recent_experiment_count,
            )| {
                WorkspaceSummary {
                    id,
                    name,
                    description,
                    created_at,
                    role,
                    experiment_count,
                    recent_experiment_count,
                }
            },
        )
        .collect();

    let frontend_url = app_state.settings.frontend_url;
    let recent_experiments: Vec<Experiment> = experiments_result
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
                    url: format!("{frontend_url:?}/experiments/{id:?}"),
                }
            },
        )
        .collect();

    let overview = DashboardOverview {
        workspaces,
        recent_experiments,
    };

    Ok(Json(Response {
        status: 200,
        data: Some(overview),
    })
    .into_response())
}
