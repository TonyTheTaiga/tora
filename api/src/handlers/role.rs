use crate::handlers::AppResult;
use crate::state::AppState;
use crate::types::{Response, WorkspaceRole};
use axum::{Json, extract::State, response::IntoResponse};

pub async fn list_workspace_roles(
    State(app_state): State<AppState>,
) -> AppResult<impl IntoResponse> {
    let roles = sqlx::query_as::<_, WorkspaceRole>(
        r#"
        SELECT id::text, name
        FROM workspace_role
        ORDER BY name
        "#,
    )
    .fetch_all(&app_state.db_pool)
    .await?;

    Ok(Json(Response {
        status: 200,
        data: Some(roles),
    })
    .into_response())
}
