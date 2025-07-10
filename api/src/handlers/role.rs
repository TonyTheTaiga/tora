use crate::ntypes::{Response, WorkspaceRole};
use axum::{Json, extract::State, http::StatusCode, response::IntoResponse};
use sqlx::PgPool;

pub async fn list_workspace_roles(State(pool): State<PgPool>) -> impl IntoResponse {
    let result = sqlx::query_as::<_, WorkspaceRole>(
        r#"
        SELECT id::text, name
        FROM workspace_role
        ORDER BY name
        "#,
    )
    .fetch_all(&pool)
    .await;

    match result {
        Ok(roles) => Json(Some(roles)).into_response(),
        Err(e) => {
            eprintln!("Database error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to fetch workspace roles".to_string()),
                }),
            )
                .into_response()
        }
    }
}
