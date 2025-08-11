use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use crate::types::{
    AppError, AppResult, CreateInvitationRequest, InvitationActionQuery, Response,
    WorkspaceInvitation,
};
use axum::{
    Extension, Json,
    extract::{Query, State},
    response::IntoResponse,
};

use uuid::Uuid;

pub async fn create_invitation(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Json(request): Json<CreateInvitationRequest>,
) -> AppResult<impl IntoResponse> {
    let from_user_uuid = crate::types::error::parse_uuid(&user.id, "user_id")?;

    let to_user_result = sqlx::query_as::<_, (Uuid,)>("SELECT id FROM auth.users WHERE email = $1")
        .bind(&request.email)
        .fetch_optional(&app_state.db_pool)
        .await?;

    let to_user_uuid = match to_user_result {
        Some((uuid,)) => uuid,
        None => return Err(AppError::NotFound("Recipient user not found".to_string())),
    };

    let workspace_uuid = crate::types::error::parse_uuid(&request.workspace_id, "workspace_id")?;
    let role_uuid = crate::types::error::parse_uuid(&request.role_id, "role_id")?;

    let invitation = sqlx::query_as::<_, WorkspaceInvitation>(
        r#"
        INSERT INTO workspace_invitations ("to", "from", workspace_id, role_id, status)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING id::text, workspace_id::text, (SELECT email FROM auth.users WHERE id = "to") as email, (SELECT name FROM workspace_role WHERE id = role_id) as role, (SELECT email FROM auth.users WHERE id = "from") as from, created_at
        "#,
    )
    .bind(to_user_uuid)
    .bind(from_user_uuid)
    .bind(workspace_uuid)
    .bind(role_uuid)
    .bind("PENDING")
    .fetch_one(&app_state.db_pool)
    .await?;

    Ok((
        axum::http::StatusCode::CREATED,
        Json(Response {
            status: 201,
            data: Some(invitation),
        }),
    )
        .into_response())
}

pub async fn list_invitations(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
) -> AppResult<impl IntoResponse> {
    let user_uuid = crate::types::error::parse_uuid(&user.id, "user_id")?;

    let invitations = sqlx::query_as::<_, WorkspaceInvitation>(
        r#"
        SELECT
            wi.id::text,
            w.id::text as workspace_id,
            w.name as workspace_name,
            u_to.email as email,
            wr.name as role,
            u_from.email as from,
            wi.created_at
        FROM workspace_invitations wi
        JOIN workspace w ON wi.workspace_id = w.id
        JOIN workspace_role wr ON wi.role_id = wr.id
        JOIN auth.users u_to ON wi.to = u_to.id
        JOIN auth.users u_from ON wi.from = u_from.id
        WHERE wi.to = $1 AND wi.status = 'PENDING'
        ORDER BY wi.created_at DESC
        "#,
    )
    .bind(user_uuid)
    .fetch_all(&app_state.db_pool)
    .await?;

    Ok(Json(Response {
        status: 200,
        data: Some(invitations),
    })
    .into_response())
}

pub async fn respond_to_invitation(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Query(query): Query<InvitationActionQuery>,
) -> AppResult<impl IntoResponse> {
    let invitation_uuid = crate::types::error::parse_uuid(&query.invitation_id, "invitation_id")?;
    let user_uuid = crate::types::error::parse_uuid(&user.id, "user_id")?;

    let mut tx = app_state.db_pool.begin().await?;

    let invitation_details = sqlx::query_as::<_, (Uuid, Uuid, Uuid, String)>(
        r#"
        SELECT "to", workspace_id, role_id, status
        FROM workspace_invitations
        WHERE id = $1 AND "to" = $2
        FOR UPDATE
        "#,
    )
    .bind(invitation_uuid)
    .bind(user_uuid)
    .fetch_optional(&mut *tx)
    .await?;

    let (to_user_id, workspace_id, role_id, current_status) = match invitation_details {
        Some((to, ws, role, status)) => (to, ws, role, status),
        None => {
            return Err(AppError::NotFound(
                "Invitation not found or not for this user".to_string(),
            ));
        }
    };

    if current_status != "PENDING" {
        return Err(AppError::BadRequest(
            "Invitation is no longer pending".to_string(),
        ));
    }

    let new_status = match query.action.as_str() {
        "accept" => "ACCEPTED",
        "deny" => "DENIED",
        _ => {
            return Err(AppError::BadRequest(
                "Invalid action. Use 'accept' or 'deny'".to_string(),
            ));
        }
    };

    sqlx::query(
        r#"
        UPDATE workspace_invitations
        SET status = $1
        WHERE id = $2
        "#,
    )
    .bind(new_status)
    .bind(invitation_uuid)
    .execute(&mut *tx)
    .await?;

    if new_status == "ACCEPTED" {
        sqlx::query(
            "INSERT INTO user_workspaces (user_id, workspace_id, role_id) VALUES ($1, $2, $3)",
        )
        .bind(to_user_id)
        .bind(workspace_id)
        .bind(role_id)
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;

    Ok(Json(Response {
        status: 200,
        data: Some(format!("Invitation {}", new_status.to_lowercase())),
    })
    .into_response())
}
