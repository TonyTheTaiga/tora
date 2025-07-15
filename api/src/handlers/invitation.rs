use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use crate::types::{CreateInvitationRequest, InvitationActionQuery, Response, WorkspaceInvitation};
use axum::{
    Extension, Json,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
};

use uuid::Uuid;

pub async fn create_invitation(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Json(request): Json<CreateInvitationRequest>,
) -> impl IntoResponse {
    let from_user_uuid = match Uuid::parse_str(&user.id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid sender user ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    let to_user_result = sqlx::query_as::<_, (Uuid,)>("SELECT id FROM auth.users WHERE email = $1")
        .bind(&request.email)
        .fetch_optional(&app_state.db_pool)
        .await;

    let to_user_uuid = match to_user_result {
        Ok(Some((uuid,))) => uuid,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(Response {
                    status: 404,
                    data: Some("Recipient user not found".to_string()),
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
                    data: Some("Failed to look up recipient user".to_string()),
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

    let role_uuid = match Uuid::parse_str(&request.role_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid role ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    let result = sqlx::query_as::<_, WorkspaceInvitation>(
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
    .bind("PENDING") // Initial status
    .fetch_one(&app_state.db_pool)
    .await;

    match result {
        Ok(invitation) => (
            StatusCode::CREATED,
            Json(Response {
                status: 201,
                data: Some(invitation),
            }),
        )
            .into_response(),
        Err(e) => {
            eprintln!("Database error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to create invitation".to_string()),
                }),
            )
                .into_response()
        }
    }
}

pub async fn list_invitations(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
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

    let result = sqlx::query_as::<_, WorkspaceInvitation>(
        r#"
        SELECT
            wi.id::text,
            w.id::text as workspace_id,
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
    .await;

    match result {
        Ok(invitations) => Json(Response {
            status: 200,
            data: Some(invitations),
        })
        .into_response(),
        Err(e) => {
            eprintln!("Database error: {e}");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to fetch invitations".to_string()),
                }),
            )
                .into_response()
        }
    }
}

pub async fn respond_to_invitation(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Query(query): Query<InvitationActionQuery>,
) -> impl IntoResponse {
    let invitation_uuid = match Uuid::parse_str(&query.invitation_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid invitation ID".to_string()),
                }),
            )
                .into_response();
        }
    };

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

    let mut tx = match app_state.db_pool.begin().await {
        Ok(tx) => tx,
        Err(e) => {
            eprintln!("Failed to begin transaction: {e}");
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to respond to invitation".to_string()),
                }),
            )
                .into_response();
        }
    };

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
    .await;

    let (to_user_id, workspace_id, role_id, current_status) = match invitation_details {
        Ok(Some((to, ws, role, status))) => (to, ws, role, status),
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(Response {
                    status: 404,
                    data: Some("Invitation not found or not for this user".to_string()),
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
                    data: Some("Failed to fetch invitation details".to_string()),
                }),
            )
                .into_response();
        }
    };

    if current_status != "PENDING" {
        return (
            StatusCode::BAD_REQUEST,
            Json(Response {
                status: 400,
                data: Some("Invitation is no longer pending".to_string()),
            }),
        )
            .into_response();
    }

    let new_status = match query.action.as_str() {
        "accept" => "ACCEPTED",
        "deny" => "DENIED",
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid action. Use 'accept' or 'deny'".to_string()),
                }),
            )
                .into_response();
        }
    };

    let update_result = sqlx::query(
        r#"
        UPDATE workspace_invitations
        SET status = $1
        WHERE id = $2
        "#,
    )
    .bind(new_status)
    .bind(invitation_uuid)
    .execute(&mut *tx)
    .await;

    if let Err(e) = update_result {
        eprintln!("Failed to update invitation status: {e}");
        let _ = tx.rollback().await;
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(Response {
                status: 500,
                data: Some("Failed to update invitation status".to_string()),
            }),
        )
            .into_response();
    }

    if new_status == "ACCEPTED" {
        let insert_user_workspace = sqlx::query(
            "INSERT INTO user_workspaces (user_id, workspace_id, role_id) VALUES ($1, $2, $3)",
        )
        .bind(to_user_id)
        .bind(workspace_id)
        .bind(role_id)
        .execute(&mut *tx)
        .await;

        if let Err(e) = insert_user_workspace {
            eprintln!("Failed to add user to workspace: {e}");
            let _ = tx.rollback().await;
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to add user to workspace".to_string()),
                }),
            )
                .into_response();
        }
    }

    if let Err(e) = tx.commit().await {
        eprintln!("Failed to commit transaction: {e}");
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(Response {
                status: 500,
                data: Some("Failed to commit transaction".to_string()),
            }),
        )
            .into_response();
    }

    Json(Response {
        status: 200,
        data: Some(format!("Invitation {}", new_status.to_lowercase())),
    })
    .into_response()
}
