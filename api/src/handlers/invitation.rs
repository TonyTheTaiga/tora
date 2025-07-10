use axum::{
    Extension, Json,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
};
use serde::Deserialize;
use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes::{WorkspaceInvitation, CreateInvitationRequest, Response};
use sqlx::PgPool;
use uuid::Uuid;

#[derive(Deserialize)]
pub struct InvitationActionQuery {
    #[serde(rename = "invitationId")]
    pub invitation_id: String,
    pub action: String, // "accept" or "deny"
}

pub async fn create_invitation(
    Extension(user): Extension<AuthenticatedUser>,
    State(pool): State<PgPool>,
    Json(request): Json<CreateInvitationRequest>,
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

    // Check if the user has permission to invite to this workspace
    let permission_check = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM user_workspaces WHERE user_id = $1 AND workspace_id = $2",
    )
    .bind(user_uuid)
    .bind(workspace_uuid)
    .fetch_one(&pool)
    .await;

    match permission_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::FORBIDDEN,
                Json(Response {
                    status: 403,
                    data: Some("You don't have permission to invite users to this workspace".to_string()),
                }),
            )
                .into_response();
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to check permissions".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    // Get the target user ID by email (assuming they exist in auth.users)
    let target_user_result = sqlx::query_as::<_, (String,)>(
        "SELECT id::text FROM auth.users WHERE email = $1",
    )
    .bind(&request.email)
    .fetch_optional(&pool)
    .await;

    let target_user_id = match target_user_result {
        Ok(Some((id,))) => id,
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(Response {
                    status: 404,
                    data: Some("User with this email not found".to_string()),
                }),
            )
                .into_response();
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to find user".to_string()),
                }),
            )
                .into_response();
        }
    };

    let target_user_uuid = match Uuid::parse_str(&target_user_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid target user ID".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Check if invitation already exists
    let existing_invitation = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM workspace_invitations WHERE workspace_id = $1 AND \"to\" = $2 AND status = 'pending'",
    )
    .bind(workspace_uuid)
    .bind(target_user_uuid)
    .fetch_one(&pool)
    .await;

    match existing_invitation {
        Ok((count,)) if count > 0 => {
            return (
                StatusCode::CONFLICT,
                Json(Response {
                    status: 409,
                    data: Some("Invitation already exists for this user and workspace".to_string()),
                }),
            )
                .into_response();
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to check existing invitations".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    let mut tx = match pool.begin().await {
        Ok(tx) => tx,
        Err(e) => {
            eprintln!("Failed to begin transaction: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to create invitation".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Create the invitation
    let invitation_result = sqlx::query_as::<_, (String, chrono::DateTime<chrono::Utc>)>(
        "INSERT INTO workspace_invitations (\"to\", \"from\", workspace_id, role_id, status) VALUES ($1, $2, $3, $4, 'pending') RETURNING id::text, created_at",
    )
    .bind(target_user_uuid)
    .bind(user_uuid)
    .bind(workspace_uuid)
    .bind(role_uuid)
    .fetch_one(&mut *tx)
    .await;

    match invitation_result {
        Ok((id, created_at)) => {
            if let Err(e) = tx.commit().await {
                eprintln!("Failed to commit transaction: {}", e);
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(Response {
                        status: 500,
                        data: Some("Failed to create invitation".to_string()),
                    }),
                )
                    .into_response();
            }

            let invitation = WorkspaceInvitation {
                id,
                workspace_id: request.workspace_id,
                email: request.email,
                role: request.role_id,
                from: user.email,
                created_at,
            };

            (
                StatusCode::CREATED,
                Json(Response {
                    status: 201,
                    data: Some(invitation),
                }),
            )
                .into_response()
        }
        Err(e) => {
            eprintln!("Failed to create invitation: {}", e);
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
    State(pool): State<PgPool>,
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

    let result = sqlx::query_as::<_, (String, String, String, String, chrono::DateTime<chrono::Utc>)>(
        r#"
        SELECT wi.id::text, w.name as workspace_name, u.email as from_email, wr.name as role_name, wi.created_at
        FROM workspace_invitations wi
        JOIN workspace w ON wi.workspace_id = w.id
        JOIN auth.users u ON wi."from" = u.id
        JOIN workspace_role wr ON wi.role_id = wr.id
        WHERE wi."to" = $1 AND wi.status = 'pending'
        ORDER BY wi.created_at DESC
        "#,
    )
    .bind(user_uuid)
    .fetch_all(&pool)
    .await;

    match result {
        Ok(rows) => {
            let invitations: Vec<WorkspaceInvitation> = rows
                .into_iter()
                .map(|(id, workspace_name, from_email, role_name, created_at)| {
                    WorkspaceInvitation {
                        id,
                        workspace_id: workspace_name,
                        email: user.email.clone(),
                        role: role_name,
                        from: from_email,
                        created_at,
                    }
                })
                .collect();

            Json(Response {
                status: 200,
                data: Some(invitations),
            })
                .into_response()
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
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
    State(pool): State<PgPool>,
    Query(query): Query<InvitationActionQuery>,
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

    let new_status = match query.action.as_str() {
        "accept" => "accepted",
        "deny" => "denied",
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(Response {
                    status: 400,
                    data: Some("Invalid action. Use 'accept' or 'deny'"),
                }),
            )
                .into_response();
        }
    };

    let mut tx = match pool.begin().await {
        Ok(tx) => tx,
        Err(e) => {
            eprintln!("Failed to begin transaction: {}", e);
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

    // Get the invitation details
    let invitation_result = sqlx::query_as::<_, (String, String)>(
        "SELECT workspace_id::text, role_id::text FROM workspace_invitations WHERE id = $1 AND \"to\" = $2 AND status = 'pending'",
    )
    .bind(invitation_uuid)
    .bind(user_uuid)
    .fetch_optional(&mut *tx)
    .await;

    let (workspace_id, role_id) = match invitation_result {
        Ok(Some((ws_id, r_id))) => (ws_id, r_id),
        Ok(None) => {
            return (
                StatusCode::NOT_FOUND,
                Json(Response {
                    status: 404,
                    data: Some("Invitation not found or already processed".to_string()),
                }),
            )
                .into_response();
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to find invitation".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Update invitation status
    let update_result = sqlx::query(
        "UPDATE workspace_invitations SET status = $1 WHERE id = $2",
    )
    .bind(&new_status)
    .bind(invitation_uuid)
    .execute(&mut *tx)
    .await;

    if let Err(e) = update_result {
        eprintln!("Failed to update invitation: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(Response {
                status: 500,
                data: Some("Failed to update invitation".to_string()),
            }),
        )
            .into_response();
    }

    // If accepted, add user to workspace
    if new_status == "accepted" {
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

        let role_uuid = match Uuid::parse_str(&role_id) {
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

        let add_user_result = sqlx::query(
            "INSERT INTO user_workspaces (user_id, workspace_id, role_id) VALUES ($1, $2, $3)",
        )
        .bind(user_uuid)
        .bind(workspace_uuid)
        .bind(role_uuid)
        .execute(&mut *tx)
        .await;

        if let Err(e) = add_user_result {
            eprintln!("Failed to add user to workspace: {}", e);
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
        eprintln!("Failed to commit transaction: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(Response {
                status: 500,
                data: Some("Failed to respond to invitation".to_string()),
            }),
        )
            .into_response();
    }

    Json(Response {
        status: 200,
        data: Some(format!("Invitation {}", new_status)),
    })
        .into_response()
}