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

    // Check if user has permission to invite to this workspace (OWNER or ADMIN)
    let permission_check = sqlx::query_as::<_, (i64,)>(
        r#"
        SELECT COUNT(*) FROM user_workspaces uw
        JOIN workspace_role wr ON uw.role_id = wr.id
        WHERE uw.workspace_id = $1 AND uw.user_id = $2 AND wr.name IN ('OWNER', 'ADMIN')
        "#,
    )
    .bind(workspace_uuid)
    .bind(user_uuid)
    .fetch_one(&pool)
    .await;

    match permission_check {
        Ok((count,)) if count == 0 => {
            return (
                StatusCode::FORBIDDEN,
                Json(Response {
                    status: 403,
                    data: Some("Insufficient permissions to invite users".to_string()),
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

    // Check if invitation already exists
    let existing_invitation = sqlx::query_as::<_, (i64,)>(
        "SELECT COUNT(*) FROM workspace_invitation WHERE workspace_id = $1 AND email = $2 AND status = 'PENDING'",
    )
    .bind(workspace_uuid)
    .bind(&request.email)
    .fetch_one(&pool)
    .await;

    match existing_invitation {
        Ok((count,)) if count > 0 => {
            return (
                StatusCode::CONFLICT,
                Json(Response {
                    status: 409,
                    data: Some("Invitation already exists for this email".to_string()),
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
                    data: Some("Failed to check existing invitation".to_string()),
                }),
            )
                .into_response();
        }
        _ => {}
    }

    // Create the invitation
    let invitation_result = sqlx::query_as::<_, (String, String, String, String, chrono::DateTime<chrono::Utc>)>(
        r#"
        INSERT INTO workspace_invitation (workspace_id, email, role_id, invited_by, status)
        VALUES ($1, $2, $3, $4, 'PENDING')
        RETURNING id::text, workspace_id::text, email, role_id::text, created_at
        "#,
    )
    .bind(workspace_uuid)
    .bind(&request.email)
    .bind(role_uuid)
    .bind(user_uuid)
    .fetch_one(&pool)
    .await;

    match invitation_result {
        Ok((id, workspace_id, email, role_id, created_at)) => {
            // Get role name
            let role_name_result = sqlx::query_as::<_, (String,)>(
                "SELECT name FROM workspace_role WHERE id = $1",
            )
            .bind(Uuid::parse_str(&role_id).unwrap())
            .fetch_one(&pool)
            .await;

            let role_name = match role_name_result {
                Ok((name,)) => name,
                Err(_) => "UNKNOWN".to_string(),
            };

            let invitation = WorkspaceInvitation {
                id,
                workspace_id,
                email,
                role: role_name,
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
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to create invitation".to_string()),
                }),
            )
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

    // Fetch invitations sent by the user
    let invitations_result = sqlx::query_as::<
        _,
        (
            String,
            String,
            String,
            String,
            String,
            chrono::DateTime<chrono::Utc>,
        ),
    >(
        r#"
        SELECT 
            wi.id::text,
            w.name as workspace_name,
            wi.email,
            wr.name as role,
            wi.email as from_email,
            wi.created_at
        FROM workspace_invitation wi
        JOIN workspace w ON wi.workspace_id = w.id
        JOIN workspace_role wr ON wi.role_id = wr.id
        WHERE wi.invited_by = $1 AND wi.status = 'PENDING'
        ORDER BY wi.created_at DESC
        "#,
    )
    .bind(user_uuid)
    .fetch_all(&pool)
    .await;

    match invitations_result {
        Ok(rows) => {
            let invitations: Vec<WorkspaceInvitation> = rows
                .into_iter()
                .map(|(id, workspace_name, email, role, from_email, created_at)| {
                    WorkspaceInvitation {
                        id,
                        workspace_id: workspace_name,
                        email,
                        role,
                        from: from_email,
                        created_at,
                    }
                })
                .collect();

            Json(Response {
                status: 200,
                data: Some(invitations),
            })
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

    let status = match query.action.as_str() {
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

    // Check if invitation exists and is for this user
    let invitation_check = sqlx::query_as::<_, (String, String, String)>(
        r#"
        SELECT workspace_id::text, role_id::text, email
        FROM workspace_invitation
        WHERE id = $1 AND email = $2 AND status = 'PENDING'
        "#,
    )
    .bind(invitation_uuid)
    .bind(&user.email)
    .fetch_one(&pool)
    .await;

    let (workspace_id, role_id, _) = match invitation_check {
        Ok(row) => row,
        Err(_) => {
            return (
                StatusCode::NOT_FOUND,
                Json(Response {
                    status: 404,
                    data: Some("Invitation not found or already processed".to_string()),
                }),
            )
                .into_response();
        }
    };

    let workspace_uuid = match Uuid::parse_str(&workspace_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Invalid workspace ID in invitation".to_string()),
                }),
            )
                .into_response();
        }
    };

    let role_uuid = match Uuid::parse_str(&role_id) {
        Ok(uuid) => uuid,
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Invalid role ID in invitation".to_string()),
                }),
            )
                .into_response();
        }
    };

    // Update invitation status
    let update_result = sqlx::query(
        "UPDATE workspace_invitation SET status = $1 WHERE id = $2",
    )
    .bind(status)
    .bind(invitation_uuid)
    .execute(&pool)
    .await;

    match update_result {
        Ok(_) => {
            if status == "ACCEPTED" {
                // Add user to workspace
                let add_user_result = sqlx::query(
                    "INSERT INTO user_workspaces (user_id, workspace_id, role_id) VALUES ($1, $2, $3)",
                )
                .bind(user_uuid)
                .bind(workspace_uuid)
                .bind(role_uuid)
                .execute(&pool)
                .await;

                match add_user_result {
                    Ok(_) => Json(Response {
                        status: 200,
                        data: Some("Invitation accepted and user added to workspace"),
                    })
                    .into_response(),
                    Err(e) => {
                        eprintln!("Database error adding user to workspace: {}", e);
                        (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(Response {
                                status: 500,
                                data: Some("Failed to add user to workspace".to_string()),
                            }),
                        )
                            .into_response()
                    }
                }
            } else {
                Json(Response {
                    status: 200,
                    data: Some("Invitation denied"),
                })
                .into_response()
            }
        }
        Err(e) => {
            eprintln!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(Response {
                    status: 500,
                    data: Some("Failed to update invitation".to_string()),
                }),
            )
                .into_response()
        }
    }
}