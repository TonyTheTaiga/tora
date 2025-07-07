use axum::{
    Extension, Json,
    extract::Query,
    http::StatusCode,
    response::IntoResponse,
};
use serde::Deserialize;
use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes::{WorkspaceInvitation, CreateInvitationRequest, Response};

#[derive(Deserialize)]
pub struct InvitationActionQuery {
    #[serde(rename = "invitationId")]
    pub invitation_id: String,
    pub action: String, // "accept" or "deny"
}

pub async fn create_invitation(
    Extension(user): Extension<AuthenticatedUser>,
    Json(request): Json<CreateInvitationRequest>,
) -> impl IntoResponse {
    let invitation = WorkspaceInvitation {
        id: format!("inv_{}", uuid::Uuid::new_v4()),
        workspace_id: request.workspace_id,
        email: request.email,
        role: request.role_id,
        from: user.email,
        created_at: chrono::Utc::now(),
    };

    (
        StatusCode::CREATED,
        Json(Response {
            status: 201,
            data: Some(invitation),
        }),
    )
}

pub async fn list_invitations(
    Extension(_user): Extension<AuthenticatedUser>,
) -> impl IntoResponse {
    // Mock data - will be replaced with database query
    let invitations = vec![
        WorkspaceInvitation {
            id: "inv_1".to_string(),
            workspace_id: "Data Science Team".to_string(),
            email: "user@example.com".to_string(),
            role: "ADMIN".to_string(),
            from: "john@example.com".to_string(),
            created_at: chrono::Utc::now(),
        }
    ];

    Json(Response {
        status: 200,
        data: Some(invitations),
    })
}

pub async fn respond_to_invitation(
    Extension(_user): Extension<AuthenticatedUser>,
    Query(query): Query<InvitationActionQuery>,
) -> impl IntoResponse {
    let action = match query.action.as_str() {
        "accept" => "accepted",
        "deny" => "denied",
        _ => return (StatusCode::BAD_REQUEST, Json(Response {
            status: 400,
            data: Some("Invalid action. Use 'accept' or 'deny'"),
        })).into_response(),
    };

    // Mock - will be replaced with database update
    println!("Invitation {} {}", query.invitation_id, action);

    Json(Response {
        status: 200,
        data: Some(format!("Invitation {}", action)),
    }).into_response()
}