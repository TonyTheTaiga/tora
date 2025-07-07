use axum::{
    Extension, Json,
    extract::{Path, Query},
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes::Response;

#[derive(Serialize, Deserialize, Debug)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub description: String,
    pub hyperparams: serde_json::Value,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub available_metrics: Vec<String>,
    pub workspace_id: Option<String>,
}

#[derive(Deserialize)]
pub struct CreateExperimentRequest {
    #[serde(rename = "experiment-name")]
    pub name: String,
    #[serde(rename = "experiment-description")]
    pub description: String,
    #[serde(rename = "workspace-id")]
    pub workspace_id: String,
    pub tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct UpdateExperimentRequest {
    #[serde(rename = "experiment-id")]
    pub id: String,
    #[serde(rename = "experiment-name")]
    pub name: String,
    #[serde(rename = "experiment-description")]
    pub description: String,
    pub tags: Option<Vec<String>>,
}

#[derive(Deserialize)]
pub struct ListExperimentsQuery {
    pub workspace: Option<String>,
}

// List experiments
pub async fn list_experiments(
    Extension(_user): Extension<AuthenticatedUser>,
    Query(query): Query<ListExperimentsQuery>,
) -> impl IntoResponse {
    // Mock data for now - will be replaced with database queries
    let experiments = vec![
        Experiment {
            id: "exp_1".to_string(),
            name: "Baseline Model".to_string(),
            description: "Initial baseline experiment".to_string(),
            hyperparams: serde_json::json!({"learning_rate": 0.001, "batch_size": 32}),
            tags: vec!["baseline".to_string(), "initial".to_string()],
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            available_metrics: vec!["accuracy".to_string(), "loss".to_string()],
            workspace_id: query.workspace,
        }
    ];

    Json(Response {
        status: 200,
        data: Some(experiments),
    })
}

// Create experiment
pub async fn create_experiment(
    Extension(_user): Extension<AuthenticatedUser>,
    Json(request): Json<CreateExperimentRequest>,
) -> impl IntoResponse {
    let experiment = Experiment {
        id: format!("exp_{}", uuid::Uuid::new_v4()),
        name: request.name,
        description: request.description,
        hyperparams: serde_json::json!({}),
        tags: request.tags.unwrap_or_default(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        available_metrics: vec![],
        workspace_id: Some(request.workspace_id),
    };

    (
        StatusCode::CREATED,
        Json(Response {
            status: 201,
            data: Some(experiment),
        }),
    )
}

// Get single experiment
pub async fn get_experiment(
    Extension(_user): Extension<AuthenticatedUser>,
    Path(experiment_id): Path<String>,
) -> impl IntoResponse {
    // Mock data - will be replaced with database query
    let experiment = Experiment {
        id: experiment_id,
        name: "Test Experiment".to_string(),
        description: "A test experiment".to_string(),
        hyperparams: serde_json::json!({"learning_rate": 0.001}),
        tags: vec!["test".to_string()],
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        available_metrics: vec!["accuracy".to_string()],
        workspace_id: Some("workspace_1".to_string()),
    };

    Json(Response {
        status: 200,
        data: Some(experiment),
    })
}

// Update experiment
pub async fn update_experiment(
    Extension(_user): Extension<AuthenticatedUser>,
    Json(request): Json<UpdateExperimentRequest>,
) -> impl IntoResponse {
    let experiment = Experiment {
        id: request.id,
        name: request.name,
        description: request.description,
        hyperparams: serde_json::json!({}),
        tags: request.tags.unwrap_or_default(),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
        available_metrics: vec![],
        workspace_id: Some("workspace_1".to_string()),
    };

    Json(Response {
        status: 200,
        data: Some(experiment),
    })
}

// Delete experiment
pub async fn delete_experiment(
    Extension(_user): Extension<AuthenticatedUser>,
    Path(experiment_id): Path<String>,
) -> impl IntoResponse {
    // Mock deletion - will be replaced with database deletion
    println!("Deleting experiment: {}", experiment_id);

    (
        StatusCode::NO_CONTENT,
        Json(Response::<()> {
            status: 204,
            data: None,
        }),
    )
}