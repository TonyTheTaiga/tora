use axum::{
    Extension, Json,
    extract::Path,
    http::StatusCode,
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use crate::middleware::auth::AuthenticatedUser;
use crate::ntypes::Response;

#[derive(Serialize, Deserialize, Debug)]
pub struct Metric {
    pub id: i64,
    pub experiment_id: String,
    pub name: String,
    pub value: f64,
    pub step: Option<i64>,
    pub metadata: Option<serde_json::Value>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Deserialize)]
pub struct CreateMetricRequest {
    pub name: String,
    pub value: f64,
    pub step: Option<i64>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Deserialize)]
pub struct BatchCreateMetricsRequest {
    pub metrics: Vec<CreateMetricRequest>,
}

// Get metrics for an experiment
pub async fn get_metrics(
    Extension(_user): Extension<AuthenticatedUser>,
    Path(experiment_id): Path<String>,
) -> impl IntoResponse {
    // Mock data - will be replaced with database query
    let metrics = vec![
        Metric {
            id: 1,
            experiment_id: experiment_id.clone(),
            name: "accuracy".to_string(),
            value: 0.95,
            step: Some(100),
            metadata: None,
            created_at: chrono::Utc::now(),
        },
        Metric {
            id: 2,
            experiment_id: experiment_id.clone(),
            name: "loss".to_string(),
            value: 0.05,
            step: Some(100),
            metadata: None,
            created_at: chrono::Utc::now(),
        },
    ];

    Json(Response {
        status: 200,
        data: Some(metrics),
    })
}

// Create single metric
pub async fn create_metric(
    Extension(_user): Extension<AuthenticatedUser>,
    Path(experiment_id): Path<String>,
    Json(request): Json<CreateMetricRequest>,
) -> impl IntoResponse {
    let metric = Metric {
        id: rand::random::<i64>().abs(),
        experiment_id,
        name: request.name,
        value: request.value,
        step: request.step,
        metadata: request.metadata,
        created_at: chrono::Utc::now(),
    };

    (
        StatusCode::CREATED,
        Json(Response {
            status: 201,
            data: Some(metric),
        }),
    )
}

// Batch create metrics
pub async fn batch_create_metrics(
    Extension(_user): Extension<AuthenticatedUser>,
    Path(experiment_id): Path<String>,
    Json(request): Json<BatchCreateMetricsRequest>,
) -> impl IntoResponse {
    let metrics: Vec<Metric> = request.metrics.into_iter().map(|req| {
        Metric {
            id: rand::random::<i64>().abs(),
            experiment_id: experiment_id.clone(),
            name: req.name,
            value: req.value,
            step: req.step,
            metadata: req.metadata,
            created_at: chrono::Utc::now(),
        }
    }).collect();

    (
        StatusCode::CREATED,
        Json(Response {
            status: 201,
            data: Some(metrics),
        }),
    )
}

// Export metrics as CSV
pub async fn export_metrics_csv(
    Extension(_user): Extension<AuthenticatedUser>,
    Path(experiment_id): Path<String>,
) -> impl IntoResponse {
    use axum::http::HeaderMap;
    
    // Mock CSV data - will be replaced with actual CSV generation
    let csv_data = format!(
        "id,experiment_id,name,value,step,created_at\n1,{},accuracy,0.95,100,2024-01-01T00:00:00Z\n2,{},loss,0.05,100,2024-01-01T00:00:00Z",
        experiment_id, experiment_id
    );

    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "text/csv".parse().unwrap());
    headers.insert("Content-Disposition", format!("attachment; filename=\"metrics_{}.csv\"", experiment_id).parse().unwrap());
    
    (StatusCode::OK, headers, csv_data)
}