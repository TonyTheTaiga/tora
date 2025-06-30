use axum::{Json, Router, routing::get, routing::post};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use std::env;
use tower_http::{
    cors::CorsLayer,
    services::{ServeDir, ServeFile},
};

mod repos;

#[derive(Serialize, Deserialize)]
struct Ping {
    msg: String,
}

async fn ping(Json(payload): Json<Ping>) -> Json<Ping> {
    Json(Ping {
        msg: format!("pong: {}", payload.msg),
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pool = match env::var("SUPABASE_PASSWORD") {
        Ok(value) => {
            let connection_string = format!(
                "postgresql://postgres.hecctslcfhdrpnwovwbc:{value}@aws-0-us-east-1.pooler.supabase.com:5432/postgres",
            );
            Some(PgPoolOptions::new().connect(&connection_string).await?)
        }
        Err(_) => {
            eprintln!("Warning: SUPABASE_PASSWORD environment variable not set");
            None
        }
    };

    if let Some(ref pool) = pool {
        let row: (i64,) = sqlx::query_as("select count(*) from experiment;")
            .fetch_one(pool)
            .await?;
        println!("{}", row.0);
    }

    let api_routes = Router::new()
        .route("/workspaces", get(repos::workspace::list_workspaces))
        .route("/health", get(|| async { "OK" }))
        .route("/ping", post(ping));

    let static_files = ServeDir::new("../web-new/build")
        .not_found_service(ServeFile::new("../web-new/build/index.html"));

    let app = Router::new()
        .nest("/api", api_routes)
        .fallback_service(static_files)
        .layer(CorsLayer::permissive());

    println!("Server starting on 0.0.0.0:8080");
    println!("API available at /api/*");
    println!("Static files served from ../web-new/build");

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
