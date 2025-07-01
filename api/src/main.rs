use axum::{Json, Router, routing::get, routing::post};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPoolOptions;
use std::env;
use tokio::signal;
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

    let static_dir = env::var("STATIC_FILES_PATH").unwrap_or_else(|_| {
        if std::path::Path::new("./static").exists() {
            "./static".to_string()
        } else {
            "../web-new/build".to_string()
        }
    });

    let static_files = ServeDir::new(&static_dir)
        .not_found_service(ServeFile::new(format!("{static_dir}/index.html")));

    let app = Router::new()
        .nest("/api", api_routes)
        .fallback_service(static_files)
        .layer(CorsLayer::permissive());

    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    println!("Shutting down gracefully...");
}
