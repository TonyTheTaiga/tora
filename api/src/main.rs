use axum::{Router, routing::get};
use tower_http::{services::{ServeDir, ServeFile}, cors::CorsLayer};
mod repos;
use sqlx::postgres::PgPoolOptions;
use std::env;

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

    // API routes
    let api_routes = Router::new()
        .route("/workspaces", get(repos::workspace::list_workspaces))
        .route("/health", get(|| async { "OK" }));

    // Static file serving for SvelteKit app
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
