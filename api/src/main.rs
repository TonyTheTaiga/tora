use axum::{Router, routing::get};
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

    let app = Router::new()
        .route("/", get(|| async { "Hello, World!" }))
        .route("/workspaces", get(repos::workspace::list_workspaces));
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
