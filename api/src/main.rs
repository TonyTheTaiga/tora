use axum::Router;
use sqlx::postgres::PgPoolOptions;
use std::env;
use tokio::signal;

mod handlers;
mod middleware;
mod types;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pool = match env::var("DATABASE_URL") {
        Ok(database_url) => {
            PgPoolOptions::new()
                .max_connections(5)
                .connect(&database_url)
                .await?
        }
        Err(_) => {
            eprintln!("Error: DATABASE_URL environment variable not set");
            return Err("DATABASE_URL not set".into());
        }
    };
    let api_routes = handlers::api_routes(&pool);
    let app = Router::new().nest("/api", api_routes).with_state(pool);
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
