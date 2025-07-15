use axum::Router;
use sqlx::postgres::PgPoolOptions;
use std::time::Duration;
use tokio::signal;

mod handlers;
mod middleware;
mod settings;
mod state;
mod types;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let settings = settings::Settings::from_env();
    let pool = PgPoolOptions::new()
        .max_connections(20) // Increase based on load
        .min_connections(5)
        .acquire_timeout(Duration::from_secs(30))
        .connect(&settings.database_url)
        .await?;

    let app_state = state::AppState {
        db_pool: pool,
        settings,
    };

    let api_routes = handlers::api_routes(&app_state);
    let app = Router::new().nest("/api", api_routes).with_state(app_state);
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
