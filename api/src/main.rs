use axum::Router;
use fred::{
    interfaces::ClientLike,
    prelude::{Config, EventInterface, TcpConfig},
    types::Builder,
};
use sqlx::postgres::PgPoolOptions;
use std::time::Duration;
use tokio::signal;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod handlers;
mod middleware;
mod settings;
mod state;
mod types;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "api=info,tower_http=info".into()),
        )
        .with(
            tracing_subscriber::fmt::layer()
                .compact()
                .with_target(false)
                .with_thread_ids(false)
                .with_thread_names(false),
        )
        .init();

    info!("Starting Tora API server");
    let settings = settings::Settings::from_env();
    info!(
        "Loaded settings: database_url configured, frontend_url: {}",
        settings.frontend_url
    );
    info!("Connecting to database...");
    let db_pool = PgPoolOptions::new()
        .max_connections(20)
        .min_connections(5)
        .acquire_timeout(Duration::from_secs(30))
        .connect(&settings.database_url)
        .await?;
    info!("Database connection established successfully");

    info!("Creating Valkey Client");
    let vk_config = Config::from_url("redis://localhost:6379/1")?;
    let vk_client = Builder::from_config(vk_config)
        .with_connection_config(|config| {
            config.connection_timeout = Duration::from_secs(5);
            config.tcp = TcpConfig {
                nodelay: Some(true),
                ..Default::default()
            };
        })
        .build()?;

    vk_client.init().await?;
    vk_client.on_error(|(error, server)| async move {
        println!("{server:?}: connection error: {error:?}");
        Ok(())
    });

    info!("Valkey client created!");

    let app_state = state::AppState {
        db_pool,
        settings,
        vk_client,
    };
    let api_routes = handlers::api_routes(&app_state);
    let app = Router::new().nest("/api", api_routes).with_state(app_state);

    info!("Starting server on 0.0.0.0:8080");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:8080").await?;
    info!("Server listening on 0.0.0.0:8080");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown complete");
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
        _ = ctrl_c => {
            info!("Received Ctrl+C signal");
        },
        _ = terminate => {
            info!("Received terminate signal");
        },
    }
    warn!("Shutting down gracefully...");
}
