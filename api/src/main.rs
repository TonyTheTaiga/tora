use axum::Router;
use fred::{
    interfaces::ClientLike,
    prelude::{Config, TcpConfig},
    types::Builder,
};
use sqlx::postgres::PgPoolOptions;
use std::time::Duration;
use tokio::{signal, task};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod handlers;
mod middleware;
mod settings;
mod state;
mod types;
mod worker;

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
    info!("Settings loaded!");

    info!("Connecting to database...");
    let db_pool = PgPoolOptions::new()
        .max_connections(20)
        .min_connections(5)
        .acquire_timeout(Duration::from_secs(30))
        .connect(&settings.database_url)
        .await?;
    info!("Database connection established successfully");

    info!("Creating Valkey Client");
    let vk_config = Config::from_url(&settings.redis_url)?;
    let vk_pool = Builder::from_config(vk_config)
        .with_connection_config(|config| {
            config.connection_timeout = Duration::from_secs(5);
            config.tcp = TcpConfig {
                nodelay: Some(true),
                ..Default::default()
            };
        })
        .with_performance_config(|config| config.broadcast_channel_capacity = 64)
        .build_pool(8)
        .expect("Failed to create valkey pool");

    vk_pool
        .init()
        .await
        .expect("Failed to connect to valkey instance");

    info!("Valkey client created!");

    let app_state = state::AppState {
        db_pool,
        settings,
        vk_pool,
    };

    task::spawn(worker::outbox_worker::run_worker(app_state.clone()));

    let api_routes = handlers::api_routes(&app_state);
    let app = Router::new().nest("/api", api_routes).with_state(app_state);
    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    info!("Server listening on {addr:?}");
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
