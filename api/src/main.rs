use axum::Router;
// use sqlx::postgres::PgPoolOptions;
use std::env;
use tokio::signal;
use tower_http::services::{ServeDir, ServeFile};

mod handlers;
mod middleware;
mod ntypes;
mod repos;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // let pool = match env::var("SUPABASE_PASSWORD") {
    //     Ok(value) => {
    //         let connection_string = format!(
    //             "postgresql://postgres.hecctslcfhdrpnwovwbc:{value}@aws-0-us-east-1.pooler.supabase.com:5432/postgres",
    //         );
    //         Some(PgPoolOptions::new().connect(&connection_string).await?)
    //     }
    //     Err(_) => {
    //         eprintln!("Warning: SUPABASE_PASSWORD environment variable not set");
    //         None
    //     }
    // };

    // if let Some(ref pool) = pool {
    //     let row: (i64,) = sqlx::query_as("select count(*) from experiment;")
    //         .fetch_one(pool)
    //         .await?;
    //     println!("{}", row.0);
    // }

    let api_routes = handlers::api_routes();
    let static_dir = env::var("STATIC_FILES_PATH").expect("STATIC_FILES_PATH not set.");
    let spa = ServeDir::new(&static_dir)
        .append_index_html_on_directories(true)
        .fallback(ServeFile::new(format!("{static_dir}/200.html")));

    let protected_spa = Router::new()
        .fallback_service(spa)
        .layer(axum::middleware::from_fn(
            crate::middleware::auth::ui_auth_middleware,
        ));

    let app = Router::new()
        .nest("/api", api_routes)
        .fallback_service(protected_spa);

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
