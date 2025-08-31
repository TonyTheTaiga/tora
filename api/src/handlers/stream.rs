use axum::{
    extract::Path,
    extract::ws::{WebSocket, WebSocketUpgrade},
    response::Response,
};

pub async fn stream_logs(Path(experiment_id): Path<String>, ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    while let Some(msg) = socket.recv().await {
        let msg = if let Ok(msg) = msg {
            msg
        } else {
            return;
        };

        if socket.send(msg).await.is_err() {
            return;
        }
    }
}
