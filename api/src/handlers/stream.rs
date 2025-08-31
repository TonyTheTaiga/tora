use axum::{
    extract::Path,
    extract::ws::{WebSocket, WebSocketUpgrade},
    response::Response,
};

pub async fn stream_logs(ws: WebSocketUpgrade, Path(experiment_id): Path<String>) -> Response {
    ws.on_upgrade(|socket| handle_socket(socket, experiment_id))
}

async fn handle_socket(mut socket: WebSocket, experiment_id: String) {
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
