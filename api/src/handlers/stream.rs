use crate::state::AppState;
use axum::body::Bytes;
use axum::{
    extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
    extract::{Path, State},
    response::Response,
};
use fred::prelude::{EventInterface, PubsubInterface};
use serde::{Deserialize, Serialize};

#[derive(Clone, Deserialize, Serialize)]
struct LogMessage {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    step: Option<i64>,
    value: f64,
}

pub async fn stream_logs(
    ws: WebSocketUpgrade,
    State(app_state): State<AppState>,
    Path(experiment_id): Path<String>,
) -> Response {
    ws.on_upgrade(|socket| handle_socket(socket, experiment_id, app_state))
}

async fn handle_socket(mut socket: WebSocket, experiment_id: String, app_state: AppState) {
    let subscriber = app_state.vk_pool.next_connected();
    subscriber
        .subscribe(format!("log:exp:{experiment_id}"))
        .await
        .expect("Failed to subscribe");

    let mut message_stream = subscriber.message_rx();
    loop {
        tokio::select! {
            msg = message_stream.recv() => {
                match msg {
                    Ok(message) => {
                        // Convert fred::Value -> Vec<u8> using FromValue::convert
                        let raw: Vec<u8> = match message.value.convert() {
                            Ok(v) => v,
                            Err(e) => {
                                eprintln!("Failed to convert pubsub value to bytes: {e}");
                                break;
                            }
                        };

                        if let Err(e) = serde_json::from_slice::<LogMessage>(&raw) {
                            if let Ok(as_str) = std::str::from_utf8(&raw) {
                                eprintln!("Invalid payload for LogMessage: {e}; raw={as_str}");
                            } else {
                                eprintln!("Invalid payload for LogMessage: {e}; raw=<non-utf8>");
                            }
                        }

                        match String::from_utf8(raw) {
                            Ok(s) => {
                                if let Err(e) = socket.send(WsMessage::Text(s.into())).await {
                                    eprintln!("WebSocket send error: {e}");
                                    break;
                                }
                            }
                            Err(e) => {
                                let bytes = Bytes::from(e.into_bytes());
                                if let Err(e) = socket.send(WsMessage::Binary(bytes)).await {
                                    eprintln!("WebSocket send error: {e}");
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Redis message stream error: {e}");
                        break;
                    }
                }
            }

            ws = socket.recv() => {
                match ws {
                    Some(Ok(WsMessage::Ping(p))) => {
                        let _ = socket.send(WsMessage::Pong(p)).await;
                    }
                    Some(Ok(WsMessage::Close(_))) | None => {
                        break;
                    }
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        eprintln!("WebSocket recv error: {e}");
                        break;
                    }
                }
            }
        }
    }

    if let Err(e) = subscriber
        .unsubscribe(format!("log:exp:{experiment_id}"))
        .await
    {
        eprintln!("Unsubscribe error: {e}");
    }
}
