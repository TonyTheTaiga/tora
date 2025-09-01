use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use axum::body::Bytes;
use axum::{
    Extension, Json,
    extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
    extract::{Path, Query, State},
    http::HeaderMap,
    response::{IntoResponse, Response},
};
use base64::Engine;
use fred::interfaces::KeysInterface;
use fred::prelude::{EventInterface, PubsubInterface};
use fred::types::Expiration;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Deserialize, Serialize)]
struct LogMessage {
    name: String,
    #[serde(rename = "type")]
    kind: String,
    step: Option<i64>,
    value: f64,
}

#[derive(Deserialize)]
pub struct StreamQuery {
    pub token: String,
}

pub async fn stream_logs(
    ws: WebSocketUpgrade,
    State(app_state): State<AppState>,
    Path(experiment_id): Path<String>,
    Query(params): Query<StreamQuery>,
    headers: HeaderMap,
) -> Response {
    if let Some(origin) = headers.get("origin").and_then(|h| h.to_str().ok()) {
        let allowed = app_state
            .settings
            .frontend_url
            .trim_end_matches('/')
            .eq(origin.trim_end_matches('/'));
        if !allowed {
            return axum::http::StatusCode::FORBIDDEN.into_response();
        }
    }
    ws.on_upgrade(|socket| handle_socket(socket, experiment_id, params, app_state))
}

async fn handle_socket(
    mut socket: WebSocket,
    experiment_id: String,
    params: StreamQuery,
    app_state: AppState,
) {
    if let Err(e) = validate_ws_token(&app_state, &params.token, &experiment_id).await {
        let _ = socket
            .send(WsMessage::Close(Some(axum::extract::ws::CloseFrame {
                code: 1008,
                reason: format!("unauthorized: {e}").into(),
            })))
            .await;
        return;
    }
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

#[derive(Serialize, Deserialize)]
struct WsTokenData {
    user_id: String,
    experiment_id: String,
}

async fn validate_ws_token(
    app_state: &AppState,
    token: &str,
    path_experiment_id: &str,
) -> Result<(), String> {
    if token.len() < 16 || token.len() > 128 {
        return Err("invalid token".into());
    }
    let client = app_state.vk_pool.next_connected();
    let key = format!("ws:token:{token}");
    let payload: Option<String> = client
        .get(key.clone())
        .await
        .map_err(|e| format!("valkey get error: {e}"))?;
    let Some(payload) = payload else {
        return Err("token not found or expired".into());
    };
    let _ = client.del::<i64, _>(key).await; // best-effort single-use
    let data: WsTokenData =
        serde_json::from_str(&payload).map_err(|_| "invalid token payload".to_string())?;
    if data.experiment_id != path_experiment_id {
        return Err("token experiment mismatch".into());
    }
    if Uuid::parse_str(&data.experiment_id).is_err() || Uuid::parse_str(&data.user_id).is_err() {
        return Err("invalid ids in token".into());
    }
    Ok(())
}

pub async fn create_stream_token(
    Extension(user): Extension<AuthenticatedUser>,
    State(app_state): State<AppState>,
    Path(experiment_id): Path<String>,
) -> impl IntoResponse {
    let user_uuid = match Uuid::parse_str(&user.id) {
        Ok(u) => u,
        Err(_) => {
            return (
                axum::http::StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({"error":"invalid user"})),
            )
                .into_response();
        }
    };
    let exp_uuid = match Uuid::parse_str(&experiment_id) {
        Ok(e) => e,
        Err(_) => {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error":"invalid experiment id"})),
            )
                .into_response();
        }
    };

    let access_check = sqlx::query_as::<_, (i64,)>(
        r#"
        SELECT COUNT(*) FROM experiment e
        JOIN workspace_experiments we ON e.id = we.experiment_id
        JOIN user_workspaces uw ON we.workspace_id = uw.workspace_id
        WHERE e.id = $1 AND uw.user_id = $2
        "#,
    )
    .bind(exp_uuid)
    .bind(user_uuid)
    .fetch_one(&app_state.db_pool)
    .await;
    if matches!(access_check, Err(_) | Ok((0,))) {
        return (
            axum::http::StatusCode::FORBIDDEN,
            Json(serde_json::json!({"error":"access denied"})),
        )
            .into_response();
    }

    let token = mint_random_token();
    let data = WsTokenData {
        user_id: user.id,
        experiment_id: experiment_id.clone(),
    };
    let payload = serde_json::to_string(&data).unwrap();
    let client = app_state.vk_pool.next_connected();
    let key = format!("ws:token:{token}");
    let ttl_secs: i64 = 60;
    if let Err(e) = client
        .set::<(), _, _>(key, payload, Some(Expiration::EX(ttl_secs)), None, false)
        .await
    {
        return (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error":format!("valkey error: {e}")})),
        )
            .into_response();
    }

    let resp = serde_json::json!({ "token": token, "expires_in": ttl_secs });
    (axum::http::StatusCode::OK, Json(resp)).into_response()
}

fn mint_random_token() -> String {
    let r1: u128 = rand::random();
    let r2: u128 = rand::random();
    let mut bytes = [0u8; 32];
    bytes[..16].copy_from_slice(&r1.to_le_bytes());
    bytes[16..].copy_from_slice(&r2.to_le_bytes());
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}
