// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Mobile relay, platform adapter config, and factory reset routes.

use crate::platform::adapter::PlatformAdapter;
use crate::web::state::SharedState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Mobile Relay ─────────────────────────────────────────────────────

pub async fn relay_status(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let st = state.read().await;
    let local_ip = local_ip().unwrap_or_else(|| "127.0.0.1".to_string());

    Json(serde_json::json!({
        "available": true,
        "hostname": hostname::get()
            .ok()
            .and_then(|h| h.into_string().ok())
            .unwrap_or_else(|| "ErnOS Desktop".to_string()),
        "model_name": st.model_spec.name,
        "model_params": st.model_spec.parameter_size,
        "context_length": st.model_spec.context_length,
        "capabilities": {
            "text": st.model_spec.capabilities.text,
            "vision": st.model_spec.capabilities.vision,
            "audio": st.model_spec.capabilities.audio,
            "tool_calling": st.model_spec.capabilities.tool_calling,
            "thinking": st.model_spec.capabilities.thinking,
        },
        "pairing": {
            "qr_payload": format!("ernos://{}:3000?model={}", local_ip, st.model_spec.name),
            "ws_url": format!("ws://{}:3000/ws/relay", local_ip),
            "local_ip": local_ip,
        }
    }))
}

fn local_ip() -> Option<String> {
    let socket = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("8.8.8.8:80").ok()?;
    Some(socket.local_addr().ok()?.ip().to_string())
}

// ── Platform adapter config ─────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct PlatformConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub configured: bool,
    #[serde(flatten)]
    pub fields: HashMap<String, serde_json::Value>,
}

pub async fn get_platforms(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let st = state.read().await;
    let platforms_file = st.config.general.data_dir.join("platforms.json");
    let configs: HashMap<String, PlatformConfig> = match std::fs::read_to_string(&platforms_file) {
        Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
        Err(_) => HashMap::new(),
    };

    let sanitized: HashMap<String, serde_json::Value> = configs
        .into_iter()
        .map(|(k, v)| {
            (k, serde_json::json!({"enabled": v.enabled, "configured": v.configured}))
        })
        .collect();

    Json(serde_json::Value::Object(
        sanitized.into_iter().map(|(k, v)| (k, v)).collect(),
    ))
}

pub async fn save_platform(
    State(state): State<SharedState>,
    Path(platform): Path<String>,
    Json(body): Json<serde_json::Value>,
) -> StatusCode {
    let allowed = ["discord", "telegram", "whatsapp", "custom"];
    if !allowed.contains(&platform.as_str()) {
        return StatusCode::BAD_REQUEST;
    }

    let platforms_file = {
        let st = state.read().await;
        st.config.general.data_dir.join("platforms.json")
    };

    let mut configs: HashMap<String, serde_json::Value> =
        match std::fs::read_to_string(&platforms_file) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
            Err(_) => HashMap::new(),
        };

    let has_creds = has_credentials(&body);
    let mut entry = body.clone();
    if let Some(obj) = entry.as_object_mut() {
        obj.insert("configured".to_string(), serde_json::Value::Bool(has_creds));
    }

    let is_enabled = body.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);

    configs.insert(platform.clone(), entry);

    match serde_json::to_string_pretty(&configs)
        .ok()
        .and_then(|s| std::fs::write(&platforms_file, s).ok())
    {
        Some(_) => {
            tracing::info!(platform = %platform, enabled = is_enabled, "Platform config saved");
        }
        None => {
            tracing::warn!(platform = %platform, "Failed to write platform config");
            return StatusCode::INTERNAL_SERVER_ERROR;
        }
    }

    // Dynamically connect or disconnect the adapter at runtime
    if is_enabled && has_creds {
        let mut st = state.write().await;

        // Build a fresh adapter with the new config from the save body
        match platform.as_str() {
            "discord" => {
                let token = body.get("token").and_then(|v| v.as_str()).unwrap_or("");
                let admin = body.get("admin_id").and_then(|v| v.as_str()).unwrap_or("");
                let listen_ch = body.get("listen_channel").and_then(|v| v.as_str()).unwrap_or("");
                let listen_channels = if listen_ch.is_empty() {
                    Vec::new()
                } else {
                    listen_ch.split(',').map(|s| s.trim().to_string()).collect()
                };
                let cfg = crate::config::DiscordConfig {
                    enabled: true,
                    token: token.to_string(),
                    admin_user_id: admin.to_string(),
                    guild_id: body.get("guild_id").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                    autonomy_channel_id: body.get("autonomy_channel")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    listen_channels,
                    onboarding_channel_id: body.get("onboarding_channel_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    new_member_role_id: body.get("new_member_role_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    member_role_id: body.get("member_role_id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string(),
                    new_role_duration_days: body.get("new_role_duration_days")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(7),
                    sentinel_enabled: body.get("sentinel_enabled")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false),
                };
                let mut new_adapter = crate::platform::discord::DiscordAdapter::new(&cfg);
                if let Err(e) = new_adapter.connect().await {
                    tracing::error!(error = %e, "Failed to connect Discord adapter at runtime");
                }
                // Store HTTP handle in SharedState so thinking threads + typing indicators work
                #[cfg(feature = "discord")]
                { st.discord_http = new_adapter.http_client(); }
                st.platform_registry.replace_adapter(Box::new(new_adapter)).await;
            }
            "telegram" => {
                let token = body.get("token").and_then(|v| v.as_str()).unwrap_or("");
                let admin = body.get("admin_user_id").and_then(|v| v.as_str()).unwrap_or("");
                let cfg = crate::config::TelegramConfig {
                    enabled: true,
                    token: token.to_string(),
                    admin_user_id: admin.to_string(),
                };
                st.platform_registry.replace_adapter(Box::new(
                    crate::platform::telegram::TelegramAdapter::new(&cfg),
                )).await;
            }
            _ => {}
        }

        // Now connect the freshly configured adapter
        match st.platform_registry.connect_by_name(&platform).await {
            Ok(()) => {
                tracing::info!(platform = %platform, "Platform adapter connected at runtime");

                // Take the message receiver and spawn a live router task
                let platform_name = platform.clone();
                let mut rx_opt = None;
                for adapter in st.platform_registry.adapters_mut() {
                    if adapter.name().eq_ignore_ascii_case(&platform_name) {
                        rx_opt = adapter.take_message_receiver();
                        break;
                    }
                }

                if let Some(mut rx) = rx_opt {
                    let state_for_router = state.clone();
                    tokio::spawn(async move {
                        tracing::info!(platform = %platform_name, "Live platform router started");
                        while let Some(msg) = rx.recv().await {
                            let state = state_for_router.clone();
                            tokio::spawn(async move {
                                tracing::info!(
                                    platform = %msg.platform,
                                    user = %msg.user_name,
                                    channel = %msg.channel_id,
                                    "Platform message received — routing to inference"
                                );
                                match crate::platform::router::process_message(&state, &msg).await {
                                    Ok(reply) => {
                                        let st = state.read().await;
                                        for adapter in st.platform_registry.adapters_iter() {
                                            if adapter.name().to_lowercase() == msg.platform {
                                                if let Err(e) = adapter.reply_to_message(&msg.channel_id, &msg.message_id, &reply).await {
                                                    tracing::error!(
                                                        platform = %msg.platform, error = %e,
                                                        "Failed to send platform reply"
                                                    );
                                                }
                                                break;
                                            }
                                        }
                                    }
                                    Err(e) => {
                                        tracing::error!(
                                            platform = %msg.platform, error = %e,
                                            "Failed to process platform message"
                                        );
                                    }
                                }
                            });
                        }
                        tracing::info!(platform = %platform_name, "Live platform router stopped");
                    });
                }
            }
            Err(e) => {
                tracing::error!(platform = %platform, error = %e, "Failed to connect platform adapter");
            }
        }
    } else if !is_enabled {
        let mut st = state.write().await;
        match st.platform_registry.disconnect_by_name(&platform).await {
            Ok(()) => {
                tracing::info!(platform = %platform, "Platform adapter disconnected at runtime");
            }
            Err(e) => {
                tracing::warn!(platform = %platform, error = %e, "Failed to disconnect platform adapter");
            }
        }
    }

    StatusCode::OK
}

fn has_credentials(body: &serde_json::Value) -> bool {
    body.as_object()
        .map(|o| o.iter().any(|(k, v)| {
            (k.contains("token") || k.contains("secret") || k.contains("webhook"))
                && !v.as_str().unwrap_or("").is_empty()
        }))
        .unwrap_or(false)
}

// ── Factory Reset ──────────────────────────────────────────────────

pub async fn factory_reset(State(state): State<SharedState>) -> StatusCode {
    let data_dir = {
        let mut st = state.write().await;
        let data_dir = st.config.general.data_dir.clone();
        tracing::warn!(data_dir = %data_dir.display(), "Factory reset initiated");
        st.memory_mgr.clear(&data_dir);
        st.session_mgr.clear_all();
        // Clear all per-user isolated contexts
        st.user_contexts.clear();
        tracing::info!("Cleared {} per-user contexts", st.user_contexts.len());
        data_dir
    };

    let errors = wipe_user_data(&data_dir);

    if errors == 0 {
        tracing::info!("Factory reset complete — all user data wiped");
        StatusCode::OK
    } else {
        tracing::warn!(errors = errors, "Factory reset completed with errors");
        StatusCode::INTERNAL_SERVER_ERROR
    }
}

fn wipe_user_data(data_dir: &std::path::Path) -> usize {
    let mut errors = 0;

    for dir in &["sessions", "learning", "logs", "training", "users"] {
        let path = data_dir.join(dir);
        if path.exists() {
            if let Err(e) = std::fs::remove_dir_all(&path) {
                tracing::error!(path = %path.display(), error = %e, "Factory reset: failed to remove dir");
                errors += 1;
            } else {
                let _ = std::fs::create_dir_all(&path);
                tracing::info!(path = %path.display(), "Factory reset: wiped directory");
            }
        }
    }

    for file in &["platforms.json", "autonomy_log.json"] {
        let path = data_dir.join(file);
        if path.exists() {
            if let Err(e) = std::fs::remove_file(&path) {
                tracing::error!(path = %path.display(), error = %e, "Factory reset: failed to remove file");
                errors += 1;
            } else {
                tracing::info!(path = %path.display(), "Factory reset: wiped file");
            }
        }
    }

    errors
}
