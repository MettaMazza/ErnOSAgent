//! Platform handler — REST API for managing platform adapters.

use crate::web::state::AppState;
use axum::{extract::State, Json};

/// GET /api/platforms — list all platform adapters and their status.
pub async fn list_platforms(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let reg = state.platforms.read().await;
    let statuses: Vec<serde_json::Value> = reg.statuses().iter().map(|s| {
        serde_json::json!({
            "name": s.name,
            "connected": s.connected,
            "error": s.error,
        })
    }).collect();

    Json(serde_json::json!({
        "platforms": statuses,
        "summary": reg.status_summary(),
    }))
}

/// POST /api/platforms/:name/connect — connect a specific platform.
pub async fn connect_platform(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    let mut reg = state.platforms.write().await;
    match reg.connect_by_name(&name).await {
        Ok(_) => Json(serde_json::json!({
            "success": true,
            "message": format!("{} connected", name),
        })),
        Err(e) => Json(serde_json::json!({
            "success": false,
            "error": e.to_string(),
        })),
    }
}

/// POST /api/platforms/:name/disconnect — disconnect a specific platform.
pub async fn disconnect_platform(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    let mut reg = state.platforms.write().await;
    match reg.disconnect_by_name(&name).await {
        Ok(_) => Json(serde_json::json!({
            "success": true,
            "message": format!("{} disconnected", name),
        })),
        Err(e) => Json(serde_json::json!({
            "success": false,
            "error": e.to_string(),
        })),
    }
}

/// GET /api/platforms/config — get platform config for the settings UI.
pub async fn get_platform_config(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let config = state.mutable_config.read().await;
    Json(serde_json::json!({
        "discord": {
            "enabled": config.discord.enabled,
            "token": config.discord.token.as_deref().map(mask_token),
            "has_token": config.discord.resolve_token().is_some(),
            "admin_ids": config.discord.admin_ids,
            "listen_channels": config.discord.listen_channels,
        },
        "telegram": {
            "enabled": config.telegram.enabled,
            "token": config.telegram.token.as_deref().map(mask_token),
            "has_token": config.telegram.resolve_token().is_some(),
            "admin_ids": config.telegram.admin_ids,
            "allowed_chats": config.telegram.allowed_chats,
        },
    }))
}

/// PUT /api/platforms/config — update platform config from the settings UI.
pub async fn update_platform_config(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let mut config = state.mutable_config.write().await;

    // Update Discord config
    if let Some(discord) = body.get("discord") {
        if let Some(enabled) = discord.get("enabled").and_then(|v| v.as_bool()) {
            config.discord.enabled = enabled;
        }
        if let Some(token) = discord.get("token").and_then(|v| v.as_str()) {
            if !token.is_empty() && !token.contains("****") {
                config.discord.token = Some(token.to_string());
            }
        }
        if let Some(ids) = discord.get("admin_ids").and_then(|v| v.as_array()) {
            config.discord.admin_ids = ids.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }
        if let Some(channels) = discord.get("listen_channels").and_then(|v| v.as_array()) {
            config.discord.listen_channels = channels.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }
    }

    // Update Telegram config
    if let Some(telegram) = body.get("telegram") {
        if let Some(enabled) = telegram.get("enabled").and_then(|v| v.as_bool()) {
            config.telegram.enabled = enabled;
        }
        if let Some(token) = telegram.get("token").and_then(|v| v.as_str()) {
            if !token.is_empty() && !token.contains("****") {
                config.telegram.token = Some(token.to_string());
            }
        }
        if let Some(ids) = telegram.get("admin_ids").and_then(|v| v.as_array()) {
            config.telegram.admin_ids = ids.iter()
                .filter_map(|v| v.as_i64())
                .collect();
        }
        if let Some(chats) = telegram.get("allowed_chats").and_then(|v| v.as_array()) {
            config.telegram.allowed_chats = chats.iter()
                .filter_map(|v| v.as_i64())
                .collect();
        }
    }

    // Persist to ern-os.toml
    if let Ok(serialized) = toml::to_string_pretty(&*config) {
        if let Err(e) = std::fs::write("ern-os.toml", serialized) {
            tracing::error!(error = %e, "Failed to persist platform config");
            return Json(serde_json::json!({
                "success": false,
                "error": format!("Failed to write config: {}", e),
            }));
        }
    }

    Json(serde_json::json!({
        "success": true,
        "message": "Platform config updated",
    }))
}

/// POST /api/chat/platform — ingest a message from a platform adapter.
/// The platform router calls this to forward messages through the inference pipeline.
pub async fn platform_ingest(
    State(state): State<AppState>,
    Json(msg): Json<crate::platform::adapter::PlatformMessage>,
) -> Json<serde_json::Value> {
    tracing::info!(
        platform = %msg.platform,
        user = %msg.user_name,
        content_len = msg.content.len(),
        "Platform message ingested"
    );

    // Per-user session isolation
    let session_id = format!("{}_{}", msg.platform, msg.user_id);

    // Build message history for the inference call
    let messages = vec![
        crate::provider::Message::text("user", &msg.content),
    ];

    let thinking = state.config.prompt.thinking_enabled;

    // L1 fast path
    match crate::inference::fast_reply::run(
        state.provider.as_ref(),
        &messages,
        thinking,
    ).await {
        Ok((_initial, rx)) => {
            match crate::inference::fast_reply::consume_stream(rx, None).await {
                Ok(result) => {
                    let response = match result {
                        crate::inference::fast_reply::FastReplyResult::Reply { text, .. } => text,
                        crate::inference::fast_reply::FastReplyResult::Escalate { objective, .. } => {
                            tracing::info!(
                                platform = %msg.platform,
                                objective = %objective,
                                "Platform message escalated to L2"
                            );
                            format!("Working on: {}", objective)
                        }
                        crate::inference::fast_reply::FastReplyResult::ToolCall { name, .. } => {
                            tracing::info!(
                                platform = %msg.platform,
                                tool = %name,
                                "Platform message triggered tool call"
                            );
                            format!("Using tool: {}", name)
                        }
                    };

                    Json(serde_json::json!({
                        "success": true,
                        "response": response,
                        "session_id": session_id,
                        "platform": msg.platform,
                        "channel_id": msg.channel_id,
                        "message_id": msg.message_id,
                    }))
                }
                Err(e) => {
                    tracing::error!(error = %e, platform = %msg.platform, "Stream consumption failed");
                    Json(serde_json::json!({
                        "success": false,
                        "error": e.to_string(),
                    }))
                }
            }
        }
        Err(e) => {
            tracing::error!(error = %e, platform = %msg.platform, "Platform inference failed");
            Json(serde_json::json!({
                "success": false,
                "error": e.to_string(),
            }))
        }
    }
}

/// Mask a token for display — show first 8 chars, mask the rest.
fn mask_token(token: &str) -> String {
    if token.len() <= 8 {
        return "****".to_string();
    }
    format!("{}****", &token[..8])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_token_long() {
        let masked = mask_token("abcdefghijklmnop");
        assert_eq!(masked, "abcdefgh****");
    }

    #[test]
    fn test_mask_token_short() {
        let masked = mask_token("abc");
        assert_eq!(masked, "****");
    }
}
