// Ern-OS — Platform router (ported from ErnOSAgent)
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Platform router — routes incoming platform messages to the Ern-OS
//! WebSocket chat API as a client, per governance §6.3.

use crate::platform::adapter::PlatformMessage;
use crate::platform::registry::PlatformRegistry;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Start routing messages from all connected platform adapters
/// into the Ern-OS chat pipeline.
pub async fn start_platform_router(
    registry: Arc<RwLock<PlatformRegistry>>,
    hub_port: u16,
) {
    let mut reg = registry.write().await;
    let adapters = reg.adapters_mut();

    for adapter in adapters.iter_mut() {
        if let Some(rx) = adapter.take_message_receiver() {
            let name = adapter.name().to_string();
            let port = hub_port;
            tokio::spawn(async move {
                route_platform_messages(name, rx, port).await;
            });
        }
    }
}

/// Route messages from a single platform adapter to the WebUI hub.
async fn route_platform_messages(
    platform: String,
    mut rx: tokio::sync::mpsc::Receiver<PlatformMessage>,
    hub_port: u16,
) {
    tracing::info!(platform = %platform, "Platform router started");

    while let Some(msg) = rx.recv().await {
        tracing::debug!(
            platform = %msg.platform,
            user = %msg.user_name,
            content_len = msg.content.len(),
            "Routing platform message to hub"
        );

        if let Err(e) = forward_to_hub(&msg, hub_port).await {
            tracing::warn!(
                platform = %msg.platform,
                error = %e,
                "Failed to forward platform message"
            );
        }
    }

    tracing::info!(platform = %platform, "Platform router stopped");
}

/// Forward a platform message to the Ern-OS hub via HTTP API.
async fn forward_to_hub(msg: &PlatformMessage, port: u16) -> anyhow::Result<()> {
    let url = format!("http://127.0.0.1:{}/api/chat/platform", port);
    let client = reqwest::Client::new();

    let payload = serde_json::json!({
        "platform": msg.platform,
        "channel_id": msg.channel_id,
        "user_id": msg.user_id,
        "user_name": msg.user_name,
        "content": msg.content,
        "message_id": msg.message_id,
        "is_admin": msg.is_admin,
    });

    let resp = client.post(&url)
        .json(&payload)
        .send()
        .await?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Hub rejected message: {} {}", status, body);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_payload_structure() {
        let msg = PlatformMessage {
            platform: "discord".to_string(),
            channel_id: "ch1".to_string(),
            user_id: "u1".to_string(),
            user_name: "Test".to_string(),
            content: "Hello".to_string(),
            attachments: vec![],
            message_id: "m1".to_string(),
            is_admin: true,
        };
        let payload = serde_json::json!({
            "platform": msg.platform,
            "content": msg.content,
            "is_admin": msg.is_admin,
        });
        assert_eq!(payload["platform"], "discord");
        assert_eq!(payload["is_admin"], true);
    }
}
