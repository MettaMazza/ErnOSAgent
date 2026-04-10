// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Platform Message Router — bridges platform adapters to the inference pipeline.
//!
//! Subscribes to all active adapter message channels and routes each incoming
//! message through non-streaming inference, sending the response back.

use crate::platform::adapter::{PlatformAdapter, PlatformMessage};
use crate::web::state::SharedState;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};

/// Run the platform message router.
///
/// Receives messages from all connected platform adapters and processes each
/// through inference, sending the response back via the adapter.
pub async fn run(
    adapters: Arc<RwLock<Vec<Box<dyn PlatformAdapter>>>>,
    state: SharedState,
) {
    // Collect all message receivers
    let mut receivers: Vec<mpsc::Receiver<PlatformMessage>> = Vec::new();
    {
        let mut adapters = adapters.write().await;
        for adapter in adapters.iter_mut() {
            if let Some(rx) = adapter.take_message_receiver() {
                receivers.push(rx);
            }
        }
    }

    if receivers.is_empty() {
        tracing::debug!("Platform router: no adapters with active receivers");
        return;
    }

    // Merge all receivers into a single channel
    let (merged_tx, mut merged_rx) = mpsc::channel::<PlatformMessage>(256);
    for mut rx in receivers {
        let tx = merged_tx.clone();
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                if tx.send(msg).await.is_err() { break; }
            }
        });
    }
    drop(merged_tx);

    tracing::info!("Platform message router started");

    while let Some(msg) = merged_rx.recv().await {
        let state = state.clone();
        let adapters = adapters.clone();

        tokio::spawn(async move {
            tracing::info!(
                platform = %msg.platform,
                user = %msg.user_name,
                channel = %msg.channel_id,
                content_len = msg.content.len(),
                "Platform message received"
            );

            let response = process_platform_message(&state, &msg).await;

            match response {
                Ok(reply) => {
                    let adapters = adapters.read().await;
                    for adapter in adapters.iter() {
                        if adapter.name().to_lowercase() == msg.platform {
                            if let Err(e) = adapter.send_message(&msg.channel_id, &reply).await {
                                tracing::error!(
                                    platform = %msg.platform,
                                    error = %e,
                                    "Failed to send platform response"
                                );
                            }
                            break;
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        platform = %msg.platform,
                        error = %e,
                        "Failed to process platform message"
                    );
                }
            }
        });
    }

    tracing::info!("Platform message router stopped");
}

/// Process a single platform message through non-streaming inference.
async fn process_platform_message(
    state: &SharedState,
    msg: &PlatformMessage,
) -> anyhow::Result<String> {
    let (provider, model_name, system_prompt, identity_prompt) = {
        let st = state.read().await;
        (
            st.provider.clone(),
            st.config.general.active_model.clone(),
            st.core_prompt.clone(),
            st.identity_prompt.clone(),
        )
    };

    // Build the message sequence: system + identity + user message
    let mut messages = Vec::with_capacity(4);
    if !system_prompt.is_empty() {
        messages.push(crate::provider::Message {
            role: "system".to_string(),
            content: system_prompt,
            images: Vec::new(),
        });
    }
    if !identity_prompt.is_empty() {
        messages.push(crate::provider::Message {
            role: "system".to_string(),
            content: identity_prompt,
            images: Vec::new(),
        });
    }

    // Add the user message with platform context
    messages.push(crate::provider::Message {
        role: "user".to_string(),
        content: format!("[via {} from {}] {}", msg.platform, msg.user_name, msg.content),
        images: Vec::new(),
    });

    // Run non-streaming inference (same as Observer uses)
    let reply = provider.chat_sync(&model_name, &messages, None).await?;
    let reply = reply.trim().to_string();

    // Record in the active session for continuity
    {
        let mut st = state.write().await;
        let session = st.session_mgr.active_mut();
        session.messages.push(crate::provider::Message {
            role: "user".to_string(),
            content: format!("[{}:{}] {}", msg.platform, msg.user_name, msg.content),
            images: Vec::new(),
        });
        session.messages.push(crate::provider::Message {
            role: "assistant".to_string(),
            content: reply.clone(),
            images: Vec::new(),
        });
    }

    Ok(reply)
}
