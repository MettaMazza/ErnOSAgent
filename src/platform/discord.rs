// Ern-OS — Discord platform adapter
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Concrete `PlatformAdapter` implementation for Discord via serenity.
//! Connects to the Discord gateway, ingests messages, and delivers responses.
//! Per governance §6.3: this is a standalone client that speaks the WebUI API.

use crate::config::DiscordConfig;
use crate::platform::adapter::{PlatformAdapter, PlatformMessage, PlatformStatus};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serenity::all::{GatewayIntents, Http};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;

/// Discord adapter — bridges Discord gateway events to the Ern-OS hub.
pub struct DiscordAdapter {
    config: DiscordConfig,
    http: Option<Arc<Http>>,
    tx: mpsc::Sender<PlatformMessage>,
    rx: Option<mpsc::Receiver<PlatformMessage>>,
    connected: Arc<AtomicBool>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl DiscordAdapter {
    /// Create a new Discord adapter from config.
    pub fn new(config: DiscordConfig) -> Self {
        let (tx, rx) = mpsc::channel(64);
        Self {
            config,
            http: None,
            tx,
            rx: Some(rx),
            connected: Arc::new(AtomicBool::new(false)),
            shutdown: None,
        }
    }
}

#[async_trait]
impl PlatformAdapter for DiscordAdapter {
    fn name(&self) -> &str {
        "Discord"
    }

    fn is_configured(&self) -> bool {
        self.config.is_configured()
    }

    async fn connect(&mut self) -> Result<()> {
        let token = self.config.resolve_token()
            .context("Discord token not configured — set in ern-os.toml or DISCORD_TOKEN env")?;

        let intents = GatewayIntents::GUILD_MESSAGES
            | GatewayIntents::DIRECT_MESSAGES
            | GatewayIntents::MESSAGE_CONTENT;

        let handler = super::discord_handler::Handler::new(
            self.tx.clone(),
            self.config.clone(),
        );

        let http = Arc::new(Http::new(&token));
        self.http = Some(http.clone());

        let connected = self.connected.clone();
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        self.shutdown = Some(shutdown_tx);

        tokio::spawn(async move {
            let mut client = match serenity::Client::builder(&token, intents)
                .event_handler(handler)
                .await
            {
                Ok(c) => c,
                Err(e) => {
                    tracing::error!(error = %e, "Failed to create Discord client");
                    return;
                }
            };

            connected.store(true, Ordering::SeqCst);
            tracing::info!("Discord adapter connected");

            tokio::select! {
                result = client.start() => {
                    if let Err(e) = result {
                        tracing::error!(error = %e, "Discord client error");
                    }
                }
                _ = &mut shutdown_rx => {
                    tracing::info!("Discord adapter shutting down");
                    client.shard_manager.shutdown_all().await;
                }
            }

            connected.store(false, Ordering::SeqCst);
        });

        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
        self.connected.store(false, Ordering::SeqCst);
        tracing::info!("Discord adapter disconnected");
        Ok(())
    }

    async fn send_message(&self, channel_id: &str, content: &str) -> Result<()> {
        let http = self.http.as_ref()
            .context("Discord not connected")?;
        let channel = channel_id.parse::<u64>()
            .context("Invalid Discord channel ID")?;

        let chunks = chunk_message(content, 2000);
        for chunk in chunks {
            serenity::all::ChannelId::new(channel)
                .say(http.as_ref(), &chunk)
                .await
                .context("Failed to send Discord message")?;
        }
        Ok(())
    }

    async fn reply_to_message(
        &self, channel_id: &str, message_id: &str, content: &str,
    ) -> Result<()> {
        let http = self.http.as_ref()
            .context("Discord not connected")?;
        let channel = channel_id.parse::<u64>()
            .context("Invalid Discord channel ID")?;
        let msg_id = message_id.parse::<u64>()
            .context("Invalid Discord message ID")?;

        let chunks = chunk_message(content, 2000);
        for (i, chunk) in chunks.iter().enumerate() {
            let ch = serenity::all::ChannelId::new(channel);
            if i == 0 {
                // First chunk: reply to the original message
                let result = ch.send_message(http.as_ref(), serenity::builder::CreateMessage::new()
                    .content(chunk)
                    .reference_message(serenity::all::MessageReference::from(
                        serenity::all::MessageId::new(msg_id),
                    ))
                ).await;

                // Fallback: if reply fails (stale reference), send as channel message
                if result.is_err() {
                    tracing::warn!("Discord reply failed, sending as channel message");
                    ch.say(http.as_ref(), chunk).await
                        .context("Failed to send Discord fallback message")?;
                }
            } else {
                ch.say(http.as_ref(), chunk).await
                    .context("Failed to send Discord chunk")?;
            }
        }
        Ok(())
    }

    fn take_message_receiver(&mut self) -> Option<mpsc::Receiver<PlatformMessage>> {
        self.rx.take()
    }

    fn status(&self) -> PlatformStatus {
        PlatformStatus {
            name: "Discord".to_string(),
            connected: self.connected.load(Ordering::SeqCst),
            error: None,
        }
    }
}

/// Split a message into chunks that respect Discord's character limit.
/// Splits at newlines when possible, otherwise at the hard limit.
fn chunk_message(content: &str, max_len: usize) -> Vec<String> {
    if content.len() <= max_len {
        return vec![content.to_string()];
    }

    let mut chunks = Vec::new();
    let mut remaining = content;

    while !remaining.is_empty() {
        if remaining.len() <= max_len {
            chunks.push(remaining.to_string());
            break;
        }

        // Find a good split point (prefer newline)
        let split_at = remaining[..max_len]
            .rfind('\n')
            .unwrap_or(max_len);

        let (chunk, rest) = remaining.split_at(split_at);
        chunks.push(chunk.to_string());
        remaining = rest.trim_start_matches('\n');
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_short_message() {
        let chunks = chunk_message("hello", 2000);
        assert_eq!(chunks, vec!["hello"]);
    }

    #[test]
    fn test_chunk_exact_limit() {
        let msg = "a".repeat(2000);
        let chunks = chunk_message(&msg, 2000);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_chunk_over_limit() {
        let msg = "a".repeat(4500);
        let chunks = chunk_message(&msg, 2000);
        assert_eq!(chunks.len(), 3);
        assert!(chunks.iter().all(|c| c.len() <= 2000));
    }

    #[test]
    fn test_chunk_prefers_newline_split() {
        let mut msg = "a".repeat(1500);
        msg.push('\n');
        msg.push_str(&"b".repeat(1500));
        let chunks = chunk_message(&msg, 2000);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].ends_with('a'));
        assert!(chunks[1].starts_with('b'));
    }

    #[test]
    fn test_adapter_name() {
        let config = DiscordConfig::default();
        let adapter = DiscordAdapter::new(config);
        assert_eq!(adapter.name(), "Discord");
    }

    #[test]
    fn test_not_configured_without_token() {
        let config = DiscordConfig::default();
        let adapter = DiscordAdapter::new(config);
        assert!(!adapter.is_configured());
    }

    #[test]
    fn test_status_disconnected() {
        let config = DiscordConfig::default();
        let adapter = DiscordAdapter::new(config);
        let status = adapter.status();
        assert!(!status.connected);
        assert_eq!(status.name, "Discord");
    }
}
