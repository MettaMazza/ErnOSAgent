// Ern-OS — Discord event handler
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Serenity event handler — translates Discord gateway events into PlatformMessages.

use crate::config::DiscordConfig;
use crate::platform::adapter::PlatformMessage;
use serenity::all::{Context, EventHandler, Message, Ready};
use async_trait::async_trait;
use tokio::sync::mpsc;

/// Serenity event handler that forwards messages to the adapter's channel.
pub struct Handler {
    tx: mpsc::Sender<PlatformMessage>,
    config: DiscordConfig,
}

impl Handler {
    /// Create a new handler with the given message sender and config.
    pub fn new(tx: mpsc::Sender<PlatformMessage>, config: DiscordConfig) -> Self {
        Self { tx, config }
    }

    /// Check if a channel is in the listen list (or list is empty = all channels).
    fn is_allowed_channel(&self, channel_id: u64) -> bool {
        if self.config.listen_channels.is_empty() {
            return true;
        }
        let id_str = channel_id.to_string();
        self.config.listen_channels.iter().any(|c| c == &id_str)
    }

    /// Check if a user ID is in the admin list.
    fn is_admin(&self, user_id: u64) -> bool {
        let id_str = user_id.to_string();
        self.config.admin_ids.iter().any(|a| a == &id_str)
    }
}

#[async_trait]
impl EventHandler for Handler {
    async fn message(&self, ctx: Context, msg: Message) {
        // Ignore messages from bots (including self)
        if msg.author.bot {
            return;
        }

        // Channel filtering
        if !self.is_allowed_channel(msg.channel_id.get()) {
            return;
        }

        tracing::debug!(
            user = %msg.author.name,
            channel = %msg.channel_id,
            content_len = msg.content.len(),
            "Discord message received"
        );

        // Send typing indicator — fire and forget
        let _ = msg.channel_id.broadcast_typing(&ctx.http).await;

        let platform_msg = PlatformMessage {
            platform: "discord".to_string(),
            channel_id: msg.channel_id.get().to_string(),
            user_id: msg.author.id.get().to_string(),
            user_name: msg.author.name.clone(),
            content: msg.content.clone(),
            attachments: msg.attachments.iter()
                .map(|a| a.url.clone())
                .collect(),
            message_id: msg.id.get().to_string(),
            is_admin: self.is_admin(msg.author.id.get()),
        };

        if let Err(e) = self.tx.send(platform_msg).await {
            tracing::error!(error = %e, "Failed to forward Discord message to adapter");
        }
    }

    async fn ready(&self, _ctx: Context, ready: Ready) {
        tracing::info!(
            user = %ready.user.name,
            guilds = ready.guilds.len(),
            "Discord bot ready"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_handler() -> Handler {
        let (tx, _rx) = mpsc::channel(16);
        let config = DiscordConfig {
            token: None,
            admin_ids: vec!["123456".to_string(), "789012".to_string()],
            listen_channels: vec!["channel_1".to_string()],
            enabled: true,
        };
        Handler::new(tx, config)
    }

    #[test]
    fn test_is_admin_true() {
        let handler = test_handler();
        assert!(handler.is_admin(123456));
    }

    #[test]
    fn test_is_admin_false() {
        let handler = test_handler();
        assert!(!handler.is_admin(999));
    }

    #[test]
    fn test_allowed_channel() {
        let handler = test_handler();
        // "channel_1" is a string, not a u64 — but the check compares string representations
        // In practice, Discord channel IDs are u64. This tests the filtering logic.
        assert!(!handler.is_allowed_channel(999));
    }

    #[test]
    fn test_empty_channels_allows_all() {
        let (tx, _rx) = mpsc::channel(16);
        let config = DiscordConfig {
            token: None,
            admin_ids: vec![],
            listen_channels: vec![], // empty = all channels allowed
            enabled: true,
        };
        let handler = Handler::new(tx, config);
        assert!(handler.is_allowed_channel(12345));
        assert!(handler.is_allowed_channel(99999));
    }
}
