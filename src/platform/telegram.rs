// Ern-OS — Telegram platform adapter
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Concrete `PlatformAdapter` implementation for Telegram via teloxide.
//! Per governance §6.3: standalone client that speaks the WebUI API.

use crate::config::TelegramConfig;
use crate::platform::adapter::{PlatformAdapter, PlatformMessage, PlatformStatus};
use anyhow::{Context, Result};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::mpsc;
use teloxide::prelude::*;
use teloxide::types::ParseMode;

/// Telegram adapter — bridges Telegram Bot API events to the Ern-OS hub.
pub struct TelegramAdapter {
    config: TelegramConfig,
    bot: Option<Bot>,
    tx: mpsc::Sender<PlatformMessage>,
    rx: Option<mpsc::Receiver<PlatformMessage>>,
    connected: Arc<AtomicBool>,
    shutdown: Option<tokio::sync::oneshot::Sender<()>>,
}

impl TelegramAdapter {
    /// Create a new Telegram adapter from config.
    pub fn new(config: TelegramConfig) -> Self {
        let (tx, rx) = mpsc::channel(64);
        Self {
            config,
            bot: None,
            tx,
            rx: Some(rx),
            connected: Arc::new(AtomicBool::new(false)),
            shutdown: None,
        }
    }

    /// Check if a chat ID is in the allowed list (empty = all chats).
    fn is_allowed_chat(&self, chat_id: i64) -> bool {
        if self.config.allowed_chats.is_empty() {
            return true;
        }
        self.config.allowed_chats.contains(&chat_id)
    }

    /// Check if a user ID is in the admin list.
    fn is_admin(&self, user_id: i64) -> bool {
        self.config.admin_ids.contains(&user_id)
    }
}

#[async_trait]
impl PlatformAdapter for TelegramAdapter {
    fn name(&self) -> &str {
        "Telegram"
    }

    fn is_configured(&self) -> bool {
        self.config.is_configured()
    }

    async fn connect(&mut self) -> Result<()> {
        let token = self.config.resolve_token()
            .context("Telegram token not configured — set in ern-os.toml or TELEGRAM_TOKEN env")?;

        let bot = Bot::new(&token);
        self.bot = Some(bot.clone());

        let connected = self.connected.clone();
        let tx = self.tx.clone();
        let config = self.config.clone();
        let (shutdown_tx, mut shutdown_rx) = tokio::sync::oneshot::channel();
        self.shutdown = Some(shutdown_tx);

        tokio::spawn(async move {
            connected.store(true, Ordering::SeqCst);
            tracing::info!("Telegram adapter connected");

            let handler = Update::filter_message().endpoint(
                move |bot: Bot, msg: teloxide::types::Message| {
                    let tx = tx.clone();
                    let config = config.clone();
                    async move {
                        handle_telegram_message(&tx, &config, &msg).await;
                        respond(())
                    }
                }
            );

            let mut dispatcher = Dispatcher::builder(bot, handler)
                .enable_ctrlc_handler()
                .build();

            tokio::select! {
                _ = dispatcher.dispatch() => {}
                _ = &mut shutdown_rx => {
                    tracing::info!("Telegram adapter shutting down");
                    dispatcher.shutdown().await;
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
        tracing::info!("Telegram adapter disconnected");
        Ok(())
    }

    async fn send_message(&self, channel_id: &str, content: &str) -> Result<()> {
        let bot = self.bot.as_ref()
            .context("Telegram not connected")?;
        let chat_id = channel_id.parse::<i64>()
            .context("Invalid Telegram chat ID")?;

        let chunks = chunk_telegram(content, 4096);
        for chunk in chunks {
            bot.send_message(ChatId(chat_id), &chunk)
                .parse_mode(ParseMode::MarkdownV2)
                .await
                .context("Failed to send Telegram message")?;
        }
        Ok(())
    }

    async fn reply_to_message(
        &self, channel_id: &str, message_id: &str, content: &str,
    ) -> Result<()> {
        let bot = self.bot.as_ref()
            .context("Telegram not connected")?;
        let chat_id = channel_id.parse::<i64>()
            .context("Invalid Telegram chat ID")?;
        let msg_id = message_id.parse::<i32>()
            .context("Invalid Telegram message ID")?;

        let chunks = chunk_telegram(content, 4096);
        for (i, chunk) in chunks.iter().enumerate() {
            let mut req = bot.send_message(ChatId(chat_id), chunk);
            if i == 0 {
                req = req.reply_to_message_id(teloxide::types::MessageId(msg_id));
            }
            req.await.context("Failed to send Telegram reply")?;
        }
        Ok(())
    }

    fn take_message_receiver(&mut self) -> Option<mpsc::Receiver<PlatformMessage>> {
        self.rx.take()
    }

    fn status(&self) -> PlatformStatus {
        PlatformStatus {
            name: "Telegram".to_string(),
            connected: self.connected.load(Ordering::SeqCst),
            error: None,
        }
    }
}

/// Handle a single Telegram message — convert to PlatformMessage and forward.
async fn handle_telegram_message(
    tx: &mpsc::Sender<PlatformMessage>,
    config: &TelegramConfig,
    msg: &teloxide::types::Message,
) {
    // Only process text messages
    let text = match msg.text() {
        Some(t) => t.to_string(),
        None => return,
    };

    let chat_id = msg.chat.id.0;
    let user = match &msg.from {
        Some(u) => u,
        None => return,
    };

    // Skip bots
    if user.is_bot {
        return;
    }

    // Chat filtering
    let allowed = if config.allowed_chats.is_empty() {
        true
    } else {
        config.allowed_chats.contains(&chat_id)
    };
    if !allowed {
        return;
    }

    let is_admin = config.admin_ids.contains(&(user.id.0 as i64));

    let platform_msg = PlatformMessage {
        platform: "telegram".to_string(),
        channel_id: chat_id.to_string(),
        user_id: user.id.0.to_string(),
        user_name: user.first_name.clone(),
        content: text,
        attachments: Vec::new(),
        message_id: msg.id.0.to_string(),
        is_admin,
    };

    if let Err(e) = tx.send(platform_msg).await {
        tracing::error!(error = %e, "Failed to forward Telegram message to adapter");
    }
}

/// Split a message into chunks that respect Telegram's 4096 character limit.
fn chunk_telegram(content: &str, max_len: usize) -> Vec<String> {
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
    fn test_adapter_name() {
        let config = TelegramConfig::default();
        let adapter = TelegramAdapter::new(config);
        assert_eq!(adapter.name(), "Telegram");
    }

    #[test]
    fn test_not_configured_without_token() {
        let config = TelegramConfig::default();
        let adapter = TelegramAdapter::new(config);
        assert!(!adapter.is_configured());
    }

    #[test]
    fn test_chunk_short() {
        let chunks = chunk_telegram("hello", 4096);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_chunk_over_limit() {
        let msg = "a".repeat(8000);
        let chunks = chunk_telegram(&msg, 4096);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().all(|c| c.len() <= 4096));
    }

    #[test]
    fn test_is_admin() {
        let config = TelegramConfig {
            admin_ids: vec![12345, 67890],
            ..TelegramConfig::default()
        };
        let adapter = TelegramAdapter::new(config);
        assert!(adapter.is_admin(12345));
        assert!(!adapter.is_admin(99999));
    }

    #[test]
    fn test_is_allowed_chat_empty_allows_all() {
        let config = TelegramConfig::default();
        let adapter = TelegramAdapter::new(config);
        assert!(adapter.is_allowed_chat(12345));
    }

    #[test]
    fn test_is_allowed_chat_filtered() {
        let config = TelegramConfig {
            allowed_chats: vec![111, 222],
            ..TelegramConfig::default()
        };
        let adapter = TelegramAdapter::new(config);
        assert!(adapter.is_allowed_chat(111));
        assert!(!adapter.is_allowed_chat(999));
    }
}
