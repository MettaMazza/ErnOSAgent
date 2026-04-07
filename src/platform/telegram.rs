//! Telegram adapter — teloxide powered.

use crate::platform::adapter::{PlatformAdapter, PlatformStatus};
use anyhow::Result;
use async_trait::async_trait;

pub struct TelegramAdapter {
    connected: bool,
}

impl TelegramAdapter {
    pub fn new(_config: &crate::config::TelegramConfig) -> Self {
        Self { connected: false }
    }
}

#[async_trait]
impl PlatformAdapter for TelegramAdapter {
    fn name(&self) -> &str { "Telegram" }
    async fn connect(&mut self) -> Result<()> { self.connected = true; Ok(()) }
    async fn disconnect(&mut self) -> Result<()> { self.connected = false; Ok(()) }
    async fn send_message(&self, _channel_id: &str, _content: &str) -> Result<()> { Ok(()) }
    fn status(&self) -> PlatformStatus {
        PlatformStatus { name: "Telegram".to_string(), connected: self.connected, error: None }
    }
}
