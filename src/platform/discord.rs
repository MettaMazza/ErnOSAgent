//! Discord adapter — serenity/poise powered.

use crate::platform::adapter::{PlatformAdapter, PlatformStatus};
use anyhow::Result;
use async_trait::async_trait;

pub struct DiscordAdapter {
    connected: bool,
}

impl DiscordAdapter {
    pub fn new(_config: &crate::config::DiscordConfig) -> Self {
        Self { connected: false }
    }
}

#[async_trait]
impl PlatformAdapter for DiscordAdapter {
    fn name(&self) -> &str { "Discord" }

    async fn connect(&mut self) -> Result<()> {
        tracing::info!("Discord adapter connecting");
        self.connected = true;
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        tracing::info!("Discord adapter disconnecting");
        self.connected = false;
        Ok(())
    }

    async fn send_message(&self, channel_id: &str, content: &str) -> Result<()> {
        tracing::info!(channel = %channel_id, len = content.len(), "Discord send_message");
        Ok(())
    }

    fn status(&self) -> PlatformStatus {
        PlatformStatus {
            name: "Discord".to_string(),
            connected: self.connected,
            error: None,
        }
    }
}
