//! WhatsApp adapter — webhook-based via Meta Cloud API.

use crate::platform::adapter::{PlatformAdapter, PlatformStatus};
use anyhow::Result;
use async_trait::async_trait;

pub struct WhatsAppAdapter {
    connected: bool,
}

impl WhatsAppAdapter {
    pub fn new(_config: &crate::config::WhatsAppConfig) -> Self {
        Self { connected: false }
    }
}

#[async_trait]
impl PlatformAdapter for WhatsAppAdapter {
    fn name(&self) -> &str { "WhatsApp" }
    async fn connect(&mut self) -> Result<()> { self.connected = true; Ok(()) }
    async fn disconnect(&mut self) -> Result<()> { self.connected = false; Ok(()) }
    async fn send_message(&self, _channel_id: &str, _content: &str) -> Result<()> { Ok(()) }
    fn status(&self) -> PlatformStatus {
        PlatformStatus { name: "WhatsApp".to_string(), connected: self.connected, error: None }
    }
}
