//! Platform adapter trait — unified interface for chat platforms.

use anyhow::Result;
use async_trait::async_trait;

/// Events from external platforms.
#[derive(Debug, Clone)]
pub struct PlatformMessage {
    pub platform: String,
    pub channel_id: String,
    pub user_id: String,
    pub user_name: String,
    pub content: String,
    pub attachments: Vec<String>,
}

/// Status of a platform connection.
#[derive(Debug, Clone)]
pub struct PlatformStatus {
    pub name: String,
    pub connected: bool,
    pub error: Option<String>,
}

#[async_trait]
pub trait PlatformAdapter: Send + Sync {
    fn name(&self) -> &str;
    async fn connect(&mut self) -> Result<()>;
    async fn disconnect(&mut self) -> Result<()>;
    async fn send_message(&self, channel_id: &str, content: &str) -> Result<()>;
    fn status(&self) -> PlatformStatus;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_message_fields() {
        let msg = PlatformMessage {
            platform: "discord".to_string(),
            channel_id: "123".to_string(),
            user_id: "456".to_string(),
            user_name: "user".to_string(),
            content: "hello".to_string(),
            attachments: Vec::new(),
        };
        assert_eq!(msg.platform, "discord");
    }
}
