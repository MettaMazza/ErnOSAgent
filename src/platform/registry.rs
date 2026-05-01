// Ern-OS — Platform registry (ported from ErnOSAgent)
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Platform registry — manages active platform connections.

use crate::platform::adapter::{PlatformAdapter, PlatformStatus, MessageComponent};

pub struct PlatformRegistry {
    adapters: Vec<Box<dyn PlatformAdapter>>,
}

impl PlatformRegistry {
    pub fn new() -> Self {
        Self { adapters: Vec::new() }
    }

    pub fn register(&mut self, adapter: Box<dyn PlatformAdapter>) {
        tracing::info!(platform = adapter.name(), "Platform adapter registered");
        self.adapters.push(adapter);
    }

    pub fn statuses(&self) -> Vec<PlatformStatus> {
        self.adapters.iter().map(|a| a.status()).collect()
    }

    pub fn status_summary(&self) -> String {
        let statuses = self.statuses();
        if statuses.is_empty() {
            return "No platforms configured".to_string();
        }
        statuses.iter()
            .map(|s| {
                let icon = if s.connected { "🟢" } else { "🔴" };
                format!("{} {}", icon, s.name)
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// Connect all configured adapters.
    pub async fn connect_all(&mut self) {
        for adapter in &mut self.adapters {
            if adapter.is_configured() {
                if let Err(e) = adapter.connect().await {
                    tracing::warn!(
                        platform = adapter.name(), error = %e,
                        "Failed to connect platform adapter"
                    );
                }
            }
        }
    }

    /// Disconnect all adapters.
    pub async fn disconnect_all(&mut self) {
        for adapter in &mut self.adapters {
            if let Err(e) = adapter.disconnect().await {
                tracing::warn!(
                    platform = adapter.name(), error = %e,
                    "Failed to disconnect platform adapter"
                );
            }
        }
    }

    /// Connect a specific adapter by name.
    pub async fn connect_by_name(&mut self, name: &str) -> anyhow::Result<()> {
        for adapter in &mut self.adapters {
            if adapter.name().eq_ignore_ascii_case(name) {
                return adapter.connect().await;
            }
        }
        anyhow::bail!("No adapter registered with name '{}'", name)
    }

    /// Disconnect a specific adapter by name.
    pub async fn disconnect_by_name(&mut self, name: &str) -> anyhow::Result<()> {
        for adapter in &mut self.adapters {
            if adapter.name().eq_ignore_ascii_case(name) {
                return adapter.disconnect().await;
            }
        }
        anyhow::bail!("No adapter registered with name '{}'", name)
    }

    /// Get mutable access to adapters.
    pub fn adapters_mut(&mut self) -> &mut Vec<Box<dyn PlatformAdapter>> {
        &mut self.adapters
    }

    /// List names of all currently connected adapters.
    pub fn list_connected(&self) -> Vec<String> {
        self.adapters.iter()
            .filter(|a| a.status().connected)
            .map(|a| a.name().to_string())
            .collect()
    }

    /// Deliver a response back to a platform's channel.
    /// Looks up the adapter by name and calls reply_to_message.
    pub async fn send_to_platform(
        &self, platform: &str, channel_id: &str, message_id: &str, content: &str,
    ) -> anyhow::Result<()> {
        for adapter in &self.adapters {
            if adapter.name().eq_ignore_ascii_case(platform) {
                return adapter.reply_to_message(channel_id, message_id, content).await;
            }
        }
        anyhow::bail!("No adapter registered for platform '{}'", platform)
    }

    /// Send a proactive message to a platform channel (no reply reference).
    /// Used for post-recompile resume delivery and other server-initiated messages.
    pub async fn send_message(
        &self, platform: &str, channel_id: &str, content: &str,
    ) -> anyhow::Result<()> {
        for adapter in &self.adapters {
            if adapter.name().eq_ignore_ascii_case(platform) {
                return adapter.send_message(channel_id, content).await;
            }
        }
        anyhow::bail!("No adapter registered for platform '{}'", platform)
    }

    /// Send typing indicator to a platform channel.
    pub async fn start_typing(&self, platform: &str, channel_id: &str) -> anyhow::Result<()> {
        for adapter in &self.adapters {
            if adapter.name().eq_ignore_ascii_case(platform) {
                return adapter.start_typing(channel_id).await;
            }
        }
        Ok(())
    }

    /// Create a thinking thread on the platform.
    pub async fn create_thinking_thread(
        &self, platform: &str, channel_id: &str, message_id: &str, title: &str,
    ) -> anyhow::Result<String> {
        for adapter in &self.adapters {
            if adapter.name().eq_ignore_ascii_case(platform) {
                return adapter.create_thinking_thread(channel_id, message_id, title).await;
            }
        }
        anyhow::bail!("No adapter for '{}'", platform)
    }

    /// Send content to a thinking thread.
    pub async fn send_to_thread(
        &self, platform: &str, thread_id: &str, content: &str,
    ) -> anyhow::Result<()> {
        for adapter in &self.adapters {
            if adapter.name().eq_ignore_ascii_case(platform) {
                return adapter.send_to_thread(thread_id, content).await;
            }
        }
        Ok(())
    }

    /// Archive a thinking thread (preserves for audit trail).
    pub async fn archive_thread(
        &self, platform: &str, thread_id: &str,
    ) -> anyhow::Result<()> {
        for adapter in &self.adapters {
            if adapter.name().eq_ignore_ascii_case(platform) {
                return adapter.archive_thread(thread_id).await;
            }
        }
        Ok(())
    }

    /// Reply with interactive button components.
    pub async fn reply_with_components(
        &self, platform: &str, channel_id: &str, message_id: &str,
        content: &str, components: &[MessageComponent],
    ) -> anyhow::Result<String> {
        for adapter in &self.adapters {
            if adapter.name().eq_ignore_ascii_case(platform) {
                return adapter.reply_with_components(
                    channel_id, message_id, content, components,
                ).await;
            }
        }
        anyhow::bail!("No adapter for '{}'", platform)
    }

    /// Delete a message on a platform.
    pub async fn delete_message(
        &self, platform: &str, channel_id: &str, message_id: &str,
    ) -> anyhow::Result<()> {
        for adapter in &self.adapters {
            if adapter.name().eq_ignore_ascii_case(platform) {
                return adapter.delete_message(channel_id, message_id).await;
            }
        }
        Ok(())
    }

    /// Send an audio file attachment to a platform channel.
    pub async fn send_audio_file(
        &self, platform: &str, channel_id: &str, audio_bytes: Vec<u8>, filename: &str,
    ) -> anyhow::Result<()> {
        for adapter in &self.adapters {
            if adapter.name().eq_ignore_ascii_case(platform) {
                return adapter.send_audio_file(channel_id, audio_bytes, filename).await;
            }
        }
        Ok(())
    }

    /// Send an image file attachment to a platform channel.
    pub async fn send_image_file(
        &self, platform: &str, channel_id: &str, message_id: &str,
        image_bytes: Vec<u8>, filename: &str, caption: &str,
    ) -> anyhow::Result<()> {
        for adapter in &self.adapters {
            if adapter.name().eq_ignore_ascii_case(platform) {
                return adapter.send_image_file(
                    channel_id, message_id, image_bytes, filename, caption,
                ).await;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_registry() {
        let registry = PlatformRegistry::new();
        assert!(registry.statuses().is_empty());
        assert!(registry.status_summary().contains("No platforms"));
    }
}
