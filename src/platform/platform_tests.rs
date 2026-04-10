// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Tests for platform adapter creation and registry.

#[cfg(test)]
mod tests {
    use crate::config::AppConfig;
    use crate::platform::adapter::PlatformAdapter;
    use crate::platform::discord::DiscordAdapter;
    use crate::platform::telegram::TelegramAdapter;
    use crate::platform::whatsapp::WhatsAppAdapter;
    use crate::platform::custom::{CustomWebhookAdapter, CustomWebhookConfig};
    use crate::platform::registry::PlatformRegistry;

    #[test]
    fn test_discord_adapter_creation() {
        let config = AppConfig::default();
        let adapter = DiscordAdapter::new(&config.platform.discord);
        let status = adapter.status();
        assert_eq!(status.name, "Discord");
        // Default config has no token, so not configured
        assert!(!adapter.is_configured());
    }

    #[test]
    fn test_telegram_adapter_creation() {
        let config = AppConfig::default();
        let adapter = TelegramAdapter::new(&config.platform.telegram);
        let status = adapter.status();
        assert_eq!(status.name, "Telegram");
        assert!(!adapter.is_configured());
    }

    #[test]
    fn test_whatsapp_adapter_creation() {
        let config = AppConfig::default();
        let adapter = WhatsAppAdapter::new(&config.platform.whatsapp);
        let status = adapter.status();
        assert_eq!(status.name, "WhatsApp");
        assert!(!adapter.is_configured());
    }

    #[test]
    fn test_custom_adapter_creation() {
        let adapter = CustomWebhookAdapter::new(CustomWebhookConfig::default());
        let status = adapter.status();
        assert_eq!(status.name, "Custom");
        assert!(!adapter.is_configured());
    }

    #[test]
    fn test_platform_registry_register() {
        let mut registry = PlatformRegistry::new();
        let config = AppConfig::default();
        registry.register(Box::new(DiscordAdapter::new(&config.platform.discord)));
        let statuses = registry.statuses();
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].name, "Discord");
    }

    #[test]
    fn test_platform_registry_all_platforms() {
        let mut registry = PlatformRegistry::new();
        let config = AppConfig::default();
        registry.register(Box::new(DiscordAdapter::new(&config.platform.discord)));
        registry.register(Box::new(TelegramAdapter::new(&config.platform.telegram)));
        registry.register(Box::new(WhatsAppAdapter::new(&config.platform.whatsapp)));
        registry.register(Box::new(CustomWebhookAdapter::new(CustomWebhookConfig::default())));
        let statuses = registry.statuses();
        assert_eq!(statuses.len(), 4);
    }

    #[test]
    fn test_platform_status_summary() {
        let mut registry = PlatformRegistry::new();
        let config = AppConfig::default();
        registry.register(Box::new(DiscordAdapter::new(&config.platform.discord)));
        let summary = registry.status_summary();
        assert!(summary.contains("Discord"), "Summary: {}", summary);
    }

    #[test]
    fn test_discord_chunk_message() {
        let chunks = crate::platform::discord::chunk_message("hello", 2000);
        assert_eq!(chunks.len(), 1);

        let long = "x".repeat(5000);
        let chunks = crate::platform::discord::chunk_message(&long, 2000);
        assert!(chunks.len() >= 3);
        for chunk in &chunks {
            assert!(chunk.len() <= 2000);
        }
    }
}
