// Ern-OS — Platform registry (ported from ErnOSAgent)
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Platform registry — manages active platform connections.

use crate::platform::adapter::{PlatformAdapter, PlatformStatus};

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
