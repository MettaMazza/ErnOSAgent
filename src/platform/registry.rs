//! Platform registry — manages active platform connections.

use crate::platform::adapter::{PlatformAdapter, PlatformStatus};
use std::collections::HashMap;

pub struct PlatformRegistry {
    adapters: HashMap<String, Box<dyn PlatformAdapter>>,
}

impl PlatformRegistry {
    pub fn new() -> Self {
        Self { adapters: HashMap::new() }
    }

    pub fn register(&mut self, adapter: Box<dyn PlatformAdapter>) {
        let name = adapter.name().to_string();
        self.adapters.insert(name, adapter);
    }

    pub fn statuses(&self) -> Vec<PlatformStatus> {
        self.adapters.values().map(|a| a.status()).collect()
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
