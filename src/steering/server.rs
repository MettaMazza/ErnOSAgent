//! llama-server lifecycle management for steering vector changes.
//!
//! When steering vectors or models change, the server must be restarted
//! with the new arguments. This module coordinates that lifecycle.

use crate::provider::llamacpp::LlamaCppProvider;
use crate::steering::vectors::SteeringConfig;
use anyhow::{Context, Result};
use std::sync::Arc;

/// Manages the llama-server lifecycle in coordination with steering config changes.
pub struct SteeringServer {
    provider: Arc<LlamaCppProvider>,
    current_config: SteeringConfig,
}

impl SteeringServer {
    pub fn new(provider: Arc<LlamaCppProvider>) -> Self {
        Self {
            provider,
            current_config: SteeringConfig::default(),
        }
    }

    /// Apply new steering config by restarting the server with updated args.
    pub async fn apply(&mut self, config: SteeringConfig) -> Result<()> {
        let args = config.to_server_args();

        tracing::info!(
            active_vectors = config.active_vectors().len(),
            layer_range = ?config.layer_range,
            "Applying steering config — restarting llama-server"
        );

        self.provider
            .restart_server(&args)
            .await
            .context("Failed to restart llama-server with new steering config")?;

        self.current_config = config;

        tracing::info!("Steering config applied successfully");
        Ok(())
    }

    /// Get the current steering config.
    pub fn config(&self) -> &SteeringConfig {
        &self.current_config
    }

    /// Get a mutable reference to the current steering config.
    pub fn config_mut(&mut self) -> &mut SteeringConfig {
        &mut self.current_config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_steering_server_default_config() {
        // Can't create a real LlamaCppProvider in tests without a server,
        // but we can verify the config starts empty.
        let config = SteeringConfig::default();
        assert!(config.vectors.is_empty());
        assert!(config.layer_range.is_none());
    }
}
