// Ern-OS — Steering system module
//! Real-time cognitive steering via GGUF control vectors.

pub mod vectors;
pub mod server;

use serde::{Deserialize, Serialize};

/// A loaded steering vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteeringVector {
    pub name: String,
    pub path: String,
    pub strength: f32,
    pub active: bool,
    pub description: String,
}

/// Steering configuration for llama-server args.
#[derive(Debug, Clone, Default)]
pub struct SteeringConfig {
    pub vectors: Vec<SteeringVector>,
}

impl SteeringConfig {
    /// Convert active vectors to llama-server command-line arguments.
    pub fn to_server_args(&self) -> Vec<String> {
        tracing::info!(module = "steering", fn_name = "to_server_args", "steering::to_server_args called");
        let mut args = Vec::new();
        for v in &self.vectors {
            if v.active {
                args.push("--control-vector".to_string());
                args.push(v.path.clone());
                args.push("--control-vector-scaled".to_string());
                args.push(format!("{:.2}", v.strength));
            }
        }
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_args_empty() {
        let config = SteeringConfig::default();
        assert!(config.to_server_args().is_empty());
    }

    #[test]
    fn test_to_args_active() {
        let config = SteeringConfig {
            vectors: vec![SteeringVector {
                name: "curiosity".into(), path: "vectors/curiosity.gguf".into(),
                strength: 0.8, active: true, description: "Increases curiosity".into(),
            }],
        };
        let args = config.to_server_args();
        assert!(args.contains(&"--control-vector".to_string()));
        assert!(args.contains(&"vectors/curiosity.gguf".to_string()));
    }
}
