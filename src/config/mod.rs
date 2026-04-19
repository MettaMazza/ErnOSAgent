// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Application configuration — loaded from `ern-os.toml` or environment.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Top-level application configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub general: GeneralConfig,
    #[serde(default)]
    pub llamacpp: LlamaCppConfig,
    #[serde(default)]
    pub ollama: OllamaConfig,
    #[serde(default)]
    pub openai_compat: OpenAICompatConfig,
    #[serde(default)]
    pub observer: ObserverConfig,
    #[serde(default)]
    pub web: WebConfig,
    #[serde(default)]
    pub prompt: PromptConfig,
    #[serde(default)]
    pub codes: CodesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Active provider: "llamacpp", "ollama", "openai_compat"
    pub active_provider: String,
    /// Data directory for persistence (logs, sessions, memory, etc.)
    pub data_dir: PathBuf,
    /// Port for local Kokoro TTS server
    #[serde(default)]
    pub kokoro_port: Option<u16>,
    /// Port for local Flux image generation server
    #[serde(default)]
    pub flux_port: Option<u16>,
    /// Port for local Whisper STT server (used by voice calls)
    #[serde(default)]
    pub whisper_port: Option<u16>,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            active_provider: "llamacpp".to_string(),
            data_dir: PathBuf::from("data"),
            kokoro_port: Some(8880),
            flux_port: Some(8890),
            whisper_port: Some(8891),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppConfig {
    /// Path to llama-server binary
    pub server_binary: String,
    /// Port for the inference server
    pub port: u16,
    /// Path to the model GGUF file
    pub model_path: String,
    /// Path to the multimodal projector GGUF (for vision)
    pub mmproj_path: Option<String>,
    /// Number of GPU layers to offload (-1 or 999 for full)
    pub n_gpu_layers: i32,
    /// Embedding server port
    pub embedding_port: u16,
    /// Embedding model path
    pub embedding_model: Option<String>,
    /// Visual token budget (70, 140, 280, 560, 1120)
    pub visual_token_budget: usize,
    /// Optional LoRA adapter to load at inference time
    pub lora_adapter: Option<String>,
}

impl Default for LlamaCppConfig {
    fn default() -> Self {
        Self {
            server_binary: "llama-server".to_string(),
            port: 8080,
            model_path: "./models/gemma-4-31B-it-Q4_K_M.gguf".to_string(),
            mmproj_path: Some(
                "./models/mmproj-F16.gguf".to_string(),
            ),
            n_gpu_layers: 999,
            embedding_port: 8081,
            embedding_model: None,
            visual_token_budget: 560,
            lora_adapter: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub base_url: String,
    pub model: String,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            model: "gemma4:26b".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
}

impl Default for OpenAICompatConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:1234/v1".to_string(),
            api_key: None,
            model: "gemma-4-26b-it".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserverConfig {
    /// Enable the observer audit system
    pub enabled: bool,
    // No bailout — retries until correct or user sends /stop
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            enabled: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebConfig {
    pub port: u16,
    pub open_browser: bool,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            port: 3000,
            open_browser: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptConfig {
    pub system_prompt: String,
    pub thinking_enabled: bool,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are a helpful, accurate, and capable AI assistant."
                .to_string(),
            thinking_enabled: true,
        }
    }
}

/// Configuration for the integrated VS Code IDE (code-server).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodesConfig {
    /// Enable auto-start of code-server.
    pub enabled: bool,
    /// Port for code-server.
    pub port: u16,
    /// Default workspace path.
    pub workspace: String,
}

impl Default for CodesConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            port: 8443,
            workspace: ".".to_string(),
        }
    }
}

impl AppConfig {
    /// Load config from `ern-os.toml` in the current directory, or use defaults.
    pub fn load() -> Result<Self> {
        let config_path = PathBuf::from("ern-os.toml");

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)
                .with_context(|| "Failed to read ern-os.toml")?;
            let config: AppConfig = toml::from_str(&content)
                .with_context(|| "Failed to parse ern-os.toml")?;
            tracing::info!("Loaded config from ern-os.toml");
            Ok(config)
        } else {
            tracing::info!("No ern-os.toml found, using defaults");
            Ok(Self::default())
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            llamacpp: LlamaCppConfig::default(),
            ollama: OllamaConfig::default(),
            openai_compat: OpenAICompatConfig::default(),
            observer: ObserverConfig::default(),
            web: WebConfig::default(),
            prompt: PromptConfig::default(),
            codes: CodesConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.general.active_provider, "llamacpp");
        assert_eq!(config.llamacpp.port, 8080);
        assert_eq!(config.web.port, 3000);
        assert!(config.observer.enabled);
    }

    #[test]
    fn test_toml_roundtrip() {
        let config = AppConfig::default();
        let serialized = toml::to_string_pretty(&config).unwrap();
        let deserialized: AppConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(deserialized.general.active_provider, "llamacpp");
    }
}
