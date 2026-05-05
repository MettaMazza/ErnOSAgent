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
    #[serde(default)]
    pub browser: BrowserConfig,
    #[serde(default)]
    pub discord: DiscordConfig,
    #[serde(default)]
    pub telegram: TelegramConfig,
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
    /// Maximum 1-second retries when waiting for the inference provider to
    /// become healthy at startup. Default 60 (matches legacy hardcoded value).
    /// Bump for setups where model load is genuinely slower than 60s — e.g.
    /// >20B models on a single GPU, or any model split across multiple
    /// backends via llama.cpp RPC where layer transfer takes time.
    #[serde(default = "default_provider_health_check_retries")]
    pub provider_health_check_retries: u32,
}

fn default_provider_health_check_retries() -> u32 { 60 }

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            active_provider: "llamacpp".to_string(),
            data_dir: PathBuf::from("data"),
            kokoro_port: Some(8880),
            flux_port: Some(8890),
            whisper_port: Some(8891),
            provider_health_check_retries: 60,
        }
    }
}

fn default_sae_embed_port() -> u16 { 8082 }

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
    /// SAE activation extraction port — runs the MAIN model in embedding-only
    /// mode for live SAE feature extraction. Separate from the embedding_port
    /// which runs a dedicated embedding model.
    #[serde(default = "default_sae_embed_port")]
    pub sae_embed_port: u16,
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
            sae_embed_port: 8082,
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

/// Browser tool configuration — controls headed/headless mode and viewport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserConfig {
    /// Show visible Chrome window (true) or run headless (false).
    /// Default: true on macOS (local dev), false on Linux (CI/server).
    pub headed: bool,
    /// Browser viewport width.
    pub window_width: u32,
    /// Browser viewport height.
    pub window_height: u32,
    /// Default timeout for element waits (ms).
    pub timeout_ms: u64,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            headed: cfg!(target_os = "macos"),
            window_width: 1280,
            window_height: 900,
            timeout_ms: 10000,
        }
    }
}

/// Discord platform adapter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordConfig {
    /// Bot token — also reads from DISCORD_TOKEN env var.
    #[serde(default)]
    pub token: Option<String>,
    /// User IDs with full (admin) tool access.
    #[serde(default)]
    pub admin_ids: Vec<String>,
    /// Channel IDs to respond in. Empty = all channels.
    #[serde(default)]
    pub listen_channels: Vec<String>,
    /// Whether the adapter is enabled.
    #[serde(default)]
    pub enabled: bool,
}

impl Default for DiscordConfig {
    fn default() -> Self {
        Self {
            token: None,
            admin_ids: Vec::new(),
            listen_channels: Vec::new(),
            enabled: false,
        }
    }
}

impl DiscordConfig {
    /// Resolve the bot token: config value takes priority, then env var.
    pub fn resolve_token(&self) -> Option<String> {
        self.token.clone()
            .or_else(|| std::env::var("DISCORD_TOKEN").ok())
    }

    /// Whether this adapter has valid credentials to connect.
    pub fn is_configured(&self) -> bool {
        self.enabled && self.resolve_token().is_some()
    }
}

/// Telegram platform adapter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramConfig {
    /// Bot token — also reads from TELEGRAM_TOKEN env var.
    #[serde(default)]
    pub token: Option<String>,
    /// User IDs with full (admin) tool access.
    #[serde(default)]
    pub admin_ids: Vec<i64>,
    /// Chat IDs to respond in. Empty = all chats.
    #[serde(default)]
    pub allowed_chats: Vec<i64>,
    /// Whether the adapter is enabled.
    #[serde(default)]
    pub enabled: bool,
}

impl Default for TelegramConfig {
    fn default() -> Self {
        Self {
            token: None,
            admin_ids: Vec::new(),
            allowed_chats: Vec::new(),
            enabled: false,
        }
    }
}

impl TelegramConfig {
    /// Resolve the bot token: config value takes priority, then env var.
    pub fn resolve_token(&self) -> Option<String> {
        self.token.clone()
            .or_else(|| std::env::var("TELEGRAM_TOKEN").ok())
    }

    /// Whether this adapter has valid credentials to connect.
    pub fn is_configured(&self) -> bool {
        self.enabled && self.resolve_token().is_some()
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
            browser: BrowserConfig::default(),
            discord: DiscordConfig::default(),
            telegram: TelegramConfig::default(),
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

    #[test]
    fn test_provider_health_check_retries_default_is_legacy_60() {
        // Default must match the value previously hardcoded in main.rs so
        // that existing tomls without this field deserialize to identical
        // pre-patch behavior.
        let config = AppConfig::default();
        assert_eq!(config.general.provider_health_check_retries, 60);
    }

    #[test]
    fn test_provider_health_check_retries_missing_field_deserializes_to_60() {
        // A toml file written before this field existed must still parse,
        // and the field must default to the legacy hardcoded value.
        let toml_without_field = r#"
            active_provider = "llamacpp"
            data_dir = "data"
        "#;
        let general: GeneralConfig = toml::from_str(toml_without_field).unwrap();
        assert_eq!(general.provider_health_check_retries, 60);
    }

    #[test]
    fn test_provider_health_check_retries_explicit_value_honored() {
        let toml_with_field = r#"
            active_provider = "llamacpp"
            data_dir = "data"
            provider_health_check_retries = 480
        "#;
        let general: GeneralConfig = toml::from_str(toml_with_field).unwrap();
        assert_eq!(general.provider_health_check_retries, 480);
    }
}
