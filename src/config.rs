use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ── General ──────────────────────────────────────────────────────────

/// Top-level application configuration.
/// Every model-specific value is left at zero / empty until auto-derived from the provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub general: GeneralConfig,
    pub ollama: OllamaConfig,
    pub llamacpp: LlamaCppConfig,
    pub lmstudio: LMStudioConfig,
    pub huggingface: HuggingFaceConfig,
    pub neo4j: Neo4jConfig,
    pub observer: ObserverConfig,
    pub prompts: PromptConfig,
    pub platform: PlatformConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    /// Active provider id: "llamacpp", "ollama", "lmstudio", "huggingface"
    pub active_provider: String,
    /// Active model name (provider-specific).
    pub active_model: String,
    /// Auto-derived from the provider — never set manually. 0 = not yet derived.
    pub context_window: u64,
    /// Stream tokens to the TUI as they arrive.
    pub stream_responses: bool,
    /// Data directory for sessions, logs, timeline, vectors, persona.
    pub data_dir: PathBuf,
}

// ── Provider Configs ─────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamaCppConfig {
    /// Path to the llama-server binary.
    pub server_binary: String,
    /// Port the server listens on.
    pub port: u16,
    /// Path to the main model GGUF file.
    pub model_path: String,
    /// Path to the multimodal projector GGUF file (if applicable).
    pub mmproj_path: String,
    /// Number of GPU layers to offload. -1 = all (Metal full offload).
    pub n_gpu_layers: i32,
    /// Additional CLI args passed to llama-server.
    pub extra_args: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaConfig {
    pub host: String,
    pub port: u16,
    /// keep_alive setting: -1 = keep loaded forever.
    pub keep_alive: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMStudioConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceConfig {
    /// HuggingFace API token. Read from HUGGINGFACE_TOKEN env. Empty = public access only.
    pub api_token: String,
    /// Default model id for inference API.
    pub model_id: String,
    /// Inference endpoint base URL.
    pub endpoint: String,
}

// ── Neo4j ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Neo4jConfig {
    pub uri: String,
    pub username: String,
    pub password: String,
    pub database: String,
}

// ── Observer ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserverConfig {
    /// Enable the Observer/Skeptic audit on every reply_request.
    pub enabled: bool,
    /// Model to use for the observer. Empty string = use the same model as chat.
    pub model: String,
    /// Enable reasoning/thinking tokens for the observer.
    pub think: bool,
    /// Maximum consecutive audit rejections before force-deliver.
    pub max_rejections: usize,
}

// ── Prompts ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptConfig {
    /// Prompt 1: Core/Kernel — operational protocols.
    pub system_prompt: String,
    /// Prompt 3: Identity — persona overlay. Loaded from persona file.
    pub identity_prompt: String,
    /// Prompt 2: Contextual HUD — regenerated automatically, never stored.
    #[serde(skip)]
    pub dynamic_prompt: String,
}

// ── Platform ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformConfig {
    pub discord: DiscordConfig,
    pub telegram: TelegramConfig,
    pub whatsapp: WhatsAppConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscordConfig {
    pub enabled: bool,
    pub token: String,
    pub admin_user_id: String,
    pub guild_id: String,
    pub autonomy_channel_id: String,
    pub listen_channels: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelegramConfig {
    pub enabled: bool,
    pub token: String,
    pub admin_user_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatsAppConfig {
    pub enabled: bool,
    pub token: String,
    pub phone_number_id: String,
    pub verify_token: String,
    pub webhook_port: u16,
}

// ── ModelDerivedDefaults ─────────────────────────────────────────────

/// Operational defaults computed from model capabilities at runtime.
/// No hardcoded values — everything scales with the active model.
#[derive(Debug, Clone)]
pub struct ModelDerivedDefaults {
    /// Consolidation triggers at this fraction of context usage.
    pub consolidation_threshold: f32,
    /// Tokens per timeline/RAG chunk (context_window / 256, clamped 128–2048).
    pub chunk_size: usize,
    /// Overlap between chunks (chunk_size / 8).
    pub chunk_overlap: usize,
    /// Auto-recall top-K (scales with context: ctx/32768, clamped 2–10).
    pub auto_recall_top_k: usize,
    /// KG edge decay rate (biological constant).
    pub kg_decay_rate: f32,
    /// Lesson auto-inject confidence threshold.
    pub lesson_auto_inject_threshold: f32,
    /// Max scratchpad tokens (context_window / 16).
    pub scratchpad_max_tokens: usize,
    /// Memory context injection budget (15% of context_window).
    pub memory_context_budget: usize,
    /// Embedding dimensions (probed from model at startup).
    pub embedding_dimensions: usize,
}

impl ModelDerivedDefaults {
    /// Compute all operational defaults from the active model's context window.
    pub fn from_context_window(ctx: usize, embedding_dims: usize) -> Self {
        Self {
            consolidation_threshold: 0.80,
            chunk_size: (ctx / 256).clamp(128, 2048),
            chunk_overlap: ((ctx / 256) / 8).clamp(16, 256),
            auto_recall_top_k: (ctx / 32768).clamp(2, 10),
            kg_decay_rate: 0.99,
            lesson_auto_inject_threshold: 0.8,
            scratchpad_max_tokens: ctx / 16,
            memory_context_budget: ctx * 15 / 100,
            embedding_dimensions: embedding_dims,
        }
    }
}

// ── Default Implementations ──────────────────────────────────────────

impl Default for AppConfig {
    fn default() -> Self {
        let data_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".ernosagent");

        Self {
            general: GeneralConfig {
                active_provider: std::env::var("ERNOSAGENT_PROVIDER")
                    .unwrap_or_else(|_| "llamacpp".to_string()),
                active_model: std::env::var("ERNOSAGENT_MODEL")
                    .unwrap_or_else(|_| "gemma4:26b".to_string()),
                context_window: 0,
                stream_responses: true,
                data_dir,
            },
            ollama: OllamaConfig {
                host: std::env::var("OLLAMA_HOST")
                    .unwrap_or_else(|_| "http://localhost:11434".to_string()),
                port: std::env::var("OLLAMA_PORT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(11434),
                keep_alive: -1,
            },
            llamacpp: LlamaCppConfig {
                server_binary: std::env::var("LLAMACPP_SERVER_BIN")
                    .unwrap_or_else(|_| "llama-server".to_string()),
                port: std::env::var("LLAMACPP_PORT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(8080),
                model_path: std::env::var("LLAMACPP_MODEL_PATH")
                    .unwrap_or_default(),
                mmproj_path: std::env::var("LLAMACPP_MMPROJ_PATH")
                    .unwrap_or_default(),
                n_gpu_layers: std::env::var("LLAMACPP_GPU_LAYERS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(-1), // -1 = full offload to Metal
                extra_args: Vec::new(),
            },
            lmstudio: LMStudioConfig {
                host: std::env::var("LMSTUDIO_HOST")
                    .unwrap_or_else(|_| "http://localhost:1234".to_string()),
                port: std::env::var("LMSTUDIO_PORT")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1234),
            },
            huggingface: HuggingFaceConfig {
                api_token: std::env::var("HUGGINGFACE_TOKEN")
                    .unwrap_or_default(),
                model_id: String::new(),
                endpoint: "https://api-inference.huggingface.co".to_string(),
            },
            neo4j: Neo4jConfig::auto_detect(),
            observer: ObserverConfig {
                enabled: std::env::var("ERNOSAGENT_OBSERVER_ENABLED")
                    .map(|v| v != "0" && v.to_lowercase() != "false")
                    .unwrap_or(true),
                model: std::env::var("ERNOSAGENT_OBSERVER_MODEL")
                    .unwrap_or_default(),
                think: false,
                max_rejections: 3,
            },
            prompts: PromptConfig {
                system_prompt: String::new(), // loaded at startup from core.rs
                identity_prompt: String::new(), // loaded from persona file
                dynamic_prompt: String::new(),
            },
            platform: PlatformConfig {
                discord: DiscordConfig {
                    enabled: std::env::var("ERNOSAGENT_DISCORD_ENABLED")
                        .map(|v| v == "1" || v.to_lowercase() == "true")
                        .unwrap_or(false),
                    token: std::env::var("ERNOSAGENT_DISCORD_TOKEN").unwrap_or_default(),
                    admin_user_id: std::env::var("ERNOSAGENT_DISCORD_ADMIN").unwrap_or_default(),
                    guild_id: std::env::var("ERNOSAGENT_DISCORD_GUILD").unwrap_or_default(),
                    autonomy_channel_id: std::env::var("ERNOSAGENT_DISCORD_AUTONOMY_CHANNEL")
                        .unwrap_or_default(),
                    listen_channels: std::env::var("ERNOSAGENT_DISCORD_LISTEN_CHANNELS")
                        .unwrap_or_default()
                        .split(',')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                },
                telegram: TelegramConfig {
                    enabled: std::env::var("ERNOSAGENT_TELEGRAM_ENABLED")
                        .map(|v| v == "1" || v.to_lowercase() == "true")
                        .unwrap_or(false),
                    token: std::env::var("ERNOSAGENT_TELEGRAM_TOKEN").unwrap_or_default(),
                    admin_user_id: std::env::var("ERNOSAGENT_TELEGRAM_ADMIN").unwrap_or_default(),
                },
                whatsapp: WhatsAppConfig {
                    enabled: std::env::var("ERNOSAGENT_WHATSAPP_ENABLED")
                        .map(|v| v == "1" || v.to_lowercase() == "true")
                        .unwrap_or(false),
                    token: std::env::var("ERNOSAGENT_WHATSAPP_TOKEN").unwrap_or_default(),
                    phone_number_id: std::env::var("ERNOSAGENT_WHATSAPP_PHONE_ID")
                        .unwrap_or_default(),
                    verify_token: std::env::var("ERNOSAGENT_WHATSAPP_VERIFY_TOKEN")
                        .unwrap_or_default(),
                    webhook_port: std::env::var("ERNOSAGENT_WHATSAPP_PORT")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(3000),
                },
            },
        }
    }
}

impl Neo4jConfig {
    /// Auto-detect Neo4j configuration from env vars, then Docker Compose if present.
    pub fn auto_detect() -> Self {
        let mut config = Self {
            uri: std::env::var("NEO4J_URI")
                .unwrap_or_else(|_| "bolt://localhost:7687".to_string()),
            username: std::env::var("NEO4J_USERNAME")
                .unwrap_or_else(|_| "neo4j".to_string()),
            password: std::env::var("NEO4J_PASSWORD")
                .unwrap_or_else(|_| "ernosagent".to_string()),
            database: std::env::var("NEO4J_DATABASE")
                .unwrap_or_else(|_| "neo4j".to_string()),
        };

        // If using default URI, check docker-compose.yml for custom port mapping
        if config.uri == "bolt://localhost:7687" {
            if let Ok(compose) = std::fs::read_to_string("docker-compose.yml") {
                if compose.contains("neo4j") {
                    for line in compose.lines() {
                        let trimmed = line.trim().trim_matches('"').trim_matches('\'');
                        if trimmed.contains(":7687") && !trimmed.starts_with('#') {
                            if let Some(host_port) = trimmed.strip_suffix(":7687") {
                                let port = host_port.trim().trim_start_matches('-').trim();
                                if port.parse::<u16>().is_ok() {
                                    config.uri = format!("bolt://localhost:{}", port);
                                }
                            }
                        }
                    }
                    for line in compose.lines() {
                        if line.contains("NEO4J_AUTH=") {
                            if let Some(auth) = line.split("NEO4J_AUTH=").nth(1) {
                                let auth = auth.trim().trim_matches('"').trim_matches('\'');
                                if let Some((user, pass)) = auth.split_once('/') {
                                    config.username = user.to_string();
                                    config.password = pass.to_string();
                                }
                            }
                        }
                    }
                }
            }
        }

        config
    }
}

impl AppConfig {
    /// Load config from the data directory, or create defaults.
    pub fn load() -> Result<Self> {
        let config = Self::default();
        let config_path = config.general.data_dir.join("config.toml");

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)
                .with_context(|| format!("Failed to read config file: {}", config_path.display()))?;
            let loaded: Self = toml::from_str(&content)
                .with_context(|| format!("Failed to parse config file: {}", config_path.display()))?;
            Ok(loaded)
        } else {
            // Ensure data directory exists
            std::fs::create_dir_all(&config.general.data_dir)
                .with_context(|| {
                    format!(
                        "Failed to create data directory: {}",
                        config.general.data_dir.display()
                    )
                })?;
            Ok(config)
        }
    }

    /// Save current config to disk.
    pub fn save(&self) -> Result<()> {
        std::fs::create_dir_all(&self.general.data_dir)
            .with_context(|| {
                format!(
                    "Failed to create data directory: {}",
                    self.general.data_dir.display()
                )
            })?;

        let config_path = self.general.data_dir.join("config.toml");
        let content = toml::to_string_pretty(self)
            .context("Failed to serialize config")?;
        std::fs::write(&config_path, content)
            .with_context(|| format!("Failed to write config file: {}", config_path.display()))?;
        Ok(())
    }

    /// Resolve the persona file path.
    pub fn persona_path(&self) -> PathBuf {
        self.general.data_dir.join("persona.txt")
    }

    /// Resolve the sessions directory path.
    pub fn sessions_dir(&self) -> PathBuf {
        self.general.data_dir.join("sessions")
    }

    /// Resolve the logs directory path.
    pub fn logs_dir(&self) -> PathBuf {
        self.general.data_dir.join("logs")
    }

    /// Resolve the vectors directory path.
    pub fn vectors_dir(&self) -> PathBuf {
        self.general.data_dir.join("vectors")
    }

    /// Resolve the timeline directory path.
    pub fn timeline_dir(&self) -> PathBuf {
        self.general.data_dir.join("timeline")
    }

    /// Get the llama-server base URL.
    pub fn llamacpp_url(&self) -> String {
        format!("http://localhost:{}", self.llamacpp.port)
    }

    /// Get the Ollama base URL.
    pub fn ollama_url(&self) -> String {
        let host = self.ollama.host.trim_end_matches('/');
        if host.contains(':') && !host.ends_with(&format!(":{}", self.ollama.port)) {
            host.to_string()
        } else {
            format!("{}:{}", host.trim_end_matches(&format!(":{}", self.ollama.port)), self.ollama.port)
        }
    }

    /// Get the LMStudio base URL.
    pub fn lmstudio_url(&self) -> String {
        format!("{}/v1", self.lmstudio.host.trim_end_matches('/'))
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_has_valid_structure() {
        let config = AppConfig::default();
        assert_eq!(config.general.active_provider, "llamacpp");
        assert_eq!(config.general.context_window, 0); // must be auto-derived
        assert!(config.general.stream_responses);
        assert!(config.general.data_dir.ends_with(".ernosagent"));
    }

    #[test]
    fn test_default_neo4j_config() {
        let neo4j = Neo4jConfig::auto_detect();
        assert!(neo4j.uri.starts_with("bolt://"));
        assert_eq!(neo4j.username, "neo4j");
        assert_eq!(neo4j.database, "neo4j");
    }

    #[test]
    fn test_observer_defaults_enabled() {
        let config = AppConfig::default();
        assert!(config.observer.enabled);
        assert!(config.observer.model.is_empty()); // same as chat model
        assert_eq!(config.observer.max_rejections, 3);
    }

    #[test]
    fn test_llamacpp_url() {
        let config = AppConfig::default();
        assert_eq!(config.llamacpp_url(), "http://localhost:8080");
    }

    #[test]
    fn test_lmstudio_url() {
        let config = AppConfig::default();
        assert_eq!(config.lmstudio_url(), "http://localhost:1234/v1");
    }

    #[test]
    fn test_data_dir_paths() {
        let config = AppConfig::default();
        assert!(config.persona_path().ends_with("persona.txt"));
        assert!(config.sessions_dir().ends_with("sessions"));
        assert!(config.logs_dir().ends_with("logs"));
        assert!(config.vectors_dir().ends_with("vectors"));
        assert!(config.timeline_dir().ends_with("timeline"));
    }

    #[test]
    fn test_model_derived_defaults_from_context() {
        let defaults = ModelDerivedDefaults::from_context_window(262144, 768);
        assert_eq!(defaults.consolidation_threshold, 0.80);
        assert_eq!(defaults.chunk_size, 1024);
        assert_eq!(defaults.chunk_overlap, 128);
        assert_eq!(defaults.auto_recall_top_k, 8);
        assert_eq!(defaults.scratchpad_max_tokens, 16384);
        assert_eq!(defaults.memory_context_budget, 39321);
        assert_eq!(defaults.embedding_dimensions, 768);
    }

    #[test]
    fn test_model_derived_defaults_clamp_small_context() {
        let defaults = ModelDerivedDefaults::from_context_window(4096, 384);
        assert_eq!(defaults.chunk_size, 128); // clamped to minimum
        assert_eq!(defaults.chunk_overlap, 16); // clamped to minimum
        assert_eq!(defaults.auto_recall_top_k, 2); // clamped to minimum
    }

    #[test]
    fn test_platform_defaults_disabled() {
        let config = AppConfig::default();
        assert!(!config.platform.discord.enabled);
        assert!(!config.platform.telegram.enabled);
        assert!(!config.platform.whatsapp.enabled);
    }

    #[test]
    fn test_config_roundtrip_serialize() {
        let config = AppConfig::default();
        let serialized = toml::to_string_pretty(&config).expect("serialize");
        let deserialized: AppConfig = toml::from_str(&serialized).expect("deserialize");
        assert_eq!(deserialized.general.active_provider, config.general.active_provider);
        assert_eq!(deserialized.general.active_model, config.general.active_model);
        assert_eq!(deserialized.neo4j.uri, config.neo4j.uri);
        assert_eq!(deserialized.observer.enabled, config.observer.enabled);
    }
}
