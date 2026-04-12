// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Expert Model Selector — auto-selects the largest available model for distillation.
//!
//! Selection priority:
//!   1. `ERNOS_EXPERT_MODEL` env override (always wins)
//!   2. Cloud API providers (if API keys are set)
//!   3. Largest local model available via the inference provider

use crate::model::spec::ModelSummary;
use crate::provider::Provider;
use std::sync::Arc;

/// The resolved expert model backend.
#[derive(Debug, Clone)]
pub enum ExpertBackend {
    /// A cloud API provider with a specific model.
    CloudApi { provider: String, model: String },
    /// A local model available through the inference provider.
    LocalModel { name: String, parameter_size: String },
    /// User-specified override via env var.
    EnvOverride { model: String },
}

impl ExpertBackend {
    /// Get the model name string for inference.
    pub fn model_name(&self) -> &str {
        match self {
            Self::CloudApi { model, .. } => model,
            Self::LocalModel { name, .. } => name,
            Self::EnvOverride { model } => model,
        }
    }
}

impl std::fmt::Display for ExpertBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CloudApi { provider, model } => write!(f, "{model} (via {provider})"),
            Self::LocalModel { name, parameter_size } => write!(f, "{name} ({parameter_size})"),
            Self::EnvOverride { model } => write!(f, "{model} (env override)"),
        }
    }
}

/// Select the best expert model for distillation.
///
/// Priority: env override → cloud API → largest local model.
pub async fn select_expert_model(
    provider: &Arc<dyn Provider>,
) -> anyhow::Result<ExpertBackend> {
    // 1. Check env override
    if let Some(backend) = check_env_override() {
        tracing::info!(model = %backend, "Expert model: env override");
        return Ok(backend);
    }

    // 2. Check cloud API providers
    if let Some(backend) = detect_cloud_provider() {
        tracing::info!(model = %backend, "Expert model: cloud API");
        return Ok(backend);
    }

    // 3. Fall back to largest local model
    select_largest_local(provider).await
}

/// Check for `ERNOS_EXPERT_MODEL` env override.
fn check_env_override() -> Option<ExpertBackend> {
    std::env::var("ERNOS_EXPERT_MODEL").ok().map(|model| {
        ExpertBackend::EnvOverride { model }
    })
}

/// Detect available cloud API providers via env var keys.
///
/// Returns the first available provider with the largest model.
fn detect_cloud_provider() -> Option<ExpertBackend> {
    // Check providers in order of preference
    let providers = [
        ("ANTHROPIC_API_KEY", "anthropic", "claude-sonnet-4-20250514"),
        ("OPENAI_API_KEY", "openai", "gpt-4o"),
        ("GOOGLE_API_KEY", "google", "gemini-2.5-pro"),
    ];

    for (key, provider, model) in &providers {
        if std::env::var(key).is_ok() {
            tracing::info!(provider = %provider, key = %key, "Cloud API key detected");
            return Some(ExpertBackend::CloudApi {
                provider: provider.to_string(),
                model: model.to_string(),
            });
        }
    }

    None
}

/// Query the local inference provider for available models and select the largest.
async fn select_largest_local(
    provider: &Arc<dyn Provider>,
) -> anyhow::Result<ExpertBackend> {
    let models = provider.list_models().await?;

    if models.is_empty() {
        anyhow::bail!(
            "No local models available from provider '{}'. \
             Set ERNOS_EXPERT_MODEL or provide an API key.",
            provider.id()
        );
    }

    let largest = find_largest_model(&models);

    tracing::info!(
        model = %largest.name,
        size = %largest.parameter_size,
        provider = %largest.provider,
        candidates = models.len(),
        "Expert model: largest local model selected"
    );

    Ok(ExpertBackend::LocalModel {
        name: largest.name.clone(),
        parameter_size: largest.parameter_size.clone(),
    })
}

/// Find the model with the largest parameter count from a list of summaries.
fn find_largest_model(models: &[ModelSummary]) -> &ModelSummary {
    models.iter().max_by_key(|m| parse_param_size(&m.parameter_size)).unwrap()
}

/// Parse a parameter size string like "26B", "122B", "7.5B" into a comparable integer.
fn parse_param_size(size: &str) -> u64 {
    let cleaned = size.trim().to_uppercase();
    let (number_str, multiplier) = if cleaned.ends_with('B') {
        (&cleaned[..cleaned.len() - 1], 1_000_000_000u64)
    } else if cleaned.ends_with('M') {
        (&cleaned[..cleaned.len() - 1], 1_000_000u64)
    } else {
        (cleaned.as_str(), 1u64)
    };

    number_str
        .parse::<f64>()
        .map(|n| (n * multiplier as f64) as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_param_size_billions() {
        assert_eq!(parse_param_size("26B"), 26_000_000_000);
        assert_eq!(parse_param_size("122B"), 122_000_000_000);
    }

    #[test]
    fn test_parse_param_size_fractional() {
        assert_eq!(parse_param_size("7.5B"), 7_500_000_000);
        assert_eq!(parse_param_size("2.7B"), 2_700_000_000);
    }

    #[test]
    fn test_parse_param_size_millions() {
        assert_eq!(parse_param_size("350M"), 350_000_000);
    }

    #[test]
    fn test_parse_param_size_unknown() {
        assert_eq!(parse_param_size(""), 0);
        assert_eq!(parse_param_size("unknown"), 0);
    }

    #[test]
    fn test_expert_backend_display() {
        let cloud = ExpertBackend::CloudApi {
            provider: "openai".to_string(),
            model: "gpt-4o".to_string(),
        };
        assert_eq!(cloud.to_string(), "gpt-4o (via openai)");

        let local = ExpertBackend::LocalModel {
            name: "gemma4:26b".to_string(),
            parameter_size: "26B".to_string(),
        };
        assert_eq!(local.to_string(), "gemma4:26b (26B)");
    }

    #[test]
    fn test_find_largest_model() {
        let models = vec![
            ModelSummary {
                name: "small".to_string(),
                provider: "test".to_string(),
                parameter_size: "7B".to_string(),
                quantization_level: String::new(),
                capabilities: Default::default(),
                context_length: 0,
            },
            ModelSummary {
                name: "large".to_string(),
                provider: "test".to_string(),
                parameter_size: "70B".to_string(),
                quantization_level: String::new(),
                capabilities: Default::default(),
                context_length: 0,
            },
            ModelSummary {
                name: "medium".to_string(),
                provider: "test".to_string(),
                parameter_size: "26B".to_string(),
                quantization_level: String::new(),
                capabilities: Default::default(),
                context_length: 0,
            },
        ];
        let largest = find_largest_model(&models);
        assert_eq!(largest.name, "large");
    }

    #[test]
    fn test_env_override_model_name() {
        let backend = ExpertBackend::EnvOverride { model: "custom:latest".to_string() };
        assert_eq!(backend.model_name(), "custom:latest");
    }
}
