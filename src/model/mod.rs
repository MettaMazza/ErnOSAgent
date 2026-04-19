// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Model specification types — provider-agnostic model metadata.

use serde::{Deserialize, Serialize};

/// Describes a loaded model's capabilities, derived from the provider at startup.
/// No model-family-specific fields — completely neutral.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    /// Human-readable model name (e.g. "gemma-4-26b-it")
    pub name: String,
    /// Maximum context window in tokens — auto-derived from provider, never hardcoded
    pub context_length: usize,
    /// Whether the model supports vision (images)
    pub supports_vision: bool,
    /// Whether the model supports video (frame sequences)
    pub supports_video: bool,
    /// Whether the model supports audio input
    pub supports_audio: bool,
    /// Whether the model supports native tool calling
    pub supports_tool_calling: bool,
    /// Whether the model supports thinking/reasoning mode
    pub supports_thinking: bool,
    /// Embedding dimensions (0 if no embedding support)
    pub embedding_dimensions: usize,
}

impl ModelSpec {
    /// Compute the consolidation threshold — 80% of context window.
    pub fn consolidation_threshold(&self) -> usize {
        (self.context_length as f64 * 0.80) as usize
    }

    /// Compute chunk size for memory recall — 20% of context window.
    pub fn memory_budget_tokens(&self) -> usize {
        (self.context_length as f64 * 0.20) as usize
    }

    /// Max tokens for a single response — 15% of context.
    pub fn max_response_tokens(&self) -> usize {
        (self.context_length as f64 * 0.15) as usize
    }
}

impl Default for ModelSpec {
    fn default() -> Self {
        Self {
            name: "unknown".to_string(),
            context_length: 0, // Must be set by provider — 0 signals "not yet derived"
            supports_vision: false,
            supports_video: false,
            supports_audio: false,
            supports_tool_calling: true,
            supports_thinking: false,
            embedding_dimensions: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_spec() {
        let spec = ModelSpec::default();
        assert_eq!(spec.context_length, 0); // 0 = not yet derived from provider
        assert!(!spec.supports_vision);
        assert!(!spec.supports_video);
        assert!(!spec.supports_audio);
    }

    #[test]
    fn test_derived_defaults() {
        let spec = ModelSpec {
            context_length: 256_000,
            ..Default::default()
        };
        assert_eq!(spec.consolidation_threshold(), 204_800);
        assert_eq!(spec.memory_budget_tokens(), 51_200);
        assert_eq!(spec.max_response_tokens(), 38_400);
    }
}
