//! Provider trait — unified interface for all inference backends.
//!
//! Every provider implements this trait. The system auto-derives model specs
//! from the provider's API. No values are hardcoded.

pub mod llamacpp;
pub mod ollama;
pub mod lmstudio;
pub mod huggingface;

use crate::model::spec::{ModelSpec, ModelSummary, Modality};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

/// A message in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Base64-encoded images (for vision models).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
}

/// A tool definition passed to the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: ToolFunction,
}

/// Function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Events emitted during streaming inference.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A text token arrived.
    Token(String),
    /// A thinking/reasoning token.
    Thinking(String),
    /// A tool call was detected.
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    /// Inference is complete.
    Done {
        total_tokens: u64,
        prompt_tokens: u64,
        completion_tokens: u64,
    },
    /// An error occurred during streaming.
    Error(String),
}

/// Provider health status.
#[derive(Debug, Clone)]
pub struct ProviderStatus {
    pub available: bool,
    pub latency_ms: Option<u64>,
    pub error: Option<String>,
    pub models_loaded: Vec<String>,
}

/// The unified provider interface.
///
/// Every inference backend (llama-server, Ollama, LMStudio, HuggingFace)
/// implements this trait. Model specs are auto-derived via `get_model_spec()`,
/// never hardcoded.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Unique identifier for this provider (e.g. "llamacpp", "ollama").
    fn id(&self) -> &str;

    /// Human-readable display name.
    fn display_name(&self) -> &str;

    /// List all available models from this provider.
    async fn list_models(&self) -> Result<Vec<ModelSummary>>;

    /// Auto-derive the full model spec from the provider's API.
    /// This is the ONLY way model specs are created — nothing is hardcoded.
    async fn get_model_spec(&self, model: &str) -> Result<ModelSpec>;

    /// Send a chat request with streaming responses.
    /// Tokens are sent through the `tx` channel as `StreamEvent`s.
    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<()>;

    /// Send a non-streaming chat request. Returns the full response.
    /// Used by the Observer audit system.
    async fn chat_sync(
        &self,
        model: &str,
        messages: &[Message],
        temperature: Option<f64>,
    ) -> Result<String>;

    /// Check if a model supports a specific modality.
    async fn supports_modality(&self, model: &str, modality: Modality) -> Result<bool>;

    /// Generate embeddings for the given text.
    async fn embed(&self, text: &str, model: &str) -> Result<Vec<f32>>;

    /// Check provider health and availability.
    async fn health(&self) -> Result<ProviderStatus>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_serialization() {
        let msg = Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
            images: vec!["base64data".to_string()],
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"images\":[\"base64data\"]"));
    }

    #[test]
    fn test_message_no_images_skipped() {
        let msg = Message {
            role: "assistant".to_string(),
            content: "Hi".to_string(),
            images: Vec::new(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("images"));
    }

    #[test]
    fn test_tool_definition_serialization() {
        let tool = ToolDefinition {
            tool_type: "function".to_string(),
            function: ToolFunction {
                name: "web_search".to_string(),
                description: "Search the web".to_string(),
                parameters: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }),
            },
        };
        let json = serde_json::to_string(&tool).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"web_search\""));
    }

    #[test]
    fn test_stream_event_variants() {
        let token = StreamEvent::Token("hello".to_string());
        let thinking = StreamEvent::Thinking("reasoning...".to_string());
        let tool_call = StreamEvent::ToolCall {
            id: "tc1".to_string(),
            name: "search".to_string(),
            arguments: "{}".to_string(),
        };
        let done = StreamEvent::Done {
            total_tokens: 100,
            prompt_tokens: 50,
            completion_tokens: 50,
        };
        let error = StreamEvent::Error("timeout".to_string());

        // Just verify they construct without panic
        assert!(matches!(token, StreamEvent::Token(_)));
        assert!(matches!(thinking, StreamEvent::Thinking(_)));
        assert!(matches!(tool_call, StreamEvent::ToolCall { .. }));
        assert!(matches!(done, StreamEvent::Done { .. }));
        assert!(matches!(error, StreamEvent::Error(_)));
    }

    #[test]
    fn test_provider_status() {
        let status = ProviderStatus {
            available: true,
            latency_ms: Some(42),
            error: None,
            models_loaded: vec!["gemma4:26b".to_string()],
        };
        assert!(status.available);
        assert_eq!(status.latency_ms, Some(42));
    }
}
