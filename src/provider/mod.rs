// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Provider abstraction — trait + factory for model-neutral inference.

pub mod llamacpp;
pub mod llamacpp_embed;
pub mod ollama;
pub mod openai_compat;
pub mod stream_parser;

use crate::config::AppConfig;
use crate::model::ModelSpec;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: serde_json::Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<String>,
    /// Tool calls made by the assistant (OpenAI-compatible format).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
    /// Tool call ID this message is responding to (for role=tool).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    /// Create a simple text message.
    pub fn text(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: serde_json::Value::String(content.to_string()),
            images: Vec::new(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create an assistant message that contains a tool call.
    pub fn assistant_tool_call(call_id: &str, name: &str, arguments: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: serde_json::Value::Null,
            images: Vec::new(),
            tool_calls: Some(vec![serde_json::json!({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments
                }
            })]),
            tool_call_id: None,
        }
    }

    /// Create an assistant message containing multiple parallel tool calls.
    /// OpenAI-compatible APIs require all parallel tool calls in a single message.
    pub fn assistant_tool_calls(calls: &[(&str, &str, &str)]) -> Self {
        Self {
            role: "assistant".to_string(),
            content: serde_json::Value::Null,
            images: Vec::new(),
            tool_calls: Some(calls.iter().map(|(id, name, args)| {
                serde_json::json!({
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": args }
                })
            }).collect()),
            tool_call_id: None,
        }
    }

    /// Create a tool result message responding to a specific tool call.
    pub fn tool_result(call_id: &str, output: &str) -> Self {
        Self {
            role: "tool".to_string(),
            content: serde_json::Value::String(output.to_string()),
            images: Vec::new(),
            tool_calls: None,
            tool_call_id: Some(call_id.to_string()),
        }
    }

    /// Create a multipart message with text and images (base64 data URIs).
    pub fn multipart(role: &str, text: &str, image_urls: Vec<String>) -> Self {
        let mut parts = Vec::new();

        // Text part
        parts.push(serde_json::json!({
            "type": "text",
            "text": text
        }));

        // Image parts
        for url in &image_urls {
            parts.push(serde_json::json!({
                "type": "image_url",
                "image_url": { "url": url }
            }));
        }

        Self {
            role: role.to_string(),
            content: serde_json::Value::Array(parts),
            images: image_urls,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a multipart message with audio content (WAV base64).
    /// Gemma 4 natively processes audio via the multimodal projector.
    pub fn multimodal_audio(role: &str, audio_b64: &str) -> Self {
        let parts = vec![
            serde_json::json!({
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": "wav"
                }
            }),
        ];
        Self {
            role: role.to_string(),
            content: serde_json::Value::Array(parts),
            images: Vec::new(),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Create a multipart message with both audio and video frame.
    pub fn multimodal_av(role: &str, audio_b64: &str, frame_b64: &str) -> Self {
        let parts = vec![
            serde_json::json!({
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": "wav"
                }
            }),
            serde_json::json!({
                "type": "image_url",
                "image_url": {
                    "url": format!("data:image/jpeg;base64,{}", frame_b64)
                }
            }),
        ];
        Self {
            role: role.to_string(),
            content: serde_json::Value::Array(parts),
            images: vec![format!("data:image/jpeg;base64,{}", frame_b64)],
            tool_calls: None,
            tool_call_id: None,
        }
    }

    /// Extract the text content regardless of format.
    pub fn text_content(&self) -> String {
        match &self.content {
            serde_json::Value::String(s) => s.clone(),
            serde_json::Value::Array(parts) => {
                parts
                    .iter()
                    .filter_map(|p| {
                        if p.get("type")?.as_str()? == "text" {
                            p.get("text")?.as_str().map(|s| s.to_string())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
            _ => String::new(),
        }
    }
}

/// Events emitted during streaming inference.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A chunk of response text.
    TextDelta(String),
    /// A chunk of thinking/reasoning content.
    ThinkingDelta(String),
    /// A complete tool call accumulated from deltas.
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
    /// Stream completed — final message role.
    Done,
    /// Stream error.
    Error(String),
}

/// Provider trait — the engine's only interface to any model backend.
/// Completely model-neutral, hardware-neutral, platform-neutral.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Unique provider identifier.
    fn id(&self) -> &str;

    /// Human-readable display name.
    fn display_name(&self) -> &str;

    /// Get the model specification from the running backend.
    async fn get_model_spec(&self) -> Result<ModelSpec>;

    /// Streaming chat completion — returns a channel of StreamEvents.
    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&serde_json::Value>,
        thinking: bool,
    ) -> Result<tokio::sync::mpsc::Receiver<StreamEvent>>;

    /// Non-streaming chat completion — returns the full response text.
    /// Used for Observer audit (thinking disabled for latency).
    async fn chat_sync(
        &self,
        messages: &[Message],
        tools: Option<&serde_json::Value>,
    ) -> Result<String>;

    /// Generate embeddings for the given text.
    async fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Health check — returns true if the backend is reachable.
    async fn health(&self) -> bool;
}

/// Factory function — create the active provider from config.
pub fn create_provider(config: &AppConfig) -> Result<Box<dyn Provider>> {
    match config.general.active_provider.as_str() {
        "llamacpp" => Ok(Box::new(llamacpp::LlamaCppProvider::new(&config.llamacpp))),
        "ollama" => Ok(Box::new(ollama::OllamaProvider::new(&config.ollama))),
        "openai_compat" => Ok(Box::new(
            openai_compat::OpenAICompatProvider::new(&config.openai_compat),
        )),
        other => anyhow::bail!(
            "Unknown provider '{}'. Valid: llamacpp, ollama, openai_compat",
            other
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_text_message() {
        let msg = Message::text("user", "Hello");
        assert_eq!(msg.text_content(), "Hello");
    }

    #[test]
    fn test_multipart_message() {
        let msg = Message::multipart("user", "Describe this", vec![
            "data:image/png;base64,abc123".to_string(),
        ]);
        assert_eq!(msg.text_content(), "Describe this");
        assert_eq!(msg.images.len(), 1);
    }

    #[test]
    fn test_empty_content() {
        let msg = Message {
            role: "user".to_string(),
            content: serde_json::Value::Null,
            images: Vec::new(),
            tool_calls: None,
            tool_call_id: None,
        };
        assert_eq!(msg.text_content(), "");
    }
}
