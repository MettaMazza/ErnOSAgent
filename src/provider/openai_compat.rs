// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! OpenAI-compatible cloud provider — accessibility fallback for users without local hardware.
//!
//! Supports any API that implements the OpenAI chat completions format:
//! - OpenAI (GPT-4, GPT-4o, etc.)
//! - Anthropic/Claude (via OpenAI-compatible endpoint)
//! - Groq (Llama, Mixtral, Gemma on Groq Cloud)
//! - Together AI
//! - Fireworks AI
//! - Perplexity
//! - OpenRouter (aggregator for all of the above)
//! - Any self-hosted vLLM, text-generation-inference, or OpenAI-compatible server
//!
//! **Important:** ErnOS is a local-first system. Cloud providers are an accessibility
//! option for users who lack local inference hardware. They have NOT been tested by the
//! maintainers (who run everything locally on Apple Silicon) and are provided as-is.
//! Local inference with llama.cpp or Ollama is the recommended and supported configuration.

use crate::model::spec::{Modality, ModelCapabilities, ModelSpec, ModelSummary};
use crate::provider::{Message, Provider, ProviderStatus, StreamEvent, ToolDefinition};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tokio::sync::mpsc;

/// Configuration for an OpenAI-compatible cloud provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAICompatConfig {
    /// Display name (e.g. "OpenAI", "Groq", "Claude")
    pub name: String,
    /// Provider identifier (e.g. "openai", "groq", "claude")
    pub provider_id: String,
    /// Base URL for the API (e.g. "https://api.openai.com/v1")
    pub base_url: String,
    /// API key (Bearer token)
    pub api_key: String,
    /// Default model to use if none specified
    pub default_model: String,
    /// Context window override (0 = auto-derive from API)
    #[serde(default)]
    pub context_window: u64,
}

impl Default for OpenAICompatConfig {
    fn default() -> Self {
        Self {
            name: "OpenAI".to_string(),
            provider_id: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            default_model: "gpt-4o".to_string(),
            context_window: 0,
        }
    }
}

pub struct OpenAICompatProvider {
    config: OpenAICompatConfig,
    client: Client,
}

impl OpenAICompatProvider {
    pub fn new(config: &OpenAICompatConfig) -> Self {
        Self {
            config: config.clone(),
            client: Client::new(),
        }
    }

    /// Create pre-configured instances for known cloud providers.
    pub fn openai(api_key: &str) -> Self {
        Self::new(&OpenAICompatConfig {
            name: "OpenAI".to_string(),
            provider_id: "openai".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: api_key.to_string(),
            default_model: "gpt-4o".to_string(),
            ..Default::default()
        })
    }

    pub fn groq(api_key: &str) -> Self {
        Self::new(&OpenAICompatConfig {
            name: "Groq".to_string(),
            provider_id: "groq".to_string(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            api_key: api_key.to_string(),
            default_model: "llama-3.3-70b-versatile".to_string(),
            ..Default::default()
        })
    }

    pub fn anthropic(api_key: &str) -> Self {
        Self::new(&OpenAICompatConfig {
            name: "Claude".to_string(),
            provider_id: "claude".to_string(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            api_key: api_key.to_string(),
            default_model: "claude-sonnet-4-20250514".to_string(),
            ..Default::default()
        })
    }

    pub fn openrouter(api_key: &str) -> Self {
        Self::new(&OpenAICompatConfig {
            name: "OpenRouter".to_string(),
            provider_id: "openrouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            api_key: api_key.to_string(),
            default_model: "google/gemma-3-27b-it".to_string(),
            ..Default::default()
        })
    }

    fn auth_headers(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut r = req.header("Authorization", format!("Bearer {}", self.config.api_key));
        // Anthropic uses a different header
        if self.config.provider_id == "claude" {
            r = r
                .header("x-api-key", &self.config.api_key)
                .header("anthropic-version", "2023-06-01");
        }
        r
    }
}

#[async_trait]
impl Provider for OpenAICompatProvider {
    fn id(&self) -> &str {
        &self.config.provider_id
    }
    fn display_name(&self) -> &str {
        &self.config.name
    }

    async fn list_models(&self) -> Result<Vec<ModelSummary>> {
        let url = format!("{}/models", self.config.base_url);
        let req = self.auth_headers(self.client.get(&url));

        match req.send().await {
            Ok(resp) if resp.status().is_success() => {
                let body: serde_json::Value = resp.json().await.unwrap_or_default();
                let models = body
                    .get("data")
                    .and_then(|d| d.as_array())
                    .map(|arr| {
                        arr.iter()
                            .map(|m| ModelSummary {
                                name: m
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("unknown")
                                    .to_string(),
                                provider: self.config.provider_id.clone(),
                                parameter_size: String::new(),
                                quantization_level: String::new(),
                                capabilities: ModelCapabilities {
                                    text: true,
                                    tool_calling: true,
                                    ..Default::default()
                                },
                                context_length: 0,
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Ok(models)
            }
            Ok(resp) => {
                tracing::warn!(status = %resp.status(), "Model list request failed — returning default model only");
                Ok(vec![ModelSummary {
                    name: self.config.default_model.clone(),
                    provider: self.config.provider_id.clone(),
                    parameter_size: String::new(),
                    quantization_level: String::new(),
                    capabilities: ModelCapabilities {
                        text: true,
                        tool_calling: true,
                        ..Default::default()
                    },
                    context_length: 0,
                }])
            }
            Err(e) => bail!("Failed to list models from {}: {}", self.config.name, e),
        }
    }

    async fn get_model_spec(&self, model: &str) -> Result<ModelSpec> {
        // Most cloud APIs don't expose detailed model specs — use sensible defaults
        let ctx = if self.config.context_window > 0 {
            self.config.context_window
        } else {
            // Well-known context windows
            match model {
                m if m.contains("gpt-4o") => 128_000,
                m if m.contains("gpt-4") => 128_000,
                m if m.contains("claude") => 200_000,
                m if m.contains("llama") => 131_072,
                m if m.contains("gemma") => 131_072,
                m if m.contains("mixtral") => 32_768,
                _ => 128_000,
            }
        };

        Ok(ModelSpec {
            name: model.to_string(),
            provider: self.config.provider_id.clone(),
            context_length: ctx,
            capabilities: ModelCapabilities {
                text: true,
                vision: model.contains("4o")
                    || model.contains("vision")
                    || model.contains("claude"),
                tool_calling: true,
                thinking: model.contains("o1") || model.contains("claude"),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<()> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages.iter().map(|m| {
                serde_json::json!({ "role": m.role, "content": m.content })
            }).collect::<Vec<_>>(),
            "stream": true,
        });

        if let Some(tools) = tools {
            body["tools"] = serde_json::to_value(tools)?;
        }

        let req = self.auth_headers(self.client.post(&url)).json(&body);
        let resp = req.send().await.context("Failed to send chat request")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            bail!("{} API error ({}): {}", self.config.name, status, error);
        }

        // Parse SSE stream
        let mut full_content = String::new();
        let body_text = resp.text().await.unwrap_or_default();

        for line in body_text.lines() {
            let line = line.trim();
            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            if data == "[DONE]" {
                break;
            }

            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(delta) = parsed
                    .get("choices")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|c| c.get("delta"))
                    .and_then(|d| d.get("content"))
                    .and_then(|c| c.as_str())
                {
                    full_content.push_str(delta);
                    let _ = tx.send(StreamEvent::Token(delta.to_string())).await;
                }
            }
        }

        let _ = tx
            .send(StreamEvent::Done {
                total_tokens: 0,
                prompt_tokens: 0,
                completion_tokens: 0,
            })
            .await;

        Ok(())
    }

    async fn chat_sync(
        &self,
        model: &str,
        messages: &[Message],
        temperature: Option<f64>,
    ) -> Result<String> {
        let url = format!("{}/chat/completions", self.config.base_url);

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages.iter().map(|m| {
                serde_json::json!({ "role": m.role, "content": m.content })
            }).collect::<Vec<_>>(),
            "stream": false,
        });

        if let Some(temp) = temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        let req = self.auth_headers(self.client.post(&url)).json(&body);
        let resp = req.send().await.context("Failed to send sync chat")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            bail!(
                "{} sync chat error ({}): {}",
                self.config.name,
                status,
                error
            );
        }

        let parsed: serde_json::Value = resp.json().await.context("Failed to parse response")?;
        Ok(parsed
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string())
    }

    async fn supports_modality(&self, model: &str, modality: Modality) -> Result<bool> {
        Ok(match modality {
            Modality::Text => true,
            Modality::Image => {
                model.contains("4o") || model.contains("vision") || model.contains("claude")
            }
            _ => false,
        })
    }

    async fn embed(&self, text: &str, model: &str) -> Result<Vec<f32>> {
        let url = format!("{}/embeddings", self.config.base_url);
        let body = serde_json::json!({
            "model": model,
            "input": text,
        });

        let req = self.auth_headers(self.client.post(&url)).json(&body);
        let resp = req
            .send()
            .await
            .context("Failed to send embedding request")?;

        if !resp.status().is_success() {
            bail!("{} embedding error: {}", self.config.name, resp.status());
        }

        let parsed: serde_json::Value = resp.json().await?;
        let embedding = parsed
            .get("data")
            .and_then(|d| d.as_array())
            .and_then(|arr| arr.first())
            .and_then(|e| e.get("embedding"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect()
            })
            .unwrap_or_default();

        Ok(embedding)
    }

    async fn health(&self) -> Result<ProviderStatus> {
        let start = Instant::now();
        let url = format!("{}/models", self.config.base_url);
        let req = self.auth_headers(self.client.get(&url));

        match req.send().await {
            Ok(resp) if resp.status().is_success() => Ok(ProviderStatus {
                available: true,
                latency_ms: Some(start.elapsed().as_millis() as u64),
                error: None,
                models_loaded: Vec::new(),
            }),
            Ok(resp) => Ok(ProviderStatus {
                available: false,
                latency_ms: Some(start.elapsed().as_millis() as u64),
                error: Some(format!(
                    "{} API returned {}",
                    self.config.name,
                    resp.status()
                )),
                models_loaded: Vec::new(),
            }),
            Err(e) => Ok(ProviderStatus {
                available: false,
                latency_ms: None,
                error: Some(format!("Cannot reach {}: {}", self.config.name, e)),
                models_loaded: Vec::new(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_id_openai() {
        let p = OpenAICompatProvider::openai("test-key");
        assert_eq!(p.id(), "openai");
        assert_eq!(p.display_name(), "OpenAI");
    }

    #[test]
    fn test_provider_id_groq() {
        let p = OpenAICompatProvider::groq("test-key");
        assert_eq!(p.id(), "groq");
        assert_eq!(p.display_name(), "Groq");
    }

    #[test]
    fn test_provider_id_claude() {
        let p = OpenAICompatProvider::anthropic("test-key");
        assert_eq!(p.id(), "claude");
        assert_eq!(p.display_name(), "Claude");
    }

    #[test]
    fn test_provider_id_openrouter() {
        let p = OpenAICompatProvider::openrouter("test-key");
        assert_eq!(p.id(), "openrouter");
        assert_eq!(p.display_name(), "OpenRouter");
    }

    #[test]
    fn test_default_config() {
        let config = OpenAICompatConfig::default();
        assert_eq!(config.base_url, "https://api.openai.com/v1");
        assert!(config.api_key.is_empty());
    }
}
