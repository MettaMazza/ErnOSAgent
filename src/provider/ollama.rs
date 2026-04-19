// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Ollama provider — secondary provider via OpenAI-compatible API.

use crate::config::OllamaConfig;
use crate::model::ModelSpec;
use crate::provider::{Message, Provider, StreamEvent, stream_parser};
use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::sync::mpsc;

pub struct OllamaProvider {
    config: OllamaConfig,
    client: reqwest::Client,
}

impl OllamaProvider {
    pub fn new(config: &OllamaConfig) -> Self {
        Self {
            config: config.clone(),
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn id(&self) -> &str { "ollama" }
    fn display_name(&self) -> &str { "Ollama" }

    async fn get_model_spec(&self) -> Result<ModelSpec> {
        let url = format!("{}/api/show", self.config.base_url);
        let body = serde_json::json!({ "name": self.config.model });

        let resp = self.client.post(&url).json(&body).send().await
            .context("Failed to connect to Ollama")?;
        let data: serde_json::Value = resp.json().await
            .context("Failed to parse Ollama model info")?;

        // Auto-derive context length — NEVER hardcode (Rule 2.1)
        let context_length = data["model_info"]
            .as_object()
            .and_then(|info| {
                info.iter()
                    .find(|(k, _)| k.contains("context_length"))
                    .and_then(|(_, v)| v.as_u64())
            })
            .map(|v| v as usize)
            .context(
                "Ollama did not report context_length in model_info. \
                 Rule 2.1 forbids hardcoded fallbacks."
            )?;

        // Auto-discover multimodal capabilities from model metadata
        let families = data["details"]["families"].as_array();
        let has_vision = families
            .map(|f| f.iter().any(|v| {
                v.as_str().map_or(false, |s| s.contains("clip") || s.contains("vision"))
            }))
            .unwrap_or(false);

        Ok(ModelSpec {
            name: self.config.model.clone(),
            context_length,
            supports_vision: has_vision,
            supports_video: has_vision, // Vision models support video as frames
            supports_audio: false,
            supports_tool_calling: true,
            supports_thinking: true,
            embedding_dimensions: 0,
        })
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&serde_json::Value>,
        _thinking: bool,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let url = format!("{}/v1/chat/completions", self.config.base_url);
        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "stream": true,
        });
        if let Some(tools) = tools {
            body["tools"] = tools.clone();
        }

        let response = self.client.post(&url).json(&body).send().await
            .context("Failed to connect to Ollama")?;

        let (tx, rx) = mpsc::channel(256);
        tokio::spawn(async move {
            if let Err(e) = stream_parser::parse_sse_stream(response, tx.clone()).await {
                let _ = tx.send(StreamEvent::Error(e.to_string())).await;
            }
        });

        Ok(rx)
    }

    async fn chat_sync(
        &self,
        messages: &[Message],
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.config.base_url);
        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "stream": false,
        });
        if let Some(tools) = tools {
            body["tools"] = tools.clone();
        }

        let resp = self.client.post(&url).json(&body).send().await
            .context("Failed to connect to Ollama for sync")?;
        let data: serde_json::Value = resp.json().await?;

        Ok(data["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/v1/embeddings", self.config.base_url);
        let body = serde_json::json!({
            "model": self.config.model,
            "input": text,
        });

        let resp = self.client.post(&url).json(&body).send().await?;
        let data: serde_json::Value = resp.json().await?;

        Ok(data["data"][0]["embedding"]
            .as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect())
    }

    async fn health(&self) -> bool {
        let url = format!("{}/api/tags", self.config.base_url);
        self.client.get(&url).send().await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_id() {
        let config = OllamaConfig::default();
        let provider = OllamaProvider::new(&config);
        assert_eq!(provider.id(), "ollama");
    }
}
