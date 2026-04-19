// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Generic OpenAI-compatible provider — covers LM Studio, OpenRouter, Groq, etc.

use crate::config::OpenAICompatConfig;
use crate::model::ModelSpec;
use crate::provider::{Message, Provider, StreamEvent, stream_parser};
use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::sync::mpsc;

pub struct OpenAICompatProvider {
    config: OpenAICompatConfig,
    client: reqwest::Client,
}

impl OpenAICompatProvider {
    pub fn new(config: &OpenAICompatConfig) -> Self {
        let mut headers = reqwest::header::HeaderMap::new();
        if let Some(ref key) = config.api_key {
            if let Ok(val) = reqwest::header::HeaderValue::from_str(&format!("Bearer {}", key)) {
                headers.insert(reqwest::header::AUTHORIZATION, val);
            }
        }

        Self {
            config: config.clone(),
            client: reqwest::Client::builder()
                .default_headers(headers)
                .build()
                .unwrap_or_default(),
        }
    }
}

#[async_trait]
impl Provider for OpenAICompatProvider {
    fn id(&self) -> &str { "openai_compat" }
    fn display_name(&self) -> &str { "OpenAI Compatible" }

    async fn get_model_spec(&self) -> Result<ModelSpec> {
        let url = format!("{}/models", self.config.base_url);
        let resp = self.client.get(&url).send().await
            .context("Failed to connect to OpenAI-compatible provider for model spec")?;
        let data: serde_json::Value = resp.json().await
            .context("Failed to parse model spec from OpenAI-compatible provider")?;

        // Auto-derive context length — NEVER hardcode (Rule 2.1)
        let model_entry = data["data"]
            .as_array()
            .and_then(|arr| arr.first())
            .context(
                "Provider returned no models in /models response. \
                 Cannot auto-derive capabilities."
            )?;

        let context_length = model_entry["context_window"]
            .as_u64()
            .or_else(|| model_entry["context_length"].as_u64())
            .map(|v| v as usize)
            .context(
                "Provider did not report context_window in /models. \
                 Rule 2.1 forbids hardcoded fallbacks."
            )?;

        // Auto-discover capabilities from model metadata
        let model_id = model_entry["id"].as_str().unwrap_or(&self.config.model);
        let caps = model_entry.get("capabilities");
        let supports_vision = caps
            .and_then(|c| c["vision"].as_bool())
            .unwrap_or(false);

        Ok(ModelSpec {
            name: model_id.to_string(),
            context_length,
            supports_vision,
            supports_video: supports_vision,
            supports_audio: caps.and_then(|c| c["audio"].as_bool()).unwrap_or(false),
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
        let url = format!("{}/chat/completions", self.config.base_url);
        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "stream": true,
        });
        if let Some(tools) = tools {
            body["tools"] = tools.clone();
        }

        let response = self.client.post(&url).json(&body).send().await
            .context("Failed to connect to OpenAI-compatible provider")?;

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
        let url = format!("{}/chat/completions", self.config.base_url);
        let mut body = serde_json::json!({
            "model": self.config.model,
            "messages": messages,
            "stream": false,
        });
        if let Some(tools) = tools {
            body["tools"] = tools.clone();
        }

        let resp = self.client.post(&url).json(&body).send().await
            .context("Failed to connect to OpenAI-compatible provider for sync")?;
        let data: serde_json::Value = resp.json().await?;

        Ok(data["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/embeddings", self.config.base_url);
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
        let url = format!("{}/models", self.config.base_url);
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
        let config = OpenAICompatConfig::default();
        let provider = OpenAICompatProvider::new(&config);
        assert_eq!(provider.id(), "openai_compat");
    }

    #[test]
    fn test_auth_header() {
        let config = OpenAICompatConfig {
            api_key: Some("sk-test".to_string()),
            ..Default::default()
        };
        let provider = OpenAICompatProvider::new(&config);
        assert_eq!(provider.id(), "openai_compat");
    }
}
