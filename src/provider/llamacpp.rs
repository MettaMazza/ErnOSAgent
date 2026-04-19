// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Primary provider — llama-server subprocess management + OpenAI-compatible API.

use crate::config::LlamaCppConfig;
use crate::model::ModelSpec;
use crate::provider::{Message, Provider, StreamEvent, stream_parser};
use anyhow::{Context, Result};
use async_trait::async_trait;
use tokio::sync::mpsc;

/// llama-server based provider — manages the server subprocess and
/// speaks the OpenAI-compatible chat completions API.
pub struct LlamaCppProvider {
    config: LlamaCppConfig,
    client: reqwest::Client,
    base_url: String,
}

impl LlamaCppProvider {
    pub fn new(config: &LlamaCppConfig) -> Self {
        Self {
            config: config.clone(),
            client: reqwest::Client::new(),
            base_url: format!("http://localhost:{}", config.port),
        }
    }

    /// Build the llama-server command-line arguments.
    pub fn build_server_args(&self) -> Vec<String> {
        let mut args = vec![
            "--model".to_string(),
            self.config.model_path.clone(),
            "--port".to_string(),
            self.config.port.to_string(),
            "--jinja".to_string(), // MANDATORY for Gemma 4 tool calling
            "-c".to_string(),
            "0".to_string(), // Auto-detect context from GGUF
            "-ngl".to_string(),
            self.config.n_gpu_layers.to_string(),
        ];

        // Multimodal projector for vision support
        if let Some(ref mmproj) = self.config.mmproj_path {
            args.push("--mmproj".to_string());
            args.push(mmproj.clone());
        }

        // LoRA adapter for incremental learning
        if let Some(ref lora) = self.config.lora_adapter {
            args.push("--lora".to_string());
            args.push(lora.clone());
        }

        args
    }

    /// Build the request body for chat completions.
    fn build_chat_body(
        &self,
        messages: &[Message],
        tools: Option<&serde_json::Value>,
        stream: bool,
        thinking: bool,
    ) -> serde_json::Value {
        let mut body = serde_json::json!({
            "messages": messages,
            "stream": stream,
            "max_tokens": -1,
        });

        if let Some(tools) = tools {
            body["tools"] = tools.clone();
        }

        // Thinking mode control
        if !thinking {
            // Disable thinking for observer/sync calls — suppress thinking channel entirely.
            // Temperature is NOT set here — auto-derived from model defaults (§2.1).
            body["reasoning_effort"] = serde_json::json!("none");
        }

        body
    }

    /// Discover model name from /v1/models endpoint.
    async fn discover_model_name(&self) -> Result<String> {
        let url = format!("{}/v1/models", self.base_url);
        let resp = self.client.get(&url).send().await
            .context("Failed to reach /v1/models — is llama-server running?")?;
        let body: serde_json::Value = resp.json().await
            .context("Invalid JSON from /v1/models")?;
        let name = body["data"][0]["id"]
            .as_str()
            .context("Provider did not report model name in /v1/models")?;
        Ok(name.to_string())
    }

    /// Fetch full props from /props endpoint (context length, capabilities).
    async fn fetch_props(&self) -> Result<serde_json::Value> {
        let url = format!("{}/props", self.base_url);
        let resp = self.client.get(&url).send().await
            .context("Failed to reach /props — is llama-server running?")?;
        let body: serde_json::Value = resp.json().await
            .context("Invalid JSON from /props")?;
        Ok(body)
    }
}

#[async_trait]
impl Provider for LlamaCppProvider {
    fn id(&self) -> &str {
        "llamacpp"
    }

    fn display_name(&self) -> &str {
        "llama.cpp (local)"
    }

    async fn get_model_spec(&self) -> Result<ModelSpec> {
        let name = self.discover_model_name().await?;
        let props = self.fetch_props().await?;

        // Context length: auto-derived, NEVER hardcoded (Rule 2.1)
        let context_length = props["default_generation_settings"]["n_ctx"]
            .as_u64()
            .map(|v| v as usize)
            .context(
                "Provider did not report context length (n_ctx). \
                 Cannot proceed — Rule 2.1 forbids hardcoded fallbacks. \
                 Ensure llama-server is running with a loaded model."
            )?;

        // Multimodal capabilities: auto-discovered from props (Rule 5, 7.1)
        let has_mmproj = props["default_generation_settings"]["mmproj"]
            .as_str()
            .map_or(false, |s| !s.is_empty());
        let supports_vision = has_mmproj
            || props["multimodal"].as_bool().unwrap_or(false)
            || self.config.mmproj_path.is_some(); // Final fallback: config presence

        // Video support: same model that supports vision typically supports
        // video as frame sequences (Gemma 4, LLaVA, etc.)
        let supports_video = supports_vision;

        // Audio: not supported by current model architectures via llama-server
        let supports_audio = false;

        tracing::info!(
            name = %name, context_length, supports_vision,
            supports_video, supports_audio,
            "Model capabilities auto-derived from provider"
        );

        Ok(ModelSpec {
            name,
            context_length,
            supports_vision,
            supports_video,
            supports_audio,
            supports_tool_calling: true,
            supports_thinking: true,
            embedding_dimensions: 0,
        })
    }

    async fn chat(
        &self,
        messages: &[Message],
        tools: Option<&serde_json::Value>,
        thinking: bool,
    ) -> Result<mpsc::Receiver<StreamEvent>> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = self.build_chat_body(messages, tools, true, thinking);

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context(format!(
                "Failed to connect to llama-server at {}. Is the server still running?",
                url
            ))?;

        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            anyhow::bail!("llama-server returned {} at {}: {}", status, url, text);
        }

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
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = self.build_chat_body(messages, tools, false, false);

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to connect to llama-server for sync chat")?;

        let status = response.status();
        let body: serde_json::Value = response.json().await
            .context("Failed to parse llama-server sync response")?;

        if !status.is_success() {
            anyhow::bail!("llama-server sync returned {}: {:?}", status, body);
        }

        let content = body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(content)
    }

    async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let url = format!(
            "http://localhost:{}/v1/embeddings",
            self.config.embedding_port
        );

        let body = serde_json::json!({
            "input": text,
            "model": "embedding"
        });

        let response = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to connect to embedding server")?;

        let result: serde_json::Value = response.json().await
            .context("Failed to parse embedding response")?;

        let embedding = result["data"][0]["embedding"]
            .as_array()
            .context("No embedding in response")?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        Ok(embedding)
    }

    async fn health(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        self.client
            .get(&url)
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_server_args_includes_jinja() {
        let config = LlamaCppConfig::default();
        let provider = LlamaCppProvider::new(&config);
        let args = provider.build_server_args();
        assert!(args.contains(&"--jinja".to_string()));
    }

    #[test]
    fn test_build_server_args_includes_mmproj() {
        let config = LlamaCppConfig::default();
        let provider = LlamaCppProvider::new(&config);
        let args = provider.build_server_args();
        assert!(args.contains(&"--mmproj".to_string()));
    }

    #[test]
    fn test_build_chat_body_stream() {
        let config = LlamaCppConfig::default();
        let provider = LlamaCppProvider::new(&config);
        let messages = vec![Message::text("user", "Hello")];
        let body = provider.build_chat_body(&messages, None, true, true);
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn test_build_chat_body_with_tools() {
        let config = LlamaCppConfig::default();
        let provider = LlamaCppProvider::new(&config);
        let messages = vec![Message::text("user", "Hello")];
        let tools = serde_json::json!([{"type": "function", "function": {"name": "test"}}]);
        let body = provider.build_chat_body(&messages, Some(&tools), true, true);
        assert!(body["tools"].is_array());
    }
}
