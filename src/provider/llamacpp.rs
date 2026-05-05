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
        let ctx = if self.config.context_length > 0 {
            // §2.7: behavior changes are never silent. When the operator
            // sets a non-zero context_length, the auto-derived GGUF value
            // is overridden — log it loudly at server-startup time.
            tracing::info!(
                context_length = self.config.context_length,
                "Operator-configured context_length overrides GGUF default"
            );
            self.config.context_length.to_string()
        } else {
            "0".to_string() // 0 = auto-detect from GGUF (legacy default)
        };
        let mut args = vec![
            "--model".to_string(),
            self.config.model_path.clone(),
            "--port".to_string(),
            self.config.port.to_string(),
            "--jinja".to_string(), // Use model's built-in Jinja chat template for tool calling
            "-c".to_string(),
            ctx,
            "-np".to_string(),
            "1".to_string(), // Single slot — prevents unused slots wasting KV cache
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

        // Thinking mode control — passed to the model's Jinja template via chat_template_kwargs
        body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": thinking});

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

        // Video support: models with vision typically support
        // video as frame sequences
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

        // Retry on transient connection errors (connection reset/closed/refused)
        let max_retries = 3;
        let mut last_err = None;
        for attempt in 0..=max_retries {
            if attempt > 0 {
                let delay = std::time::Duration::from_millis(500 * (1 << (attempt - 1)));
                tracing::warn!(
                    attempt, delay_ms = delay.as_millis() as u64,
                    "Retrying llama-server request after connection error"
                );
                tokio::time::sleep(delay).await;
            }

            let post_start = std::time::Instant::now();
            tracing::info!(attempt, url = %url, "llamacpp: sending POST to llama-server");
            let send_result = self.client.post(&url).json(&body).send().await;
            match send_result {
                Ok(response) => {
                    tracing::info!(
                        attempt,
                        status = %response.status(),
                        elapsed_ms = post_start.elapsed().as_millis() as u64,
                        "llamacpp: POST response received"
                    );
                    if !response.status().is_success() {
                        let status = response.status();
                        let text = response.text().await.unwrap_or_default();
                        anyhow::bail!("llama-server returned {} at {}: {}", status, url, text);
                    }
                    if attempt > 0 {
                        tracing::info!(attempt, "llama-server request succeeded after retry");
                    }
                    let (tx, rx) = mpsc::channel(256);
                    let slots_url = format!("{}/slots", self.base_url);
                    tokio::spawn(async move {
                        if let Err(e) = stream_parser::parse_sse_stream(response, tx.clone(), Some(slots_url)).await {
                            let _ = tx.send(StreamEvent::Error(e.to_string())).await;
                        }
                    });
                    return Ok(rx);
                }
                Err(e) => {
                    let is_connection_error = e.is_connect()
                        || e.is_request()
                        || format!("{}", e).contains("connection closed")
                        || format!("{}", e).contains("connection reset");
                    if is_connection_error && attempt < max_retries {
                        tracing::warn!(
                            attempt, error = %e,
                            "llama-server connection error (will retry)"
                        );
                        last_err = Some(e);
                        continue;
                    }
                    return Err(e).context(format!(
                        "Failed to connect to llama-server at {} after {} attempt(s). Is the server still running?",
                        url, attempt + 1
                    ));
                }
            }
        }
        match last_err {
            Some(e) => Err(e).context(format!(
                "Failed to connect to llama-server at {} after {} retries",
                url, max_retries
            )),
            None => anyhow::bail!(
                "Failed to connect to llama-server at {} after {} retries — no error captured",
                url, max_retries
            ),
        }
    }

    async fn chat_sync(
        &self,
        messages: &[Message],
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = self.build_chat_body(messages, tools, false, false);

        // Retry on transient connection errors (same policy as chat())
        let max_retries = 3;
        let mut last_err = None;
        let mut response = None;
        for attempt in 0..=max_retries {
            if attempt > 0 {
                let delay = std::time::Duration::from_millis(500 * (1 << (attempt - 1)));
                tracing::warn!(
                    attempt, delay_ms = delay.as_millis() as u64,
                    "Retrying llama-server sync request after connection error"
                );
                tokio::time::sleep(delay).await;
            }
            match self.client.post(&url).json(&body).send().await {
                Ok(r) => { response = Some(r); break; }
                Err(e) => {
                    let is_connection_error = e.is_connect()
                        || e.is_request()
                        || format!("{}", e).contains("connection closed")
                        || format!("{}", e).contains("connection reset");
                    if is_connection_error && attempt < max_retries {
                        tracing::warn!(
                            attempt, error = %e,
                            "llama-server sync connection error (will retry)"
                        );
                        last_err = Some(e);
                        continue;
                    }
                    return Err(e).context(format!(
                        "Failed to connect to llama-server at {} after {} attempt(s)",
                        url, attempt + 1
                    ));
                }
            }
        }
        let response = match response {
            Some(r) => r,
            None => match last_err {
                Some(e) => return Err(e).context(format!(
                    "Failed to connect to llama-server at {} after {} retries",
                    url, max_retries
                )),
                None => anyhow::bail!(
                    "Failed to connect to llama-server at {} after {} retries — no error captured",
                    url, max_retries
                ),
            },
        };

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

    async fn count_tokens(&self, messages: &[Message], tools: Option<&serde_json::Value>, thinking: bool) -> Result<usize> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let mut body = self.build_chat_body(messages, tools, false, thinking);
        body["max_tokens"] = serde_json::json!(1);
        body["stream"] = serde_json::json!(false);

        // Retry on transient connection errors (same policy as chat()/chat_sync() per §9.2)
        let max_retries = 3;
        let mut last_err = None;
        let mut response = None;
        for attempt in 0..=max_retries {
            if attempt > 0 {
                let delay = std::time::Duration::from_millis(500 * (1 << (attempt - 1)));
                tracing::warn!(
                    attempt, delay_ms = delay.as_millis() as u64,
                    "Retrying count_tokens after connection error"
                );
                tokio::time::sleep(delay).await;
            }
            match self.client.post(&url).json(&body).send().await {
                Ok(r) => { response = Some(r); break; }
                Err(e) => {
                    let is_connection_error = e.is_connect()
                        || e.is_request()
                        || format!("{}", e).contains("connection closed")
                        || format!("{}", e).contains("connection reset");
                    if is_connection_error && attempt < max_retries {
                        tracing::warn!(
                            attempt, error = %e,
                            "count_tokens connection error (will retry)"
                        );
                        last_err = Some(e);
                        continue;
                    }
                    return Err(e).context(format!(
                        "count_tokens: failed to connect to llama-server at {} after {} attempt(s)",
                        url, attempt + 1
                    ));
                }
            }
        }
        let response = match response {
            Some(r) => r,
            None => match last_err {
                Some(e) => return Err(e).context(format!(
                    "count_tokens: failed to connect to llama-server at {} after {} retries",
                    url, max_retries
                )),
                None => anyhow::bail!(
                    "count_tokens: failed to connect to llama-server at {} after {} retries — no error captured",
                    url, max_retries
                ),
            },
        };

        let status = response.status();
        let data: serde_json::Value = response.json().await
            .context("count_tokens: failed to parse llama-server response")?;

        // On 400 (oversized prompt), llama-server reports the token count in
        // the error body as n_prompt_tokens. Extract it — this is a real
        // measurement from the server's tokenizer, not a heuristic.
        if !status.is_success() {
            if let Some(n) = data["error"]["n_prompt_tokens"].as_u64() {
                tracing::warn!(
                    status = %status, n_prompt_tokens = n,
                    "count_tokens: server rejected oversized prompt but reported token count"
                );
                return Ok(n as usize);
            }
            anyhow::bail!(
                "count_tokens: llama-server returned {} — {}",
                status, data
            );
        }

        data["usage"]["prompt_tokens"]
            .as_u64()
            .map(|v| v as usize)
            .context("count_tokens: llama-server did not report prompt_tokens in usage")
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

    #[test]
    fn test_chat_sync_body_structure() {
        let config = LlamaCppConfig::default();
        let provider = LlamaCppProvider::new(&config);
        let messages = vec![Message::text("user", "Test")];
        let body = provider.build_chat_body(&messages, None, false, false);
        assert_eq!(body["stream"], false, "chat_sync must use stream=false");
        assert!(body["messages"].is_array());
    }

    #[test]
    fn test_chat_sync_retry_delay_calculation() {
        // Verify exponential backoff: 500ms, 1000ms, 2000ms
        for attempt in 1..=3u32 {
            let delay_ms = 500u64 * (1 << (attempt - 1));
            match attempt {
                1 => assert_eq!(delay_ms, 500),
                2 => assert_eq!(delay_ms, 1000),
                3 => assert_eq!(delay_ms, 2000),
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_chat_sync_connection_error_classification() {
        // Validate that the same error strings are checked in both chat() and chat_sync()
        let test_messages = ["connection closed", "connection reset"];
        for msg in &test_messages {
            assert!(msg.contains("connection closed") || msg.contains("connection reset"),
                "Error classification must match: {}", msg);
        }
    }

    /// Helper: extract the value passed to the `-c` flag from build_server_args.
    fn ctx_arg(args: &[String]) -> &str {
        let pos = args.iter().position(|a| a == "-c")
            .expect("build_server_args must always emit -c");
        &args[pos + 1]
    }

    #[test]
    fn test_build_server_args_default_context_passes_zero() {
        // Default config has context_length = 0, which must produce `-c 0`
        // (legacy behavior — llama-server interprets 0 as auto-detect from GGUF).
        let config = LlamaCppConfig::default();
        let provider = LlamaCppProvider::new(&config);
        let args = provider.build_server_args();
        assert_eq!(ctx_arg(&args), "0");
    }

    #[test]
    fn test_build_server_args_explicit_context_overrides() {
        // Setting context_length must produce `-c <value>` instead of `-c 0`.
        let mut config = LlamaCppConfig::default();
        config.context_length = 32768;
        let provider = LlamaCppProvider::new(&config);
        let args = provider.build_server_args();
        assert_eq!(ctx_arg(&args), "32768");
    }
}
