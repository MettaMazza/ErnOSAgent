//! llama-server provider — PRIMARY inference backend.
//!
//! Manages the llama-server process lifecycle and provides an OpenAI-compatible
//! API client for chat, model spec derivation, and streaming inference.
//! Supports control vectors for model steering.

use crate::model::spec::{ModelCapabilities, ModelSpec, ModelSummary, Modality};
use crate::provider::{Message, Provider, ProviderStatus, StreamEvent, ToolDefinition};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Instant;
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, Mutex};

/// llama-server provider configuration and state.
pub struct LlamaCppProvider {
    /// HTTP client for API calls.
    client: Client,
    /// Base URL (e.g. "http://localhost:8080").
    base_url: String,
    /// Path to llama-server binary.
    server_binary: String,
    /// Model GGUF path.
    model_path: String,
    /// Multimodal projector GGUF path.
    mmproj_path: String,
    /// GPU layers (-1 = full offload).
    n_gpu_layers: i32,
    /// Server port.
    port: u16,
    /// Extra CLI args.
    extra_args: Vec<String>,
    /// Running server process handle.
    process: Arc<Mutex<Option<Child>>>,
}

impl LlamaCppProvider {
    pub fn new(config: &crate::config::LlamaCppConfig) -> Self {
        Self {
            client: Client::new(),
            base_url: format!("http://localhost:{}", config.port),
            server_binary: config.server_binary.clone(),
            model_path: config.model_path.clone(),
            mmproj_path: config.mmproj_path.clone(),
            n_gpu_layers: config.n_gpu_layers,
            port: config.port,
            extra_args: config.extra_args.clone(),
            process: Arc::new(Mutex::new(None)),
        }
    }

    /// Build the full command-line arguments for llama-server.
    pub fn build_server_args(
        &self,
        steering_args: &[String],
    ) -> Vec<String> {
        let mut args = Vec::new();

        if !self.model_path.is_empty() {
            args.push("--model".to_string());
            args.push(self.model_path.clone());
        }

        if !self.mmproj_path.is_empty() {
            args.push("--mmproj".to_string());
            args.push(self.mmproj_path.clone());
        }

        args.push("--port".to_string());
        args.push(self.port.to_string());

        args.push("--n-gpu-layers".to_string());
        args.push(self.n_gpu_layers.to_string());

        // Control vectors
        args.extend(steering_args.iter().cloned());

        // Extra user-specified args
        args.extend(self.extra_args.iter().cloned());

        args
    }

    /// Start the llama-server process.
    pub async fn start_server(&self, steering_args: &[String]) -> Result<()> {
        let mut proc_guard = self.process.lock().await;

        // Kill existing process if running
        if let Some(ref mut child) = *proc_guard {
            tracing::info!("Stopping existing llama-server process");
            let _ = child.kill().await;
            let _ = child.wait().await;
        }

        let args = self.build_server_args(steering_args);

        tracing::info!(
            binary = %self.server_binary,
            args = ?args,
            "Starting llama-server"
        );

        let child = Command::new(&self.server_binary)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "Failed to start llama-server at '{}'. Is llama.cpp installed?",
                    self.server_binary
                )
            })?;

        *proc_guard = Some(child);

        // Wait for server to become healthy
        self.wait_for_health(30).await?;

        tracing::info!("llama-server started and healthy");
        Ok(())
    }

    /// Stop the llama-server process.
    pub async fn stop_server(&self) -> Result<()> {
        let mut proc_guard = self.process.lock().await;
        if let Some(ref mut child) = *proc_guard {
            tracing::info!("Stopping llama-server");
            child.kill().await.context("Failed to kill llama-server")?;
            child.wait().await.context("Failed to wait for llama-server exit")?;
            *proc_guard = None;
        }
        Ok(())
    }

    /// Restart the server (e.g. after model or steering vector changes).
    pub async fn restart_server(&self, steering_args: &[String]) -> Result<()> {
        self.stop_server().await?;
        self.start_server(steering_args).await
    }

    /// Wait for the server's /health endpoint to return OK.
    async fn wait_for_health(&self, timeout_secs: u64) -> Result<()> {
        let start = Instant::now();
        let health_url = format!("{}/health", self.base_url);

        loop {
            if start.elapsed().as_secs() > timeout_secs {
                bail!(
                    "llama-server failed to become healthy within {}s. \
                     Check that the model path is correct and the binary exists.",
                    timeout_secs
                );
            }

            match self.client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                _ => tokio::time::sleep(tokio::time::Duration::from_millis(500)).await,
            }
        }
    }

    /// Parse an OpenAI-compatible /v1/models response into ModelSummary entries.
    fn parse_models_response(
        &self,
        body: &serde_json::Value,
    ) -> Vec<ModelSummary> {
        let mut models = Vec::new();

        if let Some(data) = body.get("data").and_then(|d| d.as_array()) {
            for item in data {
                let name = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();

                models.push(ModelSummary {
                    name,
                    provider: "llamacpp".to_string(),
                    parameter_size: String::new(),
                    quantization_level: String::new(),
                    capabilities: ModelCapabilities {
                        text: true,
                        ..Default::default()
                    },
                    context_length: 0,
                });
            }
        }

        models
    }
}

#[async_trait]
impl Provider for LlamaCppProvider {
    fn id(&self) -> &str {
        "llamacpp"
    }

    fn display_name(&self) -> &str {
        "llama.cpp Server"
    }

    async fn list_models(&self) -> Result<Vec<ModelSummary>> {
        let url = format!("{}/v1/models", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("Failed to reach llama-server at {}", url))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse /v1/models response")?;

        Ok(self.parse_models_response(&body))
    }

    async fn get_model_spec(&self, model: &str) -> Result<ModelSpec> {
        // llama-server /v1/models returns basic info
        // For full spec, we also check /props if available
        let url = format!("{}/v1/models", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("Failed to reach llama-server at {}", url))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse /v1/models response")?;

        // Try to get props for context length
        let props = self
            .client
            .get(&format!("{}/props", self.base_url))
            .send()
            .await
            .ok()
            .and_then(|r| {
                futures::executor::block_on(r.json::<serde_json::Value>()).ok()
            });

        let context_length = props
            .as_ref()
            .and_then(|p| p.get("default_generation_settings"))
            .and_then(|s| s.get("n_ctx"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let model_name = body
            .get("data")
            .and_then(|d| d.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or(model)
            .to_string();

        Ok(ModelSpec {
            name: model_name,
            provider: "llamacpp".to_string(),
            context_length,
            capabilities: ModelCapabilities {
                text: true,
                vision: !self.mmproj_path.is_empty(),
                video: !self.mmproj_path.is_empty(),
                tool_calling: true,
                ..Default::default()
            },
            raw_info: body,
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
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": true,
        });

        if let Some(tools) = tools {
            body["tools"] = serde_json::to_value(tools)?;
        }

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| "Failed to send chat request to llama-server")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error_body = resp.text().await.unwrap_or_default();
            bail!(
                "llama-server returned error {}: {}",
                status,
                error_body
            );
        }

        // Parse SSE stream
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();

        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Stream read error from llama-server")?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete SSE lines
            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.is_empty() || line.starts_with(':') {
                    continue;
                }

                if let Some(data) = line.strip_prefix("data: ") {
                    if data.trim() == "[DONE]" {
                        let _ = tx
                            .send(StreamEvent::Done {
                                total_tokens: 0,
                                prompt_tokens: 0,
                                completion_tokens: 0,
                            })
                            .await;
                        return Ok(());
                    }

                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(choices) = parsed.get("choices").and_then(|c| c.as_array()) {
                            for choice in choices {
                                let delta = choice.get("delta").unwrap_or(choice);

                                // Content tokens
                                if let Some(content) = delta.get("content").and_then(|c| c.as_str())
                                {
                                    if !content.is_empty() {
                                        let _ = tx.send(StreamEvent::Token(content.to_string())).await;
                                    }
                                }

                                // Tool calls
                                if let Some(tool_calls) =
                                    delta.get("tool_calls").and_then(|t| t.as_array())
                                {
                                    for tc in tool_calls {
                                        if let Some(func) = tc.get("function") {
                                            let name = func
                                                .get("name")
                                                .and_then(|n| n.as_str())
                                                .unwrap_or("")
                                                .to_string();
                                            let args = func
                                                .get("arguments")
                                                .and_then(|a| a.as_str())
                                                .unwrap_or("{}")
                                                .to_string();
                                            let id = tc
                                                .get("id")
                                                .and_then(|i| i.as_str())
                                                .unwrap_or("")
                                                .to_string();

                                            if !name.is_empty() {
                                                let _ = tx
                                                    .send(StreamEvent::ToolCall {
                                                        id,
                                                        name,
                                                        arguments: args,
                                                    })
                                                    .await;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    async fn chat_sync(
        &self,
        model: &str,
        messages: &[Message],
        temperature: Option<f64>,
    ) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false,
        });

        if let Some(temp) = temperature {
            body["temperature"] = serde_json::json!(temp);
        }

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| "Failed to send sync chat request to llama-server")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error_body = resp.text().await.unwrap_or_default();
            bail!("llama-server sync chat error {}: {}", status, error_body);
        }

        let parsed: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse sync chat response")?;

        let content = parsed
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|arr| arr.first())
            .and_then(|choice| choice.get("message"))
            .and_then(|msg| msg.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        Ok(content)
    }

    async fn supports_modality(&self, _model: &str, modality: Modality) -> Result<bool> {
        match modality {
            Modality::Text => Ok(true),
            Modality::Image | Modality::Video => Ok(!self.mmproj_path.is_empty()),
            Modality::Audio => Ok(false), // Audio requires separate E2B/E4B instance
        }
    }

    async fn embed(&self, text: &str, _model: &str) -> Result<Vec<f32>> {
        let url = format!("{}/v1/embeddings", self.base_url);
        let body = serde_json::json!({
            "input": text,
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to send embedding request to llama-server")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error_body = resp.text().await.unwrap_or_default();
            bail!("llama-server embedding error {}: {}", status, error_body);
        }

        let parsed: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse embedding response")?;

        let embedding = parsed
            .get("data")
            .and_then(|d| d.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("embedding"))
            .and_then(|e| e.as_array())
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
        let url = format!("{}/health", self.base_url);

        match self.client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => Ok(ProviderStatus {
                available: true,
                latency_ms: Some(start.elapsed().as_millis() as u64),
                error: None,
                models_loaded: Vec::new(),
            }),
            Ok(resp) => Ok(ProviderStatus {
                available: false,
                latency_ms: Some(start.elapsed().as_millis() as u64),
                error: Some(format!("Health check returned {}", resp.status())),
                models_loaded: Vec::new(),
            }),
            Err(e) => Ok(ProviderStatus {
                available: false,
                latency_ms: None,
                error: Some(format!("Connection failed: {}", e)),
                models_loaded: Vec::new(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LlamaCppConfig;

    fn test_config() -> LlamaCppConfig {
        LlamaCppConfig {
            server_binary: "llama-server".to_string(),
            port: 8080,
            model_path: "/models/gemma4.gguf".to_string(),
            mmproj_path: "/models/gemma4.mmproj".to_string(),
            n_gpu_layers: -1,
            extra_args: Vec::new(),
        }
    }

    #[test]
    fn test_build_server_args_basic() {
        let provider = LlamaCppProvider::new(&test_config());
        let args = provider.build_server_args(&[]);

        assert!(args.contains(&"--model".to_string()));
        assert!(args.contains(&"/models/gemma4.gguf".to_string()));
        assert!(args.contains(&"--mmproj".to_string()));
        assert!(args.contains(&"/models/gemma4.mmproj".to_string()));
        assert!(args.contains(&"--port".to_string()));
        assert!(args.contains(&"8080".to_string()));
        assert!(args.contains(&"--n-gpu-layers".to_string()));
        assert!(args.contains(&"-1".to_string()));
    }

    #[test]
    fn test_build_server_args_with_steering() {
        let provider = LlamaCppProvider::new(&test_config());
        let steering = vec![
            "--control-vector-scaled".to_string(),
            "/vectors/honesty.gguf:1.5".to_string(),
            "--control-vector-layer-range".to_string(),
            "10".to_string(),
            "20".to_string(),
        ];
        let args = provider.build_server_args(&steering);

        assert!(args.contains(&"--control-vector-scaled".to_string()));
        assert!(args.contains(&"/vectors/honesty.gguf:1.5".to_string()));
    }

    #[test]
    fn test_build_server_args_no_mmproj() {
        let mut config = test_config();
        config.mmproj_path = String::new();
        let provider = LlamaCppProvider::new(&config);
        let args = provider.build_server_args(&[]);

        assert!(!args.contains(&"--mmproj".to_string()));
    }

    #[test]
    fn test_parse_models_response() {
        let provider = LlamaCppProvider::new(&test_config());
        let body = serde_json::json!({
            "data": [
                {"id": "gemma4:26b", "object": "model"},
                {"id": "llama3:8b", "object": "model"},
            ]
        });

        let models = provider.parse_models_response(&body);
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].name, "gemma4:26b");
        assert_eq!(models[1].name, "llama3:8b");
        assert_eq!(models[0].provider, "llamacpp");
    }

    #[test]
    fn test_parse_models_response_empty() {
        let provider = LlamaCppProvider::new(&test_config());
        let body = serde_json::json!({"data": []});
        let models = provider.parse_models_response(&body);
        assert!(models.is_empty());
    }

    #[test]
    fn test_provider_id() {
        let provider = LlamaCppProvider::new(&test_config());
        assert_eq!(provider.id(), "llamacpp");
        assert_eq!(provider.display_name(), "llama.cpp Server");
    }
}
