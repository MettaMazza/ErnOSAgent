// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
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
    pub(crate) client: Client,
    /// Base URL (e.g. "http://localhost:8080").
    base_url: String,
    /// Path to llama-server binary.
    pub(crate) server_binary: String,
    /// Model GGUF path.
    pub(crate) model_path: String,
    /// Multimodal projector GGUF path.
    mmproj_path: String,
    /// GPU layers (-1 = full offload).
    pub(crate) n_gpu_layers: i32,
    /// Server port.
    port: u16,
    /// Extra CLI args.
    extra_args: Vec<String>,
    /// Running server process handle.
    process: Arc<Mutex<Option<Child>>>,
    /// Dedicated embedding server URL (e.g. "http://localhost:8081").
    pub(crate) embedding_url: Option<String>,
    /// Embedding model GGUF path.
    pub(crate) embedding_model_path: String,
    /// Embedding server port.
    pub(crate) embedding_port: u16,
    /// Running embedding server process handle.
    pub(crate) embedding_process: Arc<Mutex<Option<Child>>>,
}

impl LlamaCppProvider {
    pub fn new(config: &crate::config::LlamaCppConfig) -> Self {
        let embedding_url = if !config.embedding_model_path.is_empty() {
            Some(format!("http://localhost:{}", config.embedding_port))
        } else {
            None
        };

        Self {
            client: Client::builder()
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| Client::new()),
            base_url: format!("http://localhost:{}", config.port),
            server_binary: config.server_binary.clone(),
            model_path: config.model_path.clone(),
            mmproj_path: config.mmproj_path.clone(),
            n_gpu_layers: config.n_gpu_layers,
            port: config.port,
            extra_args: config.extra_args.clone(),
            process: Arc::new(Mutex::new(None)),
            embedding_url,
            embedding_model_path: config.embedding_model_path.clone(),
            embedding_port: config.embedding_port,
            embedding_process: Arc::new(Mutex::new(None)),
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

        // Delegate context detection to native GGUF limits
        args.push("-c".to_string());
        args.push("0".to_string());

        // Enable embedding endpoint on main server (needed for SAE activation extraction)
        args.push("--embeddings".to_string());

        // Control vectors
        args.extend(steering_args.iter().cloned());

        // Extra user-specified args
        args.extend(self.extra_args.iter().cloned());

        args
    }

    /// Start the llama-server process.
    ///
    /// Performs system-level cleanup first: kills ALL existing llama-server
    /// processes (not just our own child handle) to prevent orphaned servers
    /// from prior runs from holding the port and causing deadlocks.
    pub async fn start_server(&self, steering_args: &[String]) -> Result<()> {
        let mut proc_guard = self.process.lock().await;

        // Kill our own child if running
        if let Some(ref mut child) = *proc_guard {
            tracing::info!("Stopping existing llama-server child process");
            let _ = child.kill().await;
            let _ = child.wait().await;
            *proc_guard = None;
        }

        // System-level cleanup: kill ALL orphaned llama-server processes.
        // A prior ernosagent crash or restart can leave stale llama-server
        // processes bound to our port, causing the Observer to deadlock on
        // chat_sync when the stale process holds the single inference slot.
        Self::kill_orphaned_servers().await;

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

    /// Kill ALL llama-server processes on the system, not just our child.
    ///
    /// This handles orphaned servers from prior ernosagent runs that crashed
    /// or were killed without cleanup. A stale llama-server holding port 8080
    /// with --parallel 1 will deadlock the Observer's chat_sync call if the
    /// main inference is using the only slot.
    async fn kill_orphaned_servers() {
        // pkill sends SIGTERM to all matching processes
        let output = tokio::process::Command::new("pkill")
            .args(["-f", "llama-server"])
            .output()
            .await;

        match output {
            Ok(o) if o.status.success() => {
                tracing::warn!("Killed orphaned llama-server processes");
                // Wait for port release
                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            }
            Ok(_) => {
                tracing::debug!("No orphaned llama-server processes found");
            }
            Err(e) => {
                tracing::warn!(error = %e, "pkill command failed — orphaned servers may persist");
            }
        }

        // Also kill any stale ernosagent processes (not ourselves)
        let our_pid = std::process::id();
        let output = tokio::process::Command::new("pgrep")
            .args(["-f", "ernosagent"])
            .output()
            .await;

        if let Ok(o) = output {
            let pids = String::from_utf8_lossy(&o.stdout);
            for line in pids.lines() {
                if let Ok(pid) = line.trim().parse::<u32>() {
                    if pid != our_pid {
                        tracing::warn!(stale_pid = pid, "Killing stale ernosagent process");
                        let _ = tokio::process::Command::new("kill")
                            .args(["-9", &pid.to_string()])
                            .output()
                            .await;
                    }
                }
            }
        }
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
        let props = match self
            .client
            .get(&format!("{}/props", self.base_url))
            .send()
            .await
        {
            Ok(resp) => resp.json::<serde_json::Value>().await.ok(),
            Err(_) => None,
        };

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

        // Transform messages: if a message has images, convert to OpenAI multipart
        // content format (array of {type: "text"} and {type: "image_url"} parts).
        let api_messages: Vec<serde_json::Value> = messages.iter().map(|msg| {
            // Filter images: only keep valid base64 or data URIs.
            // Raw HTTP URLs (from old Discord sessions) are invalid for llama-server
            // and cause 400 errors. Skip them.
            let valid_images: Vec<&String> = msg.images.iter().filter(|img| {
                if img.starts_with("http://") || img.starts_with("https://") {
                    let preview: String = img.chars().take(80).collect();
                    tracing::warn!("Skipping raw URL in image field (not base64-encoded): {}...", preview);
                    false
                } else {
                    true
                }
            }).collect();

            if valid_images.is_empty() {
                serde_json::json!({
                    "role": msg.role,
                    "content": msg.content,
                })
            } else {
                let mut parts = vec![serde_json::json!({
                    "type": "text",
                    "text": msg.content,
                })];
                for img_b64 in &valid_images {
                    // If it already has a data: prefix, use as-is; otherwise add one
                    let url = if img_b64.starts_with("data:") {
                        (*img_b64).clone()
                    } else {
                        format!("data:image/png;base64,{}", img_b64)
                    };
                    parts.push(serde_json::json!({
                        "type": "image_url",
                        "image_url": { "url": url }
                    }));
                }
                serde_json::json!({
                    "role": msg.role,
                    "content": parts,
                })
            }
        }).collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": api_messages,
            "stream": true,
        });

        if let Some(tools) = tools {
            body["tools"] = serde_json::to_value(tools)?;
            body["parallel_tool_calls"] = serde_json::json!(true);
        }

        let resp = match self.client.post(&url).json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = ?e, "llama-server request failed — retrying in 500ms");
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                self.client
                    .post(&url)
                    .json(&body)
                    .send()
                    .await
                    .with_context(|| format!("Failed to send chat request to llama-server (retry failed, original: {})", e))?
            }
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let error_body = resp.text().await.unwrap_or_default();
            bail!(
                "llama-server returned error {}: {}",
                status,
                error_body
            );
        }

        // Delegate SSE stream parsing to shared parser
        crate::provider::stream_parser::parse_sse_stream(resp, &tx).await?;

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

        // Disable thinking for sync calls (used by Observer) to avoid
        // wasting tokens on silent reasoning chains.
        body["chat_template_kwargs"] = serde_json::json!({"enable_thinking": false});

        let resp = match self.client.post(&url).json(&body).send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(error = ?e, "llama-server request failed (sync) — retrying in 500ms");
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                self.client
                    .post(&url)
                    .json(&body)
                    .send()
                    .await
                    .with_context(|| format!("Failed to send sync chat request to llama-server (retry failed, original: {})", e))?
            }
        };

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
        self.embed_via_server(text).await
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
#[path = "llamacpp_tests.rs"]
mod tests;

