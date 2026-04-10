// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Provider trait implementation for Ollama.

use super::OllamaProvider;
use crate::model::spec::{ModelCapabilities, ModelSpec, ModelSummary, Modality};
use crate::provider::{Message, Provider, ProviderStatus, StreamEvent, ToolDefinition};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use std::time::Instant;
use tokio::sync::mpsc;

#[async_trait]
impl Provider for OllamaProvider {
    fn id(&self) -> &str {
        "ollama"
    }

    fn display_name(&self) -> &str {
        "Ollama"
    }

    async fn list_models(&self) -> Result<Vec<ModelSummary>> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.client.get(&url).send().await
            .with_context(|| format!("Failed to connect to Ollama at {}", url))?;

        let body: serde_json::Value = resp.json().await
            .context("Failed to parse /api/tags response")?;

        let models = body.get("models")
            .and_then(|m| m.as_array())
            .map(|arr| arr.iter().map(parse_model_summary).collect())
            .unwrap_or_default();

        Ok(models)
    }

    async fn get_model_spec(&self, model: &str) -> Result<ModelSpec> {
        let url = format!("{}/api/show", self.base_url);
        let body = serde_json::json!({"name": model});

        let resp = self.client.post(&url).json(&body).send().await
            .with_context(|| format!("Failed to query Ollama /api/show for '{}'", model))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            bail!("Ollama /api/show returned {} for model '{}': {}", status, model, error);
        }

        let parsed: serde_json::Value = resp.json().await
            .context("Failed to parse /api/show response")?;

        Ok(self.parse_show_response(model, &parsed))
    }

    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<()> {
        let url = format!("{}/api/chat", self.base_url);

        let ollama_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                let mut msg = serde_json::json!({"role": m.role, "content": m.content});
                if !m.images.is_empty() { msg["images"] = serde_json::json!(m.images); }
                msg
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model, "messages": ollama_messages,
            "stream": true, "keep_alive": self.keep_alive,
        });

        if let Some(tools) = tools {
            body["tools"] = serde_json::to_value(tools)?;
        }

        let resp = self.client.post(&url).json(&body).send().await
            .with_context(|| "Failed to send chat request to Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            bail!("Ollama chat error {}: {}", status, error);
        }

        stream_chat_response(resp, tx).await
    }

    async fn chat_sync(
        &self,
        model: &str,
        messages: &[Message],
        temperature: Option<f64>,
    ) -> Result<String> {
        let url = format!("{}/api/chat", self.base_url);

        let ollama_messages: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
            .collect();

        let mut body = serde_json::json!({
            "model": model, "messages": ollama_messages,
            "stream": false, "keep_alive": self.keep_alive,
            "think": false,
        });

        if let Some(temp) = temperature {
            body["options"] = serde_json::json!({"temperature": temp});
        }

        let resp = self.client.post(&url).json(&body).send().await
            .context("Failed to send sync chat to Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            bail!("Ollama sync chat error {}: {}", status, error);
        }

        let parsed: serde_json::Value = resp.json().await
            .context("Failed to parse Ollama sync response")?;

        Ok(parsed.get("message").and_then(|m| m.get("content"))
            .and_then(|c| c.as_str()).unwrap_or("").to_string())
    }

    async fn supports_modality(&self, model: &str, modality: Modality) -> Result<bool> {
        match modality {
            Modality::Text => Ok(true),
            Modality::Audio => Ok(false),
            Modality::Image | Modality::Video => {
                let spec = self.get_model_spec(model).await?;
                Ok(spec.capabilities.vision)
            }
        }
    }

    async fn embed(&self, text: &str, model: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url);
        let body = serde_json::json!({"model": model, "prompt": text});

        let resp = self.client.post(&url).json(&body).send().await
            .context("Failed to send embedding request to Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            bail!("Ollama embedding error {}: {}", status, error);
        }

        let parsed: serde_json::Value = resp.json().await
            .context("Failed to parse Ollama embedding response")?;

        Ok(parsed.get("embedding")
            .and_then(|e| e.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
            .unwrap_or_default())
    }

    async fn health(&self) -> Result<ProviderStatus> {
        let start = Instant::now();
        match self.client.get(&self.base_url).send().await {
            Ok(resp) if resp.status().is_success() => Ok(ProviderStatus {
                available: true,
                latency_ms: Some(start.elapsed().as_millis() as u64),
                error: None,
                models_loaded: Vec::new(),
            }),
            Ok(resp) => Ok(ProviderStatus {
                available: false,
                latency_ms: Some(start.elapsed().as_millis() as u64),
                error: Some(format!("Ollama returned {}", resp.status())),
                models_loaded: Vec::new(),
            }),
            Err(e) => Ok(ProviderStatus {
                available: false,
                latency_ms: None,
                error: Some(format!("Cannot connect to Ollama: {}", e)),
                models_loaded: Vec::new(),
            }),
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────

fn parse_model_summary(item: &serde_json::Value) -> ModelSummary {
    let name = item.get("name").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
    let details = item.get("details").cloned().unwrap_or_default();
    let param_size = details.get("parameter_size").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let quant = details.get("quantization_level").and_then(|v| v.as_str()).unwrap_or("").to_string();
    let families: Vec<String> = details.get("families")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let has_vision = families.iter().any(|f| f == "clip" || f == "mllama");

    ModelSummary {
        name,
        provider: "ollama".to_string(),
        parameter_size: param_size,
        quantization_level: quant,
        capabilities: ModelCapabilities {
            text: true, vision: has_vision, video: has_vision, ..Default::default()
        },
        context_length: 0,
    }
}

async fn stream_chat_response(
    resp: reqwest::Response,
    tx: mpsc::Sender<StreamEvent>,
) -> Result<()> {
    let mut stream = resp.bytes_stream();
    let mut buffer = String::new();

    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Stream read error from Ollama")?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));

        while let Some(line_end) = buffer.find('\n') {
            let line = buffer[..line_end].trim().to_string();
            buffer = buffer[line_end + 1..].to_string();

            if line.is_empty() { continue; }

            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&line) {
                if parsed.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                    let total = parsed.get("eval_count").and_then(|v| v.as_u64()).unwrap_or(0);
                    let prompt = parsed.get("prompt_eval_count").and_then(|v| v.as_u64()).unwrap_or(0);
                    let _ = tx.send(StreamEvent::Done {
                        total_tokens: prompt + total, prompt_tokens: prompt, completion_tokens: total,
                    }).await;
                    return Ok(());
                }

                if let Some(msg) = parsed.get("message") {
                    if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                        if !content.is_empty() {
                            let _ = tx.send(StreamEvent::Token(content.to_string())).await;
                        }
                    }
                    parse_tool_calls(msg, &tx).await;
                }
            }
        }
    }
    Ok(())
}

async fn parse_tool_calls(msg: &serde_json::Value, tx: &mpsc::Sender<StreamEvent>) {
    if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tool_calls {
            if let Some(func) = tc.get("function") {
                let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("").to_string();
                let args = func.get("arguments").map(|a| a.to_string()).unwrap_or_else(|| "{}".to_string());
                if !name.is_empty() {
                    let _ = tx.send(StreamEvent::ToolCall {
                        id: uuid::Uuid::new_v4().to_string(), name, arguments: args,
                    }).await;
                }
            }
        }
    }
}
