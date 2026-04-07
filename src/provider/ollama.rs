//! Ollama provider — SECONDARY. Model management and embeddings.

use crate::model::spec::{ModelCapabilities, ModelSpec, ModelSummary, Modality};
use crate::provider::{Message, Provider, ProviderStatus, StreamEvent, ToolDefinition};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use std::time::Instant;
use tokio::sync::mpsc;

pub struct OllamaProvider {
    client: Client,
    base_url: String,
    keep_alive: i64,
}

impl OllamaProvider {
    pub fn new(config: &crate::config::OllamaConfig) -> Self {
        let base_url = config.host.trim_end_matches('/').to_string();
        Self {
            client: Client::new(),
            base_url,
            keep_alive: config.keep_alive,
        }
    }

    /// Parse the /api/show response into a ModelSpec.
    fn parse_show_response(&self, model_name: &str, body: &serde_json::Value) -> ModelSpec {
        let details = body.get("details").cloned().unwrap_or_default();
        let model_info = body.get("model_info").cloned().unwrap_or_default();

        let family = details
            .get("family")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let families: Vec<String> = details
            .get("families")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        let parameter_size = details
            .get("parameter_size")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let quantization_level = details
            .get("quantization_level")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let format = details
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        // Context length from model_info
        let context_length = model_info
            .get("general.context_length")
            .and_then(|v| v.as_u64())
            .or_else(|| {
                // Try parameters string
                body.get("parameters")
                    .and_then(|p| p.as_str())
                    .and_then(|params| {
                        for line in params.lines() {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() >= 2 && parts[0] == "num_ctx" {
                                return parts[1].parse().ok();
                            }
                        }
                        None
                    })
            })
            .unwrap_or(0);

        // Parse capabilities from families
        let has_vision = families.iter().any(|f| f == "clip" || f == "mllama");

        // Parse default parameters
        let mut temperature = 0.0_f64;
        let mut top_k = 0_u64;
        let mut top_p = 0.0_f64;

        if let Some(params) = body.get("parameters").and_then(|p| p.as_str()) {
            for line in params.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    match parts[0] {
                        "temperature" => temperature = parts[1].parse().unwrap_or(0.0),
                        "top_k" => top_k = parts[1].parse().unwrap_or(0),
                        "top_p" => top_p = parts[1].parse().unwrap_or(0.0),
                        _ => {}
                    }
                }
            }
        }

        let template = body
            .get("template")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        ModelSpec {
            name: model_name.to_string(),
            provider: "ollama".to_string(),
            family,
            families,
            parameter_size,
            parameter_count: 0,
            quantization_level,
            format,
            context_length,
            default_temperature: temperature,
            default_top_k: top_k,
            default_top_p: top_p,
            capabilities: ModelCapabilities {
                text: true,
                vision: has_vision,
                audio: false, // Ollama does not support audio input
                video: has_vision, // Video = frame extraction + vision
                tool_calling: true,
                thinking: false,
            },
            template,
            raw_info: body.clone(),
        }
    }
}

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
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("Failed to connect to Ollama at {}", url))?;

        let body: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse /api/tags response")?;

        let models = body
            .get("models")
            .and_then(|m| m.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|item| {
                        let name = item
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_string();
                        let details = item.get("details").cloned().unwrap_or_default();
                        let param_size = details
                            .get("parameter_size")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let quant = details
                            .get("quantization_level")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let families: Vec<String> = details
                            .get("families")
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
                                text: true,
                                vision: has_vision,
                                video: has_vision,
                                ..Default::default()
                            },
                            context_length: 0,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(models)
    }

    async fn get_model_spec(&self, model: &str) -> Result<ModelSpec> {
        let url = format!("{}/api/show", self.base_url);
        let body = serde_json::json!({"name": model});

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .with_context(|| format!("Failed to query Ollama /api/show for '{}'", model))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            bail!(
                "Ollama /api/show returned {} for model '{}': {}",
                status,
                model,
                error
            );
        }

        let parsed: serde_json::Value = resp
            .json()
            .await
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
                let mut msg = serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                });
                if !m.images.is_empty() {
                    msg["images"] = serde_json::json!(m.images);
                }
                msg
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": ollama_messages,
            "stream": true,
            "keep_alive": self.keep_alive,
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
            .with_context(|| "Failed to send chat request to Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            bail!("Ollama chat error {}: {}", status, error);
        }

        // Parse NDJSON stream
        let mut stream = resp.bytes_stream();
        let mut buffer = String::new();

        use futures::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Stream read error from Ollama")?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(line_end) = buffer.find('\n') {
                let line = buffer[..line_end].trim().to_string();
                buffer = buffer[line_end + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&line) {
                    // Check for done
                    if parsed.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                        let total = parsed
                            .get("eval_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let prompt = parsed
                            .get("prompt_eval_count")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0);
                        let _ = tx
                            .send(StreamEvent::Done {
                                total_tokens: prompt + total,
                                prompt_tokens: prompt,
                                completion_tokens: total,
                            })
                            .await;
                        return Ok(());
                    }

                    // Content
                    if let Some(msg) = parsed.get("message") {
                        if let Some(content) = msg.get("content").and_then(|c| c.as_str()) {
                            if !content.is_empty() {
                                let _ = tx.send(StreamEvent::Token(content.to_string())).await;
                            }
                        }

                        // Tool calls
                        if let Some(tool_calls) = msg.get("tool_calls").and_then(|t| t.as_array()) {
                            for tc in tool_calls {
                                if let Some(func) = tc.get("function") {
                                    let name = func
                                        .get("name")
                                        .and_then(|n| n.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let args = func
                                        .get("arguments")
                                        .map(|a| a.to_string())
                                        .unwrap_or_else(|| "{}".to_string());

                                    if !name.is_empty() {
                                        let _ = tx
                                            .send(StreamEvent::ToolCall {
                                                id: uuid::Uuid::new_v4().to_string(),
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

        Ok(())
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
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": ollama_messages,
            "stream": false,
            "keep_alive": self.keep_alive,
        });

        if let Some(temp) = temperature {
            body["options"] = serde_json::json!({"temperature": temp});
        }

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to send sync chat to Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            bail!("Ollama sync chat error {}: {}", status, error);
        }

        let parsed: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse Ollama sync response")?;

        let content = parsed
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        Ok(content)
    }

    async fn supports_modality(&self, model: &str, modality: Modality) -> Result<bool> {
        match modality {
            Modality::Text => Ok(true),
            Modality::Audio => Ok(false), // Ollama has no audio API
            Modality::Image | Modality::Video => {
                let spec = self.get_model_spec(model).await?;
                Ok(spec.capabilities.vision)
            }
        }
    }

    async fn embed(&self, text: &str, model: &str) -> Result<Vec<f32>> {
        let url = format!("{}/api/embeddings", self.base_url);
        let body = serde_json::json!({
            "model": model,
            "prompt": text,
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to send embedding request to Ollama")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            bail!("Ollama embedding error {}: {}", status, error);
        }

        let parsed: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse Ollama embedding response")?;

        let embedding = parsed
            .get("embedding")
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OllamaConfig;

    fn test_config() -> OllamaConfig {
        OllamaConfig {
            host: "http://localhost:11434".to_string(),
            port: 11434,
            keep_alive: -1,
        }
    }

    #[test]
    fn test_provider_id() {
        let provider = OllamaProvider::new(&test_config());
        assert_eq!(provider.id(), "ollama");
    }

    #[test]
    fn test_parse_show_response_gemma4() {
        let provider = OllamaProvider::new(&test_config());
        let body = serde_json::json!({
            "details": {
                "family": "gemma4",
                "families": ["gemma4"],
                "parameter_size": "26B",
                "quantization_level": "Q4_K_M",
                "format": "gguf"
            },
            "model_info": {
                "general.context_length": 262144
            },
            "parameters": "temperature 0.7\ntop_k 40\ntop_p 0.9\nnum_ctx 131072",
            "template": "{{ .System }}\n{{ .Prompt }}"
        });

        let spec = provider.parse_show_response("gemma4:26b", &body);
        assert_eq!(spec.name, "gemma4:26b");
        assert_eq!(spec.provider, "ollama");
        assert_eq!(spec.family, "gemma4");
        assert_eq!(spec.parameter_size, "26B");
        assert_eq!(spec.context_length, 262144);
        assert_eq!(spec.default_temperature, 0.7);
        assert_eq!(spec.default_top_k, 40);
        assert!(spec.capabilities.text);
        assert!(!spec.capabilities.audio);
    }

    #[test]
    fn test_parse_show_response_vision_model() {
        let provider = OllamaProvider::new(&test_config());
        let body = serde_json::json!({
            "details": {
                "family": "llama",
                "families": ["llama", "clip"],
                "parameter_size": "11B",
                "quantization_level": "Q4_0",
                "format": "gguf"
            },
            "model_info": {},
            "parameters": "",
        });

        let spec = provider.parse_show_response("llava:11b", &body);
        assert!(spec.capabilities.vision);
        assert!(spec.capabilities.video);
    }

    #[test]
    fn test_parse_show_response_minimal() {
        let provider = OllamaProvider::new(&test_config());
        let body = serde_json::json!({});

        let spec = provider.parse_show_response("unknown", &body);
        assert_eq!(spec.name, "unknown");
        assert_eq!(spec.context_length, 0);
        assert!(spec.capabilities.text);
    }
}
