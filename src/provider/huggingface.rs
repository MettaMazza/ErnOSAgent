//! HuggingFace provider — SECONDARY. Hub API, GGUF downloads, Inference API.

use crate::model::spec::{ModelCapabilities, ModelSpec, ModelSummary, Modality};
use crate::provider::{Message, Provider, ProviderStatus, StreamEvent, ToolDefinition};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use reqwest::Client;
use std::time::Instant;
use tokio::sync::mpsc;

pub struct HuggingFaceProvider {
    client: Client,
    api_token: String,
    endpoint: String,
    model_id: String,
}

impl HuggingFaceProvider {
    pub fn new(config: &crate::config::HuggingFaceConfig) -> Self {
        Self {
            client: Client::new(),
            api_token: config.api_token.clone(),
            endpoint: config.endpoint.trim_end_matches('/').to_string(),
            model_id: config.model_id.clone(),
        }
    }

    fn auth_header(&self) -> Option<String> {
        if self.api_token.is_empty() {
            None
        } else {
            Some(format!("Bearer {}", self.api_token))
        }
    }
}

#[async_trait]
impl Provider for HuggingFaceProvider {
    fn id(&self) -> &str { "huggingface" }
    fn display_name(&self) -> &str { "HuggingFace" }

    async fn list_models(&self) -> Result<Vec<ModelSummary>> {
        // HuggingFace Hub search for GGUF models
        let url = "https://huggingface.co/api/models?filter=gguf&sort=downloads&direction=-1&limit=20";
        let mut req = self.client.get(url);
        if let Some(auth) = self.auth_header() {
            req = req.header("Authorization", auth);
        }

        let resp = req.send().await.context("Failed to query HuggingFace Hub")?;
        let body: Vec<serde_json::Value> = resp.json().await.context("Failed to parse HF response")?;

        let models = body.iter().map(|item| {
            ModelSummary {
                name: item.get("modelId").and_then(|v| v.as_str()).unwrap_or("unknown").to_string(),
                provider: "huggingface".to_string(),
                parameter_size: String::new(),
                quantization_level: String::new(),
                capabilities: ModelCapabilities { text: true, ..Default::default() },
                context_length: 0,
            }
        }).collect();

        Ok(models)
    }

    async fn get_model_spec(&self, model: &str) -> Result<ModelSpec> {
        let url = format!("https://huggingface.co/api/models/{}", model);
        let mut req = self.client.get(&url);
        if let Some(auth) = self.auth_header() {
            req = req.header("Authorization", auth);
        }

        let resp = req.send().await
            .with_context(|| format!("Failed to fetch model info for '{}'", model))?;

        if !resp.status().is_success() {
            bail!("HuggingFace API returned {} for model '{}'", resp.status(), model);
        }

        let body: serde_json::Value = resp.json().await.context("Failed to parse HF model info")?;

        Ok(ModelSpec {
            name: model.to_string(),
            provider: "huggingface".to_string(),
            capabilities: ModelCapabilities { text: true, ..Default::default() },
            raw_info: body,
            ..Default::default()
        })
    }

    async fn chat(
        &self, model: &str, messages: &[Message],
        _tools: Option<&[ToolDefinition]>, tx: mpsc::Sender<StreamEvent>,
    ) -> Result<()> {
        let url = format!("{}/models/{}/v1/chat/completions", self.endpoint, model);
        let body = serde_json::json!({ "model": model, "messages": messages, "stream": false });

        let mut req = self.client.post(&url).json(&body);
        if let Some(auth) = self.auth_header() {
            req = req.header("Authorization", auth);
        }

        let resp = req.send().await.context("Failed to send chat to HuggingFace")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            bail!("HuggingFace chat error {}: {}", status, error);
        }

        let parsed: serde_json::Value = resp.json().await.context("Failed to parse HF chat response")?;
        let content = parsed.get("choices").and_then(|c| c.as_array()).and_then(|arr| arr.first())
            .and_then(|choice| choice.get("message")).and_then(|msg| msg.get("content"))
            .and_then(|c| c.as_str()).unwrap_or("");

        if !content.is_empty() {
            let _ = tx.send(StreamEvent::Token(content.to_string())).await;
        }
        let _ = tx.send(StreamEvent::Done { total_tokens: 0, prompt_tokens: 0, completion_tokens: 0 }).await;

        Ok(())
    }

    async fn chat_sync(&self, model: &str, messages: &[Message], temperature: Option<f64>) -> Result<String> {
        let url = format!("{}/models/{}/v1/chat/completions", self.endpoint, model);
        let mut body = serde_json::json!({ "model": model, "messages": messages, "stream": false });
        if let Some(temp) = temperature { body["temperature"] = serde_json::json!(temp); }

        let mut req = self.client.post(&url).json(&body);
        if let Some(auth) = self.auth_header() { req = req.header("Authorization", auth); }

        let resp = req.send().await.context("Failed to send sync chat to HuggingFace")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let error = resp.text().await.unwrap_or_default();
            bail!("HuggingFace sync chat error {}: {}", status, error);
        }

        let parsed: serde_json::Value = resp.json().await.context("Failed to parse HF response")?;
        Ok(parsed.get("choices").and_then(|c| c.as_array()).and_then(|arr| arr.first())
            .and_then(|choice| choice.get("message")).and_then(|msg| msg.get("content"))
            .and_then(|c| c.as_str()).unwrap_or("").to_string())
    }

    async fn supports_modality(&self, _model: &str, modality: Modality) -> Result<bool> {
        Ok(matches!(modality, Modality::Text))
    }

    async fn embed(&self, text: &str, model: &str) -> Result<Vec<f32>> {
        let url = format!("{}/pipeline/feature-extraction/{}", self.endpoint, model);
        let body = serde_json::json!({ "inputs": text });

        let mut req = self.client.post(&url).json(&body);
        if let Some(auth) = self.auth_header() { req = req.header("Authorization", auth); }

        let resp = req.send().await.context("Failed to send embedding to HuggingFace")?;
        if !resp.status().is_success() {
            bail!("HuggingFace embedding error {}", resp.status());
        }

        let parsed: serde_json::Value = resp.json().await.context("Failed to parse HF embedding")?;
        let embedding = parsed.as_array()
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
            .unwrap_or_default();

        Ok(embedding)
    }

    async fn health(&self) -> Result<ProviderStatus> {
        let start = Instant::now();
        match self.client.get("https://huggingface.co/api/whoami-v2")
            .header("Authorization", self.auth_header().unwrap_or_default())
            .send().await {
            Ok(resp) if resp.status().is_success() => Ok(ProviderStatus {
                available: true, latency_ms: Some(start.elapsed().as_millis() as u64),
                error: None, models_loaded: Vec::new(),
            }),
            Ok(resp) => Ok(ProviderStatus {
                available: true, latency_ms: Some(start.elapsed().as_millis() as u64),
                error: Some(format!("Auth check returned {} (may still work for public models)", resp.status())),
                models_loaded: Vec::new(),
            }),
            Err(e) => Ok(ProviderStatus {
                available: false, latency_ms: None,
                error: Some(format!("Cannot reach HuggingFace: {}", e)), models_loaded: Vec::new(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HuggingFaceConfig;

    #[test]
    fn test_provider_id() {
        let provider = HuggingFaceProvider::new(&HuggingFaceConfig {
            api_token: String::new(), model_id: String::new(),
            endpoint: "https://api-inference.huggingface.co".to_string(),
        });
        assert_eq!(provider.id(), "huggingface");
    }

    #[test]
    fn test_auth_header_empty_token() {
        let provider = HuggingFaceProvider::new(&HuggingFaceConfig {
            api_token: String::new(), model_id: String::new(),
            endpoint: "https://api-inference.huggingface.co".to_string(),
        });
        assert!(provider.auth_header().is_none());
    }

    #[test]
    fn test_auth_header_with_token() {
        let provider = HuggingFaceProvider::new(&HuggingFaceConfig {
            api_token: "hf_test123".to_string(), model_id: String::new(),
            endpoint: "https://api-inference.huggingface.co".to_string(),
        });
        assert_eq!(provider.auth_header(), Some("Bearer hf_test123".to_string()));
    }
}
