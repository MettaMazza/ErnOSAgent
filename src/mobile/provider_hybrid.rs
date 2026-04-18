// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Hybrid provider — smart routing between local (E2B/E4B) and desktop (26B) inference.
//!
//! This is the default inference mode. It classifies prompt complexity
//! and routes simple queries to the on-device model for speed/privacy,
//! while routing complex queries to the desktop for 26B reasoning quality.

use super::provider_local::MobileLocalProvider;
use super::provider_relay::DesktopRelayProvider;
use crate::model::spec::{Modality, ModelSpec, ModelSummary};
use crate::provider::{Message, Provider, ProviderStatus, StreamEvent, ToolDefinition};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Prompt complexity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Complexity {
    /// Simple: greetings, short factual, memory recall, single-turn
    Simple,
    /// Complex: multi-step reasoning, code review, document analysis, tool-heavy
    Complex,
}

/// Hybrid provider that routes between local and desktop inference.
pub struct HybridProvider {
    local: Arc<MobileLocalProvider>,
    relay: Arc<DesktopRelayProvider>,
}

impl HybridProvider {
    pub fn new(local: Arc<MobileLocalProvider>, relay: Arc<DesktopRelayProvider>) -> Self {
        Self { local, relay }
    }

    /// Classify a prompt's complexity to decide routing.
    ///
    /// This is a heuristic that considers:
    /// - Token count (long prompts → complex)
    /// - Presence of images/audio (multimodal → prefer local for privacy)
    /// - Keywords indicating complexity (code, analyze, compare, explain in detail)
    /// - Number of questions (multi-part → complex)
    /// - Tool requirements (web search, code exec → complex)
    pub fn classify_complexity(
        &self,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        has_images: bool,
        has_audio: bool,
    ) -> Complexity {
        let last_user_msg = messages
            .iter()
            .rev()
            .find(|m| m.role == "user")
            .map(|m| m.content.as_str())
            .unwrap_or("");

        let word_count = last_user_msg.split_whitespace().count();
        let question_marks = last_user_msg.matches('?').count();

        // Complex indicators
        let complex_keywords = [
            "analyze",
            "compare",
            "explain in detail",
            "review",
            "refactor",
            "debug",
            "implement",
            "architecture",
            "step by step",
            "comprehensive",
            "thorough",
            "write a",
            "create a",
            "build a",
            "design a",
            "multi-step",
            "workflow",
            "pipeline",
        ];

        let has_complex_keyword = complex_keywords
            .iter()
            .any(|kw| last_user_msg.to_lowercase().contains(kw));

        let has_many_tools = tools.map_or(false, |t| t.len() > 3);
        let is_multi_part = question_marks >= 2;
        let is_long = word_count > 50;

        // Multimodal with images/audio → prefer local for privacy
        // (the edge models handle this natively)
        if has_images || has_audio {
            return Complexity::Simple; // Process locally for privacy
        }

        // Score complexity
        let mut complexity_score = 0u32;
        if has_complex_keyword {
            complexity_score += 3;
        }
        if is_multi_part {
            complexity_score += 2;
        }
        if is_long {
            complexity_score += 2;
        }
        if has_many_tools {
            complexity_score += 2;
        }

        if complexity_score >= 3 {
            Complexity::Complex
        } else {
            Complexity::Simple
        }
    }

    /// Determine which provider to use for this request.
    fn route(&self, messages: &[Message], tools: Option<&[ToolDefinition]>) -> RouteDecision {
        let has_images = messages.iter().any(|m| !m.images.is_empty());
        // Audio would be passed as binary data, checked elsewhere
        let has_audio = false;

        let complexity = self.classify_complexity(messages, tools, has_images, has_audio);
        let desktop_available = self.relay.is_connected();

        match (complexity, desktop_available) {
            // Simple → always local
            (Complexity::Simple, _) => RouteDecision::Local,
            // Complex + desktop → relay
            (Complexity::Complex, true) => RouteDecision::Desktop,
            // Complex + no desktop → local (fallback)
            (Complexity::Complex, false) => {
                tracing::info!("Complex prompt but desktop unavailable, falling back to local");
                RouteDecision::Local
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum RouteDecision {
    Local,
    Desktop,
}

#[async_trait]
impl Provider for HybridProvider {
    fn id(&self) -> &str {
        "hybrid"
    }

    fn display_name(&self) -> &str {
        "Hybrid (smart routing)"
    }

    async fn list_models(&self) -> Result<Vec<ModelSummary>> {
        let mut models = self.local.list_models().await?;
        models.extend(self.relay.list_models().await?);
        Ok(models)
    }

    async fn get_model_spec(&self, model: &str) -> Result<ModelSpec> {
        // Try local first, fall back to relay
        match self.local.get_model_spec(model).await {
            Ok(spec) => Ok(spec),
            Err(_) => self.relay.get_model_spec(model).await,
        }
    }

    async fn chat(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[ToolDefinition]>,
        tx: mpsc::Sender<StreamEvent>,
    ) -> Result<()> {
        let decision = self.route(messages, tools);

        tracing::info!(
            route = ?decision,
            "Hybrid provider routing"
        );

        match decision {
            RouteDecision::Local => self.local.chat(model, messages, tools, tx).await,
            RouteDecision::Desktop => self.relay.chat(model, messages, tools, tx).await,
        }
    }

    async fn chat_sync(
        &self,
        model: &str,
        messages: &[Message],
        temperature: Option<f64>,
    ) -> Result<String> {
        let decision = self.route(messages, None);
        match decision {
            RouteDecision::Local => self.local.chat_sync(model, messages, temperature).await,
            RouteDecision::Desktop => self.relay.chat_sync(model, messages, temperature).await,
        }
    }

    async fn supports_modality(&self, model: &str, modality: Modality) -> Result<bool> {
        // If either provider supports it, we support it
        let local = self
            .local
            .supports_modality(model, modality)
            .await
            .unwrap_or(false);
        let relay = self
            .relay
            .supports_modality(model, modality)
            .await
            .unwrap_or(false);
        Ok(local || relay)
    }

    async fn embed(&self, text: &str, model: &str) -> Result<Vec<f32>> {
        // Try relay first (desktop has better embeddings)
        match self.relay.embed(text, model).await {
            Ok(v) => Ok(v),
            Err(_) => self.local.embed(text, model).await,
        }
    }

    async fn health(&self) -> Result<ProviderStatus> {
        let local = self.local.health().await?;
        let relay = self.relay.health().await.unwrap_or(ProviderStatus {
            available: false,
            latency_ms: None,
            error: Some("Desktop unavailable".to_string()),
            models_loaded: vec![],
        });

        let mut models = local.models_loaded.clone();
        models.extend(relay.models_loaded);

        Ok(ProviderStatus {
            available: local.available || relay.available,
            latency_ms: local.latency_ms,
            error: if !local.available && !relay.available {
                Some("No inference available (no local model, no desktop)".to_string())
            } else {
                None
            },
            models_loaded: models,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_providers() -> (Arc<MobileLocalProvider>, Arc<DesktopRelayProvider>) {
        (
            Arc::new(MobileLocalProvider::new()),
            Arc::new(DesktopRelayProvider::new()),
        )
    }

    fn user_msg(content: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: content.to_string(),
            images: vec![],
        }
    }

    #[test]
    fn test_simple_prompts() {
        let (local, relay) = make_providers();
        let hybrid = HybridProvider::new(local, relay);

        // Simple greetings
        assert_eq!(
            hybrid.classify_complexity(&[user_msg("Hello!")], None, false, false),
            Complexity::Simple
        );
        assert_eq!(
            hybrid.classify_complexity(&[user_msg("What time is it?")], None, false, false),
            Complexity::Simple
        );
        assert_eq!(
            hybrid.classify_complexity(&[user_msg("Thanks for the help")], None, false, false),
            Complexity::Simple
        );
    }

    #[test]
    fn test_complex_prompts() {
        let (local, relay) = make_providers();
        let hybrid = HybridProvider::new(local, relay);

        // Complex: code review
        assert_eq!(
            hybrid.classify_complexity(
                &[user_msg("Analyze this architecture and explain in detail how the ReAct loop works, then compare it to other approaches")],
                None, false, false
            ),
            Complexity::Complex
        );

        // Complex: multi-part with reasoning requirements
        assert_eq!(
            hybrid.classify_complexity(
                &[user_msg("Compare Rust and Go in detail: What are the key architectural differences? How does the borrow checker work? Why is it faster for systems programming?")],
                None, false, false
            ),
            Complexity::Complex
        );
    }

    #[test]
    fn test_multimodal_routes_local() {
        let (local, relay) = make_providers();
        let hybrid = HybridProvider::new(local, relay);

        // Images → local (privacy)
        assert_eq!(
            hybrid.classify_complexity(&[user_msg("What is this?")], None, true, false),
            Complexity::Simple
        );

        // Audio → local (privacy)
        assert_eq!(
            hybrid.classify_complexity(&[user_msg("")], None, false, true),
            Complexity::Simple
        );
    }

    #[test]
    fn test_routing_fallback() {
        let (local, relay) = make_providers();
        let hybrid = HybridProvider::new(local, relay);

        // Complex prompt + no desktop → should still route (to local fallback)
        let decision = hybrid.route(
            &[user_msg(
                "Analyze and refactor this entire codebase step by step",
            )],
            None,
        );
        // Desktop not connected → falls back to local
        assert!(matches!(decision, RouteDecision::Local));
    }

    #[test]
    fn test_provider_metadata() {
        let (local, relay) = make_providers();
        let hybrid = HybridProvider::new(local, relay);
        assert_eq!(hybrid.id(), "hybrid");
        assert_eq!(hybrid.display_name(), "Hybrid (smart routing)");
    }

    #[tokio::test]
    async fn test_health_local_only() {
        let (local, relay) = make_providers();
        let tmp = tempfile::TempDir::new().unwrap();
        let model_path = tmp.path().join("test.gguf");
        std::fs::write(&model_path, b"fake").unwrap();
        local.load_model(&model_path, None, -1, 4).unwrap();

        let hybrid = HybridProvider::new(local, relay);
        let health = hybrid.health().await.unwrap();
        assert!(health.available);
    }

    #[tokio::test]
    async fn test_health_nothing_available() {
        let (local, relay) = make_providers();
        let hybrid = HybridProvider::new(local, relay);
        let health = hybrid.health().await.unwrap();
        assert!(!health.available);
        assert!(health.error.is_some());
    }
}
