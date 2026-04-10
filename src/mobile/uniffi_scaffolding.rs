// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! UniFFI scaffolding — bridges the UDL interface to the Rust engine.
//!
//! This module implements the types and interfaces defined in ernosagent.udl.
//! UniFFI generates Kotlin and Swift bindings from the UDL, and this module
//! provides the actual Rust implementations that those bindings call into.
//!
//! The generated code lives in:
//!   Android: mobile/androidApp/src/main/kotlin/generated/
//!   iOS: mobile/iosApp/ErnOS/Generated/

use crate::mobile::{self, DownloadProgress, InferenceMode};
use crate::mobile::model_manager::{ModelManager, ModelSpec, ModelStatus};
use std::path::PathBuf;

// ═══════════════════════════════════════════════════════════
//  Error type
// ═══════════════════════════════════════════════════════════

/// Error type exported to native platforms.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("No model loaded")]
    ModelNotLoaded,
    #[error("Download failed: {0}")]
    DownloadFailed(String),
    #[error("Inference error: {0}")]
    InferenceError(String),
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    #[error("Internal error: {0}")]
    InternalError(String),
}

// ═══════════════════════════════════════════════════════════
//  Data types
// ═══════════════════════════════════════════════════════════

/// Model spec for the mobile UI.
#[derive(Debug, Clone)]
pub struct MobileModelSpec {
    pub name: String,
    pub effective_params: String,
    pub size_human: String,
    pub download_size: String,
    pub min_ram_gb: u32,
    pub modalities: Vec<String>,
    pub context_length: u32,
}

impl From<&ModelSpec> for MobileModelSpec {
    fn from(spec: &ModelSpec) -> Self {
        Self {
            name: spec.name.clone(),
            effective_params: spec.effective_params.clone(),
            size_human: spec.size_human.clone(),
            download_size: ModelManager::total_download_size_human(spec),
            min_ram_gb: spec.min_ram_gb,
            modalities: spec.modalities.clone(),
            context_length: spec.context_length,
        }
    }
}

/// Engine status snapshot for the UI.
#[derive(Debug, Clone)]
pub struct EngineStatus {
    pub model_name: String,
    pub inference_mode: String,
    pub is_ready: bool,
    pub is_desktop_connected: bool,
    pub status_summary: String,
}

/// A streamed token from inference.
#[derive(Debug, Clone)]
pub struct ChatToken {
    pub content: String,
    pub is_thinking: bool,
    pub is_eos: bool,
}

/// Final chat result after inference completes.
#[derive(Debug, Clone)]
pub struct ChatResult {
    pub response: String,
    pub total_tokens: u64,
    pub context_usage: f32,
    pub neural_snapshot_json: Option<String>,
}

// ═══════════════════════════════════════════════════════════
//  Callbacks
// ═══════════════════════════════════════════════════════════

/// Callback for streaming tokens during inference.
pub trait StreamCallback: Send + Sync {
    fn on_token(&self, token: ChatToken);
    fn on_tool_call(&self, name: String, arguments: String);
    fn on_complete(&self, result: ChatResult);
    fn on_error(&self, message: String);
}

/// Callback for model download progress.
pub trait DownloadCallback: Send + Sync {
    fn on_progress(&self, progress: DownloadProgress);
    fn on_complete(&self, model_name: String);
    fn on_error(&self, message: String);
}

// ═══════════════════════════════════════════════════════════
//  Engine implementation for UniFFI
// ═══════════════════════════════════════════════════════════

/// The UniFFI-exported engine.
///
/// This wraps our internal `mobile::engine::ErnOSEngine` and adapts
/// its API to the types defined in the UDL.
pub struct UniFFIEngine {
    inner: mobile::engine::ErnOSEngine,
}

impl UniFFIEngine {
    /// Create a new engine.
    pub fn new(data_dir: String) -> Result<Self, EngineError> {
        let engine = mobile::engine::ErnOSEngine::new(PathBuf::from(data_dir))
            .map_err(|e| EngineError::InternalError(e.to_string()))?;
        Ok(Self { inner: engine })
    }

    // ── Engine State ──

    pub fn is_ready(&self) -> bool {
        self.inner.is_ready()
    }

    pub fn get_status(&self) -> EngineStatus {
        EngineStatus {
            model_name: self.inner.loaded_model().unwrap_or_default(),
            inference_mode: self.inner.get_inference_mode().to_string(),
            is_ready: self.inner.is_ready(),
            is_desktop_connected: self.inner.is_desktop_connected(),
            status_summary: self.inner.status_summary(),
        }
    }

    // ── Inference Mode ──

    pub fn set_inference_mode(&self, mode: InferenceMode) {
        self.inner.set_inference_mode(mode);
    }

    pub fn get_inference_mode(&self) -> InferenceMode {
        self.inner.get_inference_mode()
    }

    // ── Model Management ──

    pub fn available_models(&self) -> Vec<MobileModelSpec> {
        self.inner
            .available_models()
            .iter()
            .map(MobileModelSpec::from)
            .collect()
    }

    pub fn recommended_model(&self, available_ram_mb: u64) -> MobileModelSpec {
        MobileModelSpec::from(&self.inner.recommended_model(available_ram_mb))
    }

    pub fn model_status(&self, model_name: &str) -> ModelStatus {
        self.inner.model_status(model_name)
    }

    pub fn download_size(&self, model_name: &str) -> Option<String> {
        self.inner.download_size(model_name)
    }

    pub fn load_model(&self, model_name: &str) -> Result<(), EngineError> {
        self.inner
            .load_model(model_name)
            .map_err(|e| EngineError::ModelNotFound(e.to_string()))
    }

    pub fn loaded_model(&self) -> Option<String> {
        self.inner.loaded_model()
    }

    pub fn delete_model(&self, model_name: &str) -> Result<(), EngineError> {
        self.inner
            .delete_model(model_name)
            .map_err(|e| EngineError::ModelNotFound(e.to_string()))
    }

    // ── Download ──

    pub fn download_model(
        &self,
        model_name: &str,
        callback: Box<dyn DownloadCallback>,
    ) -> Result<(), EngineError> {
        callback.on_error(format!(
            "Model download not yet implemented (Phase 6). \
             Please manually place {} in the models directory.",
            model_name
        ));
        Err(EngineError::DownloadFailed(
            "Download pipeline not yet implemented".to_string(),
        ))
    }

    // ── Chat ──

    pub fn chat(
        &self,
        message: &str,
        callback: Box<dyn StreamCallback>,
    ) -> Result<(), EngineError> {
        if !self.inner.is_ready() {
            return Err(EngineError::ModelNotLoaded);
        }

        // Stub: signal back via callback
        callback.on_token(ChatToken {
            content: format!("(stub) Processing: {}\n", message),
            is_thinking: false,
            is_eos: false,
        });

        callback.on_complete(ChatResult {
            response: format!("(stub) Response to: {}", message),
            total_tokens: 1,
            context_usage: 0.0,
            neural_snapshot_json: None,
        });

        Ok(())
    }

    pub fn chat_with_images(
        &self,
        message: &str,
        _images: Vec<Vec<u8>>,
        callback: Box<dyn StreamCallback>,
    ) -> Result<(), EngineError> {
        self.chat(message, callback)
    }

    pub fn chat_with_audio(
        &self,
        message: &str,
        _audio_data: Vec<u8>,
        callback: Box<dyn StreamCallback>,
    ) -> Result<(), EngineError> {
        self.chat(message, callback)
    }

    // ── Desktop Relay ──

    pub fn disconnect_desktop(&self) {
        self.inner.disconnect_desktop();
    }

    pub fn is_desktop_connected(&self) -> bool {
        self.inner.is_desktop_connected()
    }

    // ── Memory ──

    pub fn memory_summary(&self) -> String {
        "Memory system active".to_string()
    }

    pub fn total_lessons_learned(&self) -> u64 {
        let les_path = self.inner.data_dir().join("lessons.json");
        std::fs::read_to_string(&les_path)
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<serde_json::Value>>(&s).ok())
            .map(|v| v.len() as u64)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct TestStreamCallback {
        token_count: AtomicU32,
        completed: AtomicU32,
    }

    impl TestStreamCallback {
        fn new() -> Self {
            Self {
                token_count: AtomicU32::new(0),
                completed: AtomicU32::new(0),
            }
        }
    }

    impl StreamCallback for TestStreamCallback {
        fn on_token(&self, _token: ChatToken) {
            self.token_count.fetch_add(1, Ordering::Relaxed);
        }
        fn on_tool_call(&self, _name: String, _args: String) {}
        fn on_complete(&self, _result: ChatResult) {
            self.completed.fetch_add(1, Ordering::Relaxed);
        }
        fn on_error(&self, _message: String) {}
    }

    #[test]
    fn test_engine_creation() {
        let tmp = tempfile::TempDir::new().unwrap();
        let engine = UniFFIEngine::new(tmp.path().to_str().unwrap().to_string());
        assert!(engine.is_ok());
        assert!(!engine.unwrap().is_ready());
    }

    #[test]
    fn test_engine_status() {
        let tmp = tempfile::TempDir::new().unwrap();
        let engine = UniFFIEngine::new(tmp.path().to_str().unwrap().to_string()).unwrap();
        let status = engine.get_status();
        assert!(!status.is_ready);
        assert_eq!(status.inference_mode, "Hybrid (smart routing)");
    }

    #[test]
    fn test_inference_mode_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let engine = UniFFIEngine::new(tmp.path().to_str().unwrap().to_string()).unwrap();
        engine.set_inference_mode(InferenceMode::Local);
        assert_eq!(engine.get_inference_mode(), InferenceMode::Local);
        engine.set_inference_mode(InferenceMode::ChainOfAgents);
        assert_eq!(engine.get_inference_mode(), InferenceMode::ChainOfAgents);
    }

    #[test]
    fn test_available_models() {
        let tmp = tempfile::TempDir::new().unwrap();
        let engine = UniFFIEngine::new(tmp.path().to_str().unwrap().to_string()).unwrap();
        let models = engine.available_models();
        assert_eq!(models.len(), 2);
        assert!(models.iter().any(|m| m.name.contains("E2B")));
        assert!(models.iter().any(|m| m.name.contains("E4B")));
    }

    #[test]
    fn test_recommended_model() {
        let tmp = tempfile::TempDir::new().unwrap();
        let engine = UniFFIEngine::new(tmp.path().to_str().unwrap().to_string()).unwrap();
        let rec_low = engine.recommended_model(8192);
        assert!(rec_low.name.contains("E2B"));
        let rec_high = engine.recommended_model(16384);
        assert!(rec_high.name.contains("E4B"));
    }

    #[test]
    fn test_chat_without_model() {
        let tmp = tempfile::TempDir::new().unwrap();
        let engine = UniFFIEngine::new(tmp.path().to_str().unwrap().to_string()).unwrap();
        let callback = TestStreamCallback::new();
        let result = engine.chat("Hello", Box::new(callback));
        assert!(result.is_err());
    }

    #[test]
    fn test_model_status_not_downloaded() {
        let tmp = tempfile::TempDir::new().unwrap();
        let engine = UniFFIEngine::new(tmp.path().to_str().unwrap().to_string()).unwrap();
        let status = engine.model_status("Gemma 4 E2B");
        assert_eq!(status, ModelStatus::NotDownloaded);
    }

    #[test]
    fn test_desktop_connection() {
        let tmp = tempfile::TempDir::new().unwrap();
        let engine = UniFFIEngine::new(tmp.path().to_str().unwrap().to_string()).unwrap();
        assert!(!engine.is_desktop_connected());
        engine.disconnect_desktop();
        assert!(!engine.is_desktop_connected());
    }

    #[test]
    fn test_mobile_model_spec_from() {
        let specs = ModelManager::available_models();
        let mobile_spec = MobileModelSpec::from(&specs[0]);
        assert!(mobile_spec.name.contains("E2B"));
        assert!(mobile_spec.download_size.contains("GB"));
        assert_eq!(mobile_spec.min_ram_gb, 8);
        assert_eq!(mobile_spec.context_length, 128_000);
    }

    #[test]
    fn test_engine_error_display() {
        let err = EngineError::ModelNotFound("test".to_string());
        assert!(err.to_string().contains("test"));
        let err = EngineError::ModelNotLoaded;
        assert!(err.to_string().contains("loaded"));
    }
}
