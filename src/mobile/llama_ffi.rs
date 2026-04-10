// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Safe Rust FFI bindings to llama.cpp C API.
//!
//! This module provides safe wrappers around the llama.cpp C functions.
//! On mobile, llama.cpp is linked as a static library (built with NDK/Metal).
//! On desktop, this module is also available but the HTTP-based provider
//! (llamacpp.rs) is preferred since it manages a server subprocess.
//!
//! The FFI bindings are conditionally compiled — they are only active when
//! the `mobile-native` feature is enabled (which requires the actual C library
//! to be linked). When the feature is off, the module provides stub types
//! that allow the rest of the codebase to compile and test.

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

// ═══════════════════════════════════════════════════════════
//  Types (always available for API surface)
// ═══════════════════════════════════════════════════════════

/// GPU acceleration backend for the platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// No GPU — CPU only
    None,
    /// Apple Metal (iOS, macOS)
    Metal,
    /// OpenCL (Android Adreno)
    OpenCL,
    /// Vulkan (Android, some desktop)
    Vulkan,
    /// CUDA (NVIDIA desktop)
    Cuda,
}

impl GpuBackend {
    /// Auto-detect the best GPU backend for this platform.
    pub fn detect() -> Self {
        #[cfg(target_os = "ios")]
        return Self::Metal;
        #[cfg(target_os = "macos")]
        return Self::Metal;
        #[cfg(target_os = "android")]
        return Self::OpenCL; // Adreno GPUs
        #[cfg(not(any(target_os = "ios", target_os = "macos", target_os = "android")))]
        return Self::None;
    }
}

/// Parameters for loading a GGUF model.
#[derive(Debug, Clone)]
pub struct ModelParams {
    /// Path to the GGUF model file.
    pub model_path: String,
    /// Optional path to the multimodal projector GGUF.
    pub mmproj_path: Option<String>,
    /// Number of layers to offload to GPU (-1 = all).
    pub n_gpu_layers: i32,
    /// GPU backend to use.
    pub gpu_backend: GpuBackend,
    /// Whether to use memory mapping.
    pub use_mmap: bool,
    /// Whether to use memory locking (pinned RAM).
    pub use_mlock: bool,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            mmproj_path: None,
            n_gpu_layers: -1, // Full GPU offload
            gpu_backend: GpuBackend::detect(),
            use_mmap: true,
            use_mlock: false,
        }
    }
}

/// Parameters for a generation context.
#[derive(Debug, Clone)]
pub struct ContextParams {
    /// Context window size in tokens.
    pub n_ctx: u32,
    /// Batch size for prompt processing.
    pub n_batch: u32,
    /// Number of CPU threads.
    pub n_threads: u32,
    /// Number of CPU threads for batch processing.
    pub n_threads_batch: u32,
}

impl Default for ContextParams {
    fn default() -> Self {
        let n_threads = std::thread::available_parallelism()
            .map(|p| p.get() as u32)
            .unwrap_or(4)
            .min(8); // Cap for mobile thermal management

        Self {
            n_ctx: 4096,  // Reasonable default for mobile
            n_batch: 512,
            n_threads,
            n_threads_batch: n_threads,
        }
    }
}

/// Sampling parameters for text generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub seed: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 0, // 0 = random
        }
    }
}

/// A token emitted during generation.
#[derive(Debug, Clone)]
pub struct GeneratedToken {
    /// The token ID.
    pub id: i32,
    /// The decoded text piece.
    pub text: String,
    /// Whether this is the end-of-generation token.
    pub is_eos: bool,
}

/// Model info retrieved after loading.
#[derive(Debug, Clone)]
pub struct NativeModelInfo {
    /// Model name from metadata.
    pub name: String,
    /// Architecture (e.g., "gemma2", "llama").
    pub architecture: String,
    /// Total parameter count.
    pub n_params: u64,
    /// Maximum context length.
    pub n_ctx_train: u32,
    /// Vocabulary size.
    pub n_vocab: u32,
    /// Embedding dimension.
    pub n_embd: u32,
    /// Number of layers.
    pub n_layer: u32,
    /// Number of attention heads.
    pub n_head: u32,
    /// File size in bytes.
    pub file_size: u64,
    /// Quantization type string.
    pub quantization: String,
}

/// Opaque handle to a loaded model.
///
/// When `mobile-native` is enabled, this wraps *mut llama_model.
/// In stub mode, this is a placeholder.
pub struct LlamaModel {
    pub(crate) info: NativeModelInfo,
    pub(crate) params: ModelParams,
    // When FFI is linked: _ptr: *mut ffi::llama_model,
}

impl std::fmt::Debug for LlamaModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaModel")
            .field("name", &self.info.name)
            .field("params", &self.info.n_params)
            .field("ctx_train", &self.info.n_ctx_train)
            .finish()
    }
}

/// Opaque handle to a generation context.
///
/// When `mobile-native` is enabled, this wraps *mut llama_context.
/// In stub mode, this is a placeholder.
pub struct LlamaContext {
    pub(crate) ctx_params: ContextParams,
    // When FFI is linked: _ptr: *mut ffi::llama_context,
}

impl std::fmt::Debug for LlamaContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaContext")
            .field("n_ctx", &self.ctx_params.n_ctx)
            .field("n_threads", &self.ctx_params.n_threads)
            .finish()
    }
}

/// Cancellation token — set to true to abort generation.
pub struct CancelFlag(AtomicBool);

impl CancelFlag {
    pub fn new() -> Self {
        Self(AtomicBool::new(false))
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.0.store(false, Ordering::Relaxed);
    }
}

// ═══════════════════════════════════════════════════════════
//  Public API (stub implementation — Phase 2 FFI link)
// ═══════════════════════════════════════════════════════════

impl LlamaModel {
    /// Load a GGUF model from disk.
    ///
    /// When the `mobile-native` feature is active, this calls
    /// `llama_model_load_from_file()` via C FFI. In stub mode,
    /// it creates a model handle with metadata.
    pub fn load(params: ModelParams) -> anyhow::Result<Self> {
        let path = Path::new(&params.model_path);
        if !path.exists() {
            anyhow::bail!("Model file not found: {}", params.model_path);
        }

        let file_size = std::fs::metadata(path)
            .map(|m| m.len())
            .unwrap_or(0);

        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Detect architecture and params from filename heuristics
        let (architecture, n_params) = if name.contains("E2B") {
            ("gemma2".to_string(), 2_300_000_000u64)
        } else if name.contains("E4B") {
            ("gemma2".to_string(), 4_500_000_000)
        } else if name.contains("26b") || name.contains("27b") {
            ("gemma2".to_string(), 26_000_000_000)
        } else {
            ("unknown".to_string(), 0)
        };

        let n_ctx_train = if name.contains("E2B") || name.contains("E4B") {
            128_000
        } else {
            256_000
        };

        let quantization = if name.contains("Q4_K_M") {
            "Q4_K_M"
        } else if name.contains("Q8_0") {
            "Q8_0"
        } else if name.contains("BF16") {
            "BF16"
        } else {
            "unknown"
        }
        .to_string();

        let info = NativeModelInfo {
            name: name.clone(),
            architecture,
            n_params,
            n_ctx_train,
            n_vocab: 0, // Will be populated from actual model metadata
            n_embd: 0,
            n_layer: 0,
            n_head: 0,
            file_size,
            quantization,
        };

        tracing::info!(
            model = %name,
            params = n_params,
            ctx = n_ctx_train,
            gpu_backend = ?params.gpu_backend,
            gpu_layers = params.n_gpu_layers,
            "LlamaModel::load (stub — FFI will replace)"
        );

        Ok(Self { info, params })
    }

    /// Get model information.
    pub fn info(&self) -> &NativeModelInfo {
        &self.info
    }

    /// Check if multimodal projector is loaded.
    pub fn has_mmproj(&self) -> bool {
        self.params.mmproj_path.is_some()
    }
}

impl LlamaContext {
    /// Create a new generation context for a loaded model.
    pub fn new(_model: &LlamaModel, params: ContextParams) -> anyhow::Result<Self> {
        tracing::info!(
            n_ctx = params.n_ctx,
            n_batch = params.n_batch,
            n_threads = params.n_threads,
            "LlamaContext::new (stub — FFI will replace)"
        );

        Ok(Self { ctx_params: params })
    }

    /// Tokenize a text string into token IDs.
    pub fn tokenize(&self, _text: &str, _add_bos: bool) -> Vec<i32> {
        // Stub: return empty. Real impl calls llama_tokenize().
        vec![]
    }

    /// Decode a token ID to its text piece.
    pub fn token_to_text(&self, _token: i32) -> String {
        // Stub: return empty. Real impl calls llama_token_to_piece().
        String::new()
    }

    /// Run a generation loop, yielding tokens one at a time.
    ///
    /// This is the core inference loop:
    /// 1. Process input tokens through the model (llama_decode)
    /// 2. Sample the next token (llama_sampling_sample)
    /// 3. Check for EOS or cancellation
    /// 4. Yield the token via the callback
    /// 5. Repeat until max_tokens or EOS
    pub fn generate(
        &mut self,
        _input_tokens: &[i32],
        _sampling: &SamplingParams,
        _max_tokens: u32,
        cancel: &CancelFlag,
        mut callback: impl FnMut(GeneratedToken),
    ) -> anyhow::Result<u32> {
        // Stub implementation — generates a placeholder response
        // Real implementation will:
        // 1. llama_decode() the input tokens
        // 2. Loop: llama_sampling_sample() → llama_decode() → callback
        // 3. Return total tokens generated

        // Signal that this is a stub
        let stub_tokens = [
            "This", " response", " is", " from", " the",
            " llama.cpp", " FFI", " stub.", " Real",
            " on-device", " inference", " will", " replace",
            " this", " in", " Phase", " 2",
            " when", " the", " C", " library",
            " is", " linked.", "",
        ];

        let mut n_generated = 0u32;
        for (i, text) in stub_tokens.iter().enumerate() {
            if cancel.is_cancelled() {
                tracing::info!(tokens = n_generated, "Generation cancelled");
                break;
            }

            let is_eos = i == stub_tokens.len() - 1;
            callback(GeneratedToken {
                id: i as i32,
                text: text.to_string(),
                is_eos,
            });
            n_generated += 1;

            if is_eos {
                break;
            }

            // Simulate ~20 tok/s on E2B
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        Ok(n_generated)
    }

    /// Process multimodal input (image/audio) through the mmproj encoder.
    ///
    /// Returns the embedded tokens that should be prepended to the text tokens.
    pub fn encode_multimodal(
        &self,
        _image_data: Option<&[u8]>,
        _audio_data: Option<&[u8]>,
    ) -> anyhow::Result<Vec<i32>> {
        // Stub: Real impl calls llama_clip_free / llama_clip_image_encode
        // and the audio path through the mmproj
        anyhow::bail!("Multimodal encoding requires FFI link (Phase 2)")
    }
}

// ═══════════════════════════════════════════════════════════
//  C FFI declarations (gated behind mobile-native feature)
// ═══════════════════════════════════════════════════════════

/// Raw C FFI declarations for llama.cpp.
///
/// These are only used when compiling with the actual C library linked.
/// The safe wrappers above call these functions.
#[cfg(feature = "mobile-native")]
#[allow(non_camel_case_types, dead_code)]
pub(crate) mod ffi {
    use std::os::raw::{c_char, c_float, c_int, c_void};

    pub type llama_model = c_void;
    pub type llama_context = c_void;
    pub type llama_token = c_int;

    #[repr(C)]
    pub struct llama_model_params {
        pub n_gpu_layers: c_int,
        pub use_mmap: bool,
        pub use_mlock: bool,
    }

    #[repr(C)]
    pub struct llama_context_params {
        pub n_ctx: u32,
        pub n_batch: u32,
        pub n_threads: u32,
        pub n_threads_batch: u32,
    }

    extern "C" {
        pub fn llama_backend_init();
        pub fn llama_backend_free();

        pub fn llama_model_default_params() -> llama_model_params;
        pub fn llama_context_default_params() -> llama_context_params;

        pub fn llama_load_model_from_file(
            path_model: *const c_char,
            params: llama_model_params,
        ) -> *mut llama_model;

        pub fn llama_free_model(model: *mut llama_model);

        pub fn llama_new_context_with_model(
            model: *mut llama_model,
            params: llama_context_params,
        ) -> *mut llama_context;

        pub fn llama_free(ctx: *mut llama_context);

        pub fn llama_n_ctx(ctx: *const llama_context) -> c_int;
        pub fn llama_n_vocab(model: *const llama_model) -> c_int;
        pub fn llama_n_embd(model: *const llama_model) -> c_int;

        pub fn llama_tokenize(
            model: *const llama_model,
            text: *const c_char,
            text_len: c_int,
            tokens: *mut llama_token,
            n_tokens_max: c_int,
            add_special: bool,
            parse_special: bool,
        ) -> c_int;

        pub fn llama_token_to_piece(
            model: *const llama_model,
            token: llama_token,
            buf: *mut c_char,
            length: c_int,
            lstrip: c_int,
            special: bool,
        ) -> c_int;

        pub fn llama_token_eos(model: *const llama_model) -> llama_token;
        pub fn llama_token_bos(model: *const llama_model) -> llama_token;
    }
}


#[cfg(test)]
#[path = "llama_ffi_tests.rs"]
mod tests;
