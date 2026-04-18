// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Sleep Metal — macOS Metal resource management for training.
//!
//! On Apple Silicon, the inference model (llama.cpp/Ollama) holds IOSurface
//! handles on the Metal GPU. These must be released before training to avoid
//! competition for GPU memory. After training, the model is reloaded.
//!
//! On non-macOS platforms, these are no-ops.

/// Unload the inference model to free Metal GPU memory before training.
///
/// On macOS, sends an unload request to the local Ollama/llama-server instance.
/// On other platforms, this is a no-op.
pub async fn unload_inference_model() {
    #[cfg(target_os = "macos")]
    {
        tracing::info!("Unloading inference model to free Metal resources for training");
        if let Err(e) = unload_ollama().await {
            tracing::warn!(error = %e, "Failed to unload Ollama model — training may be slower");
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        tracing::debug!("Metal unload: no-op on non-macOS platform");
    }
}

/// Reload the inference model after training completes.
///
/// On macOS, the model will be lazily reloaded on the next inference request.
/// We just log the intent here.
pub async fn reload_inference_model() {
    #[cfg(target_os = "macos")]
    {
        tracing::info!("Training complete — inference model will reload on next request");
    }

    #[cfg(not(target_os = "macos"))]
    {
        tracing::debug!("Metal reload: no-op on non-macOS platform");
    }
}

/// Send an unload request to the local Ollama instance.
#[cfg(target_os = "macos")]
async fn unload_ollama() -> anyhow::Result<()> {
    let ollama_host =
        std::env::var("OLLAMA_HOST").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());

    let client = reqwest::Client::new();

    // Ollama unloads a model by sending a generate request with keep_alive=0
    let response = client
        .post(format!("{ollama_host}/api/generate"))
        .json(&serde_json::json!({
            "model": "",
            "keep_alive": 0
        }))
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await;

    match response {
        Ok(resp) if resp.status().is_success() => {
            tracing::info!("Ollama model unloaded successfully");
            Ok(())
        }
        Ok(resp) => {
            // Non-success but we tried — not critical
            tracing::debug!(status = %resp.status(), "Ollama unload returned non-success (may not have a model loaded)");
            Ok(())
        }
        Err(e) => {
            // Ollama might not be running — that's fine, llama-server is the other option
            tracing::debug!(error = %e, "Ollama not reachable for unload (may be using llama-server)");
            Ok(())
        }
    }
}
