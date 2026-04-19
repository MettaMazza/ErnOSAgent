// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Embedding server management — dedicated llama-server for embeddings.

use anyhow::Result;
use std::process::Child;

/// Manages a dedicated llama-server instance for embeddings.
pub struct EmbeddingServer {
    port: u16,
    model_path: String,
    process: Option<Child>,
}

impl EmbeddingServer {
    pub fn new(port: u16, model_path: &str) -> Self {
        Self {
            port,
            model_path: model_path.to_string(),
            process: None,
        }
    }

    /// Start the embedding server as a subprocess.
    pub fn start(&mut self, server_binary: &str) -> Result<()> {
        let child = std::process::Command::new(server_binary)
            .args([
                "--model",
                &self.model_path,
                "--port",
                &self.port.to_string(),
                "--embedding",
                "--pooling",
                "mean",
                "-c",
                "0", // Auto-detect context from GGUF — never hardcode (Rule 2.1)
                "-ngl",
                "999",
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()?;

        tracing::info!(
            port = self.port,
            model = %self.model_path,
            pid = child.id(),
            "Embedding server started"
        );

        self.process = Some(child);
        Ok(())
    }

    /// Stop the embedding server subprocess.
    pub fn stop(&mut self) {
        if let Some(ref mut process) = self.process {
            let _ = process.kill();
            let _ = process.wait();
            tracing::info!(port = self.port, "Embedding server stopped");
        }
        self.process = None;
    }

    /// Check if the embedding server process is still running.
    pub fn is_running(&mut self) -> bool {
        match &mut self.process {
            Some(p) => p.try_wait().ok().flatten().is_none(),
            None => false,
        }
    }

    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for EmbeddingServer {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_embedding_server() {
        let server = EmbeddingServer::new(8081, "test.gguf");
        assert_eq!(server.port(), 8081);
    }

    #[test]
    fn test_not_running_initially() {
        let mut server = EmbeddingServer::new(8081, "test.gguf");
        assert!(!server.is_running());
    }
}
