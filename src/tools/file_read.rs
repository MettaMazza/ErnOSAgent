// Ern-OS — File read tool

use anyhow::{Context, Result};
use tracing;

pub async fn execute(args: &serde_json::Value) -> Result<String> {
    let path = args["path"].as_str().context("file_read requires 'path'")?;
    tracing::info!(path = %path, "file_read START");
    match std::fs::read_to_string(path) {
        Ok(content) => {
            tracing::info!(path = %path, bytes = content.len(), "file_read OK");
            Ok(content)
        }
        Err(e) => {
            tracing::warn!(path = %path, err = %e, "file_read FAILED");
            Err(e).with_context(|| format!("Failed to read file: {}", path))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_read_existing_file() {
        let args = serde_json::json!({"path": "Cargo.toml"});
        let result = execute(&args).await.unwrap();
        assert!(result.contains("[package]"));
    }

    #[tokio::test]
    async fn test_read_missing_file() {
        let args = serde_json::json!({"path": "/nonexistent/file.txt"});
        assert!(execute(&args).await.is_err());
    }
}
