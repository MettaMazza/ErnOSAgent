// Ern-OS — File write tool

use anyhow::{Context, Result};
use std::path::Path;

pub async fn execute(args: &serde_json::Value) -> Result<String> {
    let path = args["path"].as_str().context("file_write requires 'path'")?;
    let content = args["content"].as_str().context("file_write requires 'content'")?;

    tracing::info!(path = %path, content_len = content.len(), "file_write START");

    if let Some(parent) = Path::new(path).parent() {
        if !parent.exists() {
            tracing::debug!(dir = %parent.display(), "file_write: creating parent directory");
        }
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    match std::fs::write(path, content) {
        Ok(_) => {
            tracing::info!(path = %path, bytes = content.len(), "file_write OK");
            Ok(format!("Written {} bytes to {}", content.len(), path))
        }
        Err(e) => {
            tracing::error!(path = %path, err = %e, "file_write FAILED");
            Err(e).with_context(|| format!("Failed to write file: {}", path))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_write_new_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.txt");
        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": "hello world"
        });
        let result = execute(&args).await.unwrap();
        assert!(result.contains("11 bytes"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");
    }

    #[tokio::test]
    async fn test_write_creates_dirs() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("sub").join("dir").join("file.txt");
        let args = serde_json::json!({
            "path": path.to_str().unwrap(),
            "content": "nested"
        });
        let result = execute(&args).await.unwrap();
        assert!(result.contains("bytes"));
    }
}
