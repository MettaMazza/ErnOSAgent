//! Artifact tool — create persistent markdown documents rendered as interactive cards.
//! Artifacts are reports, plans, analyses, or code references that persist beyond chat.

use anyhow::{Context, Result};

/// Create and persist a markdown artifact.
pub async fn execute(args: &serde_json::Value) -> Result<String> {
    let title = args["title"].as_str().unwrap_or("Untitled");
    let content = args["content"].as_str().unwrap_or("");
    let artifact_type = args["artifact_type"].as_str().unwrap_or("report");

    if content.is_empty() {
        anyhow::bail!("Artifact content cannot be empty");
    }

    let id = uuid::Uuid::new_v4().to_string();
    let dir = std::path::PathBuf::from("data/artifacts");
    std::fs::create_dir_all(&dir)
        .context("Failed to create artifacts directory")?;

    let artifact = serde_json::json!({
        "id": id,
        "title": title,
        "artifact_type": artifact_type,
        "content": content,
        "created_at": chrono::Utc::now().to_rfc3339(),
    });

    let path = dir.join(format!("{}.json", id));
    std::fs::write(&path, serde_json::to_string_pretty(&artifact)?)
        .with_context(|| format!("Failed to write artifact to {:?}", path))?;

    tracing::info!(
        artifact_id = %id,
        title = %title,
        artifact_type = %artifact_type,
        content_len = content.len(),
        "Artifact created"
    );

    Ok(serde_json::to_string(&artifact)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_artifact() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join("data/artifacts");
        std::fs::create_dir_all(&dir).unwrap();

        // Override data dir via env for test isolation
        let args = serde_json::json!({
            "title": "Test Report",
            "content": "# Heading\n\nSome analysis.",
            "artifact_type": "report"
        });
        let result = execute(&args).await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed["title"], "Test Report");
        assert_eq!(parsed["artifact_type"], "report");
        assert!(!parsed["id"].as_str().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_empty_content_rejected() {
        let args = serde_json::json!({
            "title": "Empty",
            "content": "",
        });
        assert!(execute(&args).await.is_err());
    }
}
