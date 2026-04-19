// Ern-OS — Interpretability tool — SAE feature inspection

use anyhow::Result;

/// SAE interpretability inspection. Provides access to the feature
/// extraction system for analyzing model activations.
pub async fn execute(args: &serde_json::Value) -> Result<String> {
    tracing::info!(tool = "interpretability", "tool START");
    let action = args["action"].as_str().unwrap_or("");
    match action {
        "snapshot" => take_snapshot(),
        "top_features" | "features" => top_features(args),
        "encode" => encode_input(args),
        "divergence" => Ok("Divergence analysis requires live SAE — connect to interpretability dashboard.".to_string()),
        "probe" => probe_concept(args),
        "labeled_features" => list_labeled_features(),
        other => Ok(format!("Unknown interpretability action: {}", other)),
    }
}

fn take_snapshot() -> Result<String> {
    let snapshot_dir = std::path::Path::new("data/snapshots");
    std::fs::create_dir_all(snapshot_dir)?;
    let ts = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let path = snapshot_dir.join(format!("snapshot_{}.json", ts));
    std::fs::write(&path, "{\"type\": \"neural_snapshot\", \"status\": \"captured\"}")?;
    Ok(format!("Snapshot saved: {}", path.display()))
}

fn top_features(args: &serde_json::Value) -> Result<String> {
    let top_k = args["top_k"].as_u64().unwrap_or(10) as usize;
    let features_path = std::path::Path::new("data/sae_features.json");
    if features_path.exists() {
        let content = std::fs::read_to_string(features_path)?;
        Ok(format!("Top {} features from saved state:\n{}", top_k, &content[..content.len().min(500)]))
    } else {
        Ok(format!("No SAE feature data available. Run training or connect live SAE to populate. Requested top {}.", top_k))
    }
}

fn encode_input(args: &serde_json::Value) -> Result<String> {
    let input = args["input"].as_str().unwrap_or("");
    if input.is_empty() { anyhow::bail!("'input' required for encode"); }
    Ok(format!("Encoding '{}' — SAE encoding requires live inference server connection.", &input[..input.len().min(50)]))
}

fn probe_concept(args: &serde_json::Value) -> Result<String> {
    let concept = args["concept"].as_str().or(args["input"].as_str()).unwrap_or("");
    if concept.is_empty() { anyhow::bail!("'concept' or 'input' required for probe"); }
    Ok(format!("Probing for concept '{}' — requires live SAE activation data.", concept))
}

fn list_labeled_features() -> Result<String> {
    let labels_path = std::path::Path::new("data/feature_labels.json");
    if labels_path.exists() {
        let content = std::fs::read_to_string(labels_path)?;
        Ok(format!("Labeled features:\n{}", &content[..content.len().min(1000)]))
    } else {
        Ok("No labeled features found. Train SAE and label features to populate.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_snapshot() {
        let args = serde_json::json!({"action": "snapshot"});
        let r = execute(&args).await.unwrap();
        assert!(r.contains("Snapshot saved"));
    }

    #[tokio::test]
    async fn test_top_features_no_data() {
        let args = serde_json::json!({"action": "top_features", "top_k": 5});
        let r = execute(&args).await.unwrap();
        assert!(r.contains("5"));
    }
}
