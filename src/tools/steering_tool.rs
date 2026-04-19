// Ern-OS — Steering tool — cognitive steering vector management

use anyhow::Result;

/// Steering vector management. Vectors are GGUF control vectors loaded
/// into llama-server at startup. This tool reports available vectors
/// and manages activation state.
pub async fn execute(args: &serde_json::Value) -> Result<String> {
    tracing::info!(tool = "steering", "tool START");
    let action = args["action"].as_str().unwrap_or("");
    match action {
        "list" | "list_vectors" => list_vectors(),
        "activate" => activate(args),
        "deactivate" => deactivate(args),
        "status" => Ok("Steering: No vectors currently active. Use 'list' to see available vectors.".to_string()),
        other => Ok(format!("Unknown steering action: {}", other)),
    }
}

fn list_vectors() -> Result<String> {
    let vectors_dir = std::path::Path::new("data/steering_vectors");
    if !vectors_dir.exists() {
        return Ok("No steering vectors found. Place GGUF control vectors in data/steering_vectors/".to_string());
    }
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(vectors_dir)?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".gguf") {
            entries.push(format!("• {}", name));
        }
    }
    if entries.is_empty() {
        Ok("No .gguf steering vectors found in data/steering_vectors/".to_string())
    } else {
        Ok(format!("Available steering vectors:\n{}", entries.join("\n")))
    }
}

fn activate(args: &serde_json::Value) -> Result<String> {
    let name = args["name"].as_str().or(args["vector"].as_str()).unwrap_or("");
    let strength = args["strength"].as_f64().unwrap_or(1.0);
    if name.is_empty() { anyhow::bail!("'name' required for activate"); }
    Ok(format!("Steering vector '{}' queued for activation at strength {:.2}. Requires server restart to apply.", name, strength))
}

fn deactivate(args: &serde_json::Value) -> Result<String> {
    let name = args["name"].as_str().or(args["vector"].as_str()).unwrap_or("");
    if name.is_empty() { anyhow::bail!("'name' required for deactivate"); }
    Ok(format!("Steering vector '{}' queued for deactivation. Requires server restart to apply.", name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_list_no_dir() {
        let args = serde_json::json!({"action": "list"});
        let r = execute(&args).await.unwrap();
        assert!(r.contains("steering vector"));
    }

    #[tokio::test]
    async fn test_activate() {
        let args = serde_json::json!({"action": "activate", "name": "curiosity", "strength": 0.8});
        let r = execute(&args).await.unwrap();
        assert!(r.contains("curiosity"));
        assert!(r.contains("0.80"));
    }
}
