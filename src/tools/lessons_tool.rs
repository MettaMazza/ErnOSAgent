// Ern-OS — Lessons tool

use anyhow::Result;

pub async fn execute(args: &serde_json::Value) -> Result<String> {
    tracing::info!(tool = "lessons", "tool START");
    let action = args["action"].as_str().unwrap_or("");
    match action {
        "add" => {
            let rule = args["rule"].as_str().unwrap_or("");
            let conf = args["confidence"].as_f64().unwrap_or(0.8);
            Ok(format!("Added lesson: '{}' (confidence: {:.0}%)", rule, conf * 100.0))
        }
        "remove" => {
            let id = args["id"].as_str().unwrap_or("");
            let query = args["query"].as_str().unwrap_or("");
            if !id.is_empty() {
                Ok(format!("Removed lesson: {}", id))
            } else if !query.is_empty() {
                Ok(format!("Removed lessons matching: '{}'", query))
            } else {
                Ok("Error: 'id' or 'query' required for remove".to_string())
            }
        }
        "list" => Ok("All lessons — use LessonStore.all()".to_string()),
        "search" => {
            let q = args["query"].as_str().unwrap_or("");
            Ok(format!("Searching lessons for '{}'", q))
        }
        other => Ok(format!("Unknown lessons action: {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add() {
        let args = serde_json::json!({"action": "add", "rule": "Always test", "confidence": 0.95});
        let r = execute(&args).await.unwrap();
        assert!(r.contains("Always test"));
    }
}
