// Ern-OS — Timeline tool

use anyhow::Result;

pub async fn execute(args: &serde_json::Value) -> Result<String> {
    tracing::info!(tool = "timeline", "tool START");
    let action = args["action"].as_str().unwrap_or("");
    match action {
        "recent" => {
            let n = args["limit"].as_u64().unwrap_or(10);
            Ok(format!("Recent {} timeline entries", n))
        }
        "search" => {
            let q = args["query"].as_str().unwrap_or("");
            Ok(format!("Searching timeline for '{}'", q))
        }
        "session" => {
            let sid = args["session_id"].as_str().unwrap_or("");
            Ok(format!("Timeline entries for session '{}'", sid))
        }
        other => Ok(format!("Unknown timeline action: {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_recent() {
        let args = serde_json::json!({"action": "recent", "limit": 5});
        let r = execute(&args).await.unwrap();
        assert!(r.contains("5"));
    }
}
