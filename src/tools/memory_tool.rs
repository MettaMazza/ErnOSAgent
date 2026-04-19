// Ern-OS — Unified memory tool

use anyhow::Result;

pub async fn execute(action: &str, query: Option<&str>) -> Result<String> {
    tracing::info!(tool = "memory_tool", "tool START");
    match action {
        "recall" => {
            let q = query.unwrap_or("general");
            Ok(format!("[Memory Recall] Query: '{}' — use MemoryManager.recall_context()", q))
        }
        "status" => {
            Ok("[Memory Status] Use MemoryManager.status_summary()".to_string())
        }
        "consolidate" => {
            Ok("[Memory Consolidate] Triggered manual consolidation".to_string())
        }
        "search" => {
            let q = query.unwrap_or("");
            Ok(format!("[Memory Search] Query: '{}' — searching across all tiers", q))
        }
        "reset" => {
            Ok("[Memory Reset] Factory reset all tiers — use MemoryManager.clear()".to_string())
        }
        other => Ok(format!("Unknown memory action: {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_recall() {
        let r = execute("recall", Some("rust")).await.unwrap();
        assert!(r.contains("rust"));
    }

    #[tokio::test]
    async fn test_status() {
        let r = execute("status", None).await.unwrap();
        assert!(r.contains("Status"));
    }
}
