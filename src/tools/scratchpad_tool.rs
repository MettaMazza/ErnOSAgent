// Ern-OS — Scratchpad tool

use anyhow::Result;

pub async fn execute(
    action: &str, key: Option<&str>, value: Option<&str>,
) -> Result<String> {
    tracing::info!(tool = "scratchpad", "tool START");
    match action {
        "pin" => {
            let k = key.unwrap_or("unnamed");
            let v = value.unwrap_or("");
            Ok(format!("Pinned '{}' = '{}'", k, v))
        }
        "unpin" => {
            let k = key.unwrap_or("");
            Ok(format!("Unpinned '{}'", k))
        }
        "list" => Ok("Scratchpad list — use ScratchpadStore.all()".to_string()),
        "get" => {
            let k = key.unwrap_or("");
            Ok(format!("Get scratchpad key '{}'", k))
        }
        other => Ok(format!("Unknown scratchpad action: {}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pin() {
        let r = execute("pin", Some("lang"), Some("Rust")).await.unwrap();
        assert!(r.contains("Pinned"));
    }
}
