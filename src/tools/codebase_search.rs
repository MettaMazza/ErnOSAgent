// Ern-OS — Codebase search tool — recursive directory search

use anyhow::{Context, Result};
use std::path::Path;

pub async fn execute(args: &serde_json::Value) -> Result<String> {
    let query = args["query"].as_str().context("codebase_search requires 'query'")?;
    let path = args["path"].as_str().unwrap_or(".");
    let max_results = args["max_results"].as_u64().unwrap_or(20) as usize;

    tracing::info!(query = %query, path = %path, max_results = max_results, "codebase_search START");
    let start = std::time::Instant::now();

    let root = Path::new(path);
    if !root.exists() {
        tracing::warn!(path = %path, "codebase_search: path does not exist");
        anyhow::bail!("Path does not exist: {}", path);
    }

    let mut results = Vec::new();
    search_recursive(root, query, &mut results, max_results)?;

    let elapsed_ms = start.elapsed().as_millis() as u64;
    tracing::info!(query = %query, matches = results.len(), elapsed_ms = elapsed_ms, "codebase_search COMPLETE");

    if results.is_empty() {
        return Ok(format!("No matches found for '{}' in {}", query, path));
    }
    Ok(results.join("\n"))
}

fn search_recursive(
    dir: &Path, query: &str, results: &mut Vec<String>, max: usize,
) -> Result<()> {
    if results.len() >= max { return Ok(()); }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Ok(()),
    };

    for entry in entries.flatten() {
        if results.len() >= max { break; }
        let path = entry.path();
        let name = path.file_name().unwrap_or_default().to_string_lossy();

        // Skip hidden dirs and common noise
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }

        if path.is_dir() {
            search_recursive(&path, query, results, max)?;
        } else if is_searchable(&path) {
            search_file(&path, query, results, max)?;
        }
    }
    Ok(())
}

fn search_file(
    path: &Path, query: &str, results: &mut Vec<String>, max: usize,
) -> Result<()> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return Ok(()),
    };
    let query_lower = query.to_lowercase();
    for (i, line) in content.lines().enumerate() {
        if results.len() >= max { break; }
        if line.to_lowercase().contains(&query_lower) {
            results.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
        }
    }
    Ok(())
}

fn is_searchable(path: &Path) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(ext,
        "rs" | "toml" | "md" | "txt" | "json" | "yaml" | "yml" |
        "js" | "ts" | "css" | "html" | "py" | "sh" | "cfg" | "conf"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_search_current_dir() {
        let args = serde_json::json!({"query": "fn main", "path": ".", "max_results": 5});
        let result = execute(&args).await.unwrap();
        assert!(!result.is_empty());
    }

    #[tokio::test]
    async fn test_search_no_match() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("test.rs"), "fn hello() {}").unwrap();
        let args = serde_json::json!({
            "query": "ZZZZUNIQUEZZZZNOTEXIST",
            "path": tmp.path().to_str().unwrap()
        });
        let result = execute(&args).await.unwrap();
        assert!(result.contains("No matches"));
    }
}
