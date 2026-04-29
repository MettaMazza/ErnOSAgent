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
            let snippet = extract_match_context(line, &query_lower, 200);
            results.push(format!("{}:{}: {}", path.display(), i + 1, snippet));
        }
    }
    Ok(())
}

/// Extract a context snippet around the first match occurrence.
/// Shows up to `context_chars` before and after the match.
/// Search tools show context, not full line dumps — full content is available via file_read.
fn extract_match_context(line: &str, query_lower: &str, context_chars: usize) -> String {
    let trimmed = line.trim();
    if trimmed.len() <= context_chars * 2 + query_lower.len() {
        return trimmed.to_string();
    }
    let line_lower = trimmed.to_lowercase();
    let match_pos = line_lower.find(query_lower).unwrap_or(0);
    let start = match_pos.saturating_sub(context_chars);
    let end = (match_pos + query_lower.len() + context_chars).min(trimmed.len());

    // Align to char boundaries
    let start = trimmed.floor_char_boundary(start);
    let end = trimmed.ceil_char_boundary(end);

    let prefix = if start > 0 { "…" } else { "" };
    let suffix = if end < trimmed.len() { "…" } else { "" };
    format!("{}{}{}", prefix, &trimmed[start..end], suffix)
}

fn is_searchable(path: &Path) -> bool {
    // Skip runtime data artifacts — not source code.
    if is_data_artifact(path) {
        return false;
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    matches!(ext,
        "rs" | "toml" | "md" | "txt" | "json" | "yaml" | "yml" |
        "js" | "ts" | "css" | "html" | "py" | "sh" | "cfg" | "conf"
    )
}

/// Returns true for files that are runtime data artifacts (not source code).
/// These are large binary-equivalent files (embeddings, training buffers, etc.)
/// that should not appear in code search results.
fn is_data_artifact(path: &Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    matches!(name,
        "embeddings.json" | "golden_buffer.jsonl" | "rejection_buffer.jsonl" |
        "quarantine.json" | "review_deck.json" | "training_manifest.json"
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

    #[test]
    fn test_extract_match_context_short_line() {
        let result = extract_match_context("fn hello_world() {}", "hello", 200);
        assert_eq!(result, "fn hello_world() {}");
    }

    #[test]
    fn test_extract_match_context_long_line() {
        let long = "a".repeat(1000) + "NEEDLE" + &"b".repeat(1000);
        let result = extract_match_context(&long, "needle", 200);
        assert!(result.starts_with('…'));
        assert!(result.ends_with('…'));
        assert!(result.contains("NEEDLE"));
        assert!(result.len() < 600); // 200 + 6 + 200 + ellipsis overhead
    }

    #[test]
    fn test_search_file_caps_long_lines() {
        let tmp = tempfile::TempDir::new().unwrap();
        let big_line = "x".repeat(500_000) + "TARGET_MATCH" + &"y".repeat(500_000);
        std::fs::write(tmp.path().join("big.rs"), &big_line).unwrap();
        let mut results = Vec::new();
        search_file(&tmp.path().join("big.rs"), "target_match", &mut results, 5).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].len() < 1000, "Result should be bounded, got {} bytes", results[0].len());
        assert!(results[0].contains("TARGET_MATCH"));
    }

    #[test]
    fn test_is_data_artifact_embeddings() {
        assert!(is_data_artifact(Path::new("data/embeddings.json")));
        assert!(is_data_artifact(Path::new("golden_buffer.jsonl")));
    }

    #[test]
    fn test_is_data_artifact_source_code() {
        assert!(!is_data_artifact(Path::new("src/main.rs")));
        assert!(!is_data_artifact(Path::new("Cargo.toml")));
        assert!(!is_data_artifact(Path::new("package.json")));
    }
}
