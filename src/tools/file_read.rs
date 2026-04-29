// Ern-OS — File read tool (universal extraction with pagination)
//! Reads and extracts content from any file type, with line-based pagination.
//! Page size is derived from the model's context_length per governance §2.1.

use anyhow::{Context, Result};
use tracing;

/// Derive page size (in characters) from the model's context_length (in tokens).
/// At ~4 chars per token, `context_length / 8` chars ≈ ~12.5% of the model's
/// token budget. This keeps each deep-read summarisation call fast (~8s)
/// and focused, producing higher-quality per-page summaries.
/// For 262K context: 262144 / 8 = 32768 chars per page ≈ 8K tokens.
fn page_size_chars(context_length: usize) -> usize {
    context_length / 8
}

pub async fn execute(args: &serde_json::Value, context_length: usize) -> Result<String> {
    let path = args["path"].as_str().context("file_read requires 'path'")?;
    let start_line = args["start_line"].as_u64().map(|n| n as usize);
    let end_line = args["end_line"].as_u64().map(|n| n as usize);
    tracing::info!(path = %path, start_line = ?start_line, end_line = ?end_line, "file_read START");

    let budget = page_size_chars(context_length);

    // Use universal file extractor for all file types
    match crate::tools::file_extractor::extract(path) {
        Ok(result) => {
            tracing::info!(
                path = %path,
                mime = %result.mime_type,
                len = result.content.len(),
                lang = ?result.language,
                images = result.image_data_urls.len(),
                "file_read OK"
            );

            let mut output = result.content;

            // If there are images, include them for vision
            for url in &result.image_data_urls {
                output.push_str(&format!("\n\n[IMAGE DATA]\n{}", url));
            }

            // Apply pagination if line range is specified
            let paginated = paginate(&output, start_line, end_line, budget);
            Ok(paginated)
        }
        Err(e) => {
            tracing::warn!(path = %path, err = %e, "file_read FAILED");
            Err(e).with_context(|| format!("Failed to read file: {}", path))
        }
    }
}

/// Apply line-based pagination to file content.
fn paginate(content: &str, start_line: Option<usize>, end_line: Option<usize>, budget: usize) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let total_lines = lines.len();

    match (start_line, end_line) {
        (Some(start), end_opt) => paginate_range(&lines, total_lines, start, end_opt, budget),
        (None, _) => paginate_auto(content, &lines, total_lines, budget),
    }
}

/// Paginate with an explicit start line (and optional end line).
fn paginate_range(lines: &[&str], total: usize, start: usize, end_opt: Option<usize>, budget: usize) -> String {
    let start_idx = start.saturating_sub(1);
    if start_idx >= total {
        return format!("[Line {} is past end of file ({} total lines)]", start, total);
    }

    let end_idx = match end_opt {
        Some(end) => end.min(total),
        None => compute_page_end(lines, start_idx, budget),
    };

    let shown = lines[start_idx..end_idx].join("\n");
    format_page(start_idx + 1, end_idx, total, &shown)
}

/// Paginate automatically when no range is given and content exceeds budget.
fn paginate_auto(content: &str, lines: &[&str], total: usize, budget: usize) -> String {
    if content.len() <= budget {
        return content.to_string();
    }

    let last_line = compute_page_end(lines, 0, budget);
    let shown = lines[..last_line].join("\n");
    format_page(1, last_line, total, &shown)
}

/// Calculate how many lines fit in one page starting from `start_idx`.
fn compute_page_end(lines: &[&str], start_idx: usize, budget: usize) -> usize {
    let mut chars = 0;
    let mut last = start_idx;
    for (i, line) in lines[start_idx..].iter().enumerate() {
        chars += line.len() + 1;
        last = start_idx + i + 1;
        if chars >= budget {
            break;
        }
    }
    last.min(lines.len())
}

/// Format a page with header and bookmark (or END OF FILE marker).
fn format_page(start: usize, end: usize, total: usize, content: &str) -> String {
    if end >= total {
        format!("[Lines {}-{} of {} (END OF FILE)]\n{}", start, total, total, content)
    } else {
        format!(
            "[Lines {}-{} of {}]\n{}\n\n[BOOKMARK: line {} — use file_read with start_line={} to continue]",
            start, end, total, content, end + 1, end + 1
        )
    }
}

// ── Auto-stitch: transparent multi-page file reading ─────────────────

/// Parse a `[BOOKMARK: line N — use file_read with start_line=N to continue]`
/// marker from a tool result. Returns the continuation start_line if found.
pub fn parse_bookmark(output: &str) -> Option<usize> {
    // Look for the exact bookmark pattern at the end of the output
    let marker = "[BOOKMARK: line ";
    let pos = output.rfind(marker)?;
    let after_marker = &output[pos + marker.len()..];
    // Extract the number before the " — " separator
    let end = after_marker.find(' ')?;
    after_marker[..end].parse::<usize>().ok()
}

/// Strip the bookmark line from a paginated result, returning just the content.
fn strip_bookmark(output: &str) -> &str {
    match output.rfind("\n\n[BOOKMARK:") {
        Some(pos) => &output[..pos],
        None => output,
    }
}

/// Strip the page header (e.g., `[Lines 101-200 of 500]`) from content.
fn strip_page_header(output: &str) -> &str {
    // Header is always the first line: [Lines X-Y of Z]
    match output.find("]\n") {
        Some(pos) if output.starts_with("[Lines ") => &output[pos + 2..],
        _ => output,
    }
}

/// Auto-stitch a file_read result by detecting [BOOKMARK] markers and
/// transparently fetching subsequent pages. Returns the stitched content.
///
/// This eliminates the need for inference rounds just to continue reading.
/// Max 10 continuations to prevent runaway on very large files.
/// Total stitched content is capped at `page_size_chars(context_length)`.
pub async fn auto_stitch(
    initial_result: &str,
    original_args: &serde_json::Value,
    context_length: usize,
) -> String {
    let total_budget = page_size_chars(context_length);
    let max_continuations = 10;
    let mut stitched = initial_result.to_string();

    for continuation in 0..max_continuations {
        let next_line = match parse_bookmark(&stitched) {
            Some(line) => line,
            None => break, // No bookmark = we have all the content
        };

        // Budget check: stop if we've accumulated enough content
        if stitched.len() >= total_budget {
            tracing::info!(
                len = stitched.len(), budget = total_budget,
                "Auto-stitch: budget reached, stopping"
            );
            break;
        }

        // Build continuation args: same path, new start_line
        let path = match original_args["path"].as_str() {
            Some(p) => p,
            None => break,
        };
        let cont_args = serde_json::json!({
            "path": path,
            "start_line": next_line,
        });

        tracing::debug!(
            path = %path, start_line = next_line, continuation,
            "Auto-stitch: fetching next page"
        );

        match execute(&cont_args, context_length).await {
            Ok(next_page) => {
                // Strip the bookmark from current content, strip header from next page
                let current_content = strip_bookmark(&stitched);
                let next_content = strip_page_header(&next_page);
                stitched = format!("{}\n{}", current_content, next_content);
            }
            Err(e) => {
                tracing::warn!(
                    error = %e, continuation,
                    "Auto-stitch: continuation failed, returning partial content"
                );
                break;
            }
        }
    }

    if stitched.len() > initial_result.len() {
        tracing::info!(
            initial_len = initial_result.len(),
            stitched_len = stitched.len(),
            "Auto-stitch: expanded file_read result"
        );
    }

    stitched
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_read_existing_file() {
        let args = serde_json::json!({"path": "Cargo.toml"});
        let result = execute(&args, 32768).await.unwrap();
        assert!(result.contains("[package]"));
    }

    #[tokio::test]
    async fn test_read_missing_file() {
        let args = serde_json::json!({"path": "/nonexistent/file.txt"});
        assert!(execute(&args, 32768).await.is_err());
    }

    #[test]
    fn test_paginate_small_content() {
        let content = "line one\nline two\nline three";
        let budget = page_size_chars(32768);
        let result = paginate(content, None, None, budget);
        assert_eq!(result, content);
    }

    #[test]
    fn test_paginate_large_content_auto() {
        let budget = page_size_chars(32768);
        let line = "This is a line of content for the pagination test.\n";
        let large = line.repeat(budget / line.len() + 100);
        let result = paginate(&large, None, None, budget);
        assert!(result.contains("[BOOKMARK"));
        assert!(result.contains("start_line="));
        assert!(result.len() < large.len());
    }

    #[test]
    fn test_paginate_explicit_range() {
        let budget = page_size_chars(32768);
        let content = (1..=100).map(|i| format!("Line {}", i)).collect::<Vec<_>>().join("\n");
        let result = paginate(&content, Some(10), Some(20), budget);
        assert!(result.contains("[Lines 10-20 of 100]"));
        assert!(result.contains("Line 10"));
        assert!(result.contains("Line 20"));
        assert!(result.contains("[BOOKMARK: line 21"));
    }

    #[test]
    fn test_paginate_range_at_end() {
        let budget = page_size_chars(32768);
        let content = (1..=10).map(|i| format!("Line {}", i)).collect::<Vec<_>>().join("\n");
        let result = paginate(&content, Some(8), Some(10), budget);
        assert!(result.contains("END OF FILE"));
        assert!(!result.contains("[BOOKMARK"));
    }

    #[test]
    fn test_paginate_past_end() {
        let budget = page_size_chars(32768);
        let content = "one\ntwo\nthree";
        let result = paginate(content, Some(999), None, budget);
        assert!(result.contains("past end of file"));
    }

    #[test]
    fn test_page_size_scales_with_context() {
        let small = page_size_chars(8192);
        let large = page_size_chars(131072);
        assert!(large > small);
        assert!(small > 0);
    }

    #[test]
    fn test_page_size_is_context_length() {
        assert_eq!(page_size_chars(262144), 32768);  // 262144 / 8
        assert_eq!(page_size_chars(32768), 4096);    // 32768 / 8
    }

    #[test]
    fn test_parse_bookmark_found() {
        let output = "[Lines 1-100 of 500]\ncontent here\n\n\
                      [BOOKMARK: line 101 — use file_read with start_line=101 to continue]";
        assert_eq!(parse_bookmark(output), Some(101));
    }

    #[test]
    fn test_parse_bookmark_not_found() {
        let output = "[Lines 1-10 of 10 (END OF FILE)]\ncontent here";
        assert_eq!(parse_bookmark(output), None);
    }

    #[test]
    fn test_strip_bookmark() {
        let output = "[Lines 1-100 of 500]\ncontent\n\n\
                      [BOOKMARK: line 101 — use file_read with start_line=101 to continue]";
        assert_eq!(strip_bookmark(output), "[Lines 1-100 of 500]\ncontent");
    }

    #[test]
    fn test_strip_page_header() {
        let output = "[Lines 101-200 of 500]\nactual content\nmore content";
        assert_eq!(strip_page_header(output), "actual content\nmore content");
    }

    #[test]
    fn test_strip_page_header_no_header() {
        let output = "just plain content";
        assert_eq!(strip_page_header(output), "just plain content");
    }

    #[tokio::test]
    async fn test_auto_stitch_no_bookmark() {
        // Small file — no bookmark, auto_stitch should return unchanged
        let args = serde_json::json!({"path": "Cargo.toml"});
        let result = execute(&args, 262144).await.unwrap();
        let stitched = auto_stitch(&result, &args, 262144).await;
        assert_eq!(stitched, result);
    }
}

