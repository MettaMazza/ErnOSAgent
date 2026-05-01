//! Context budget management for platform execution.
//!
//! Handles context window enforcement, tool result compression, and
//! message accounting for the L1/L2 tool execution pipelines.

/// Append tool call and result messages to the conversation.
pub fn append_tool_messages(
    messages: &mut Vec<crate::provider::Message>,
    tc: &crate::tools::schema::ToolCall,
    result: &crate::tools::schema::ToolResult,
) {
    messages.push(crate::provider::Message::assistant_tool_call(
        &tc.id, &tc.name, &tc.arguments,
    ));
    messages.push(crate::provider::Message::tool_result(&tc.id, &result.output));
}

/// Ensure total context fits within the model's context window.
/// Trims tool result messages oldest-first until the total fits.
/// Old tool results are dead weight — the model already processed them.
///
/// Uses a 60% safety margin: the `chars / 2` heuristic can underestimate
/// by 1.5x for short-line content (directory listings, line-per-entry output).
/// Trimming to 60% of context_length guarantees actual tokens stay within
/// budget even with worst-case estimation error.
pub fn enforce_context_budget(
    messages: &mut Vec<crate::provider::Message>,
    context_length: usize,
) {
    let total_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
    let estimated_tokens = total_chars / 2 + 2000;
    let budget = (context_length as f64 * 0.60) as usize;

    if estimated_tokens <= budget {
        return;
    }

    tracing::warn!(
        estimated_tokens,
        budget,
        context_length,
        overshoot = estimated_tokens - budget,
        "Context budget exceeded (60% margin) — trimming tool results"
    );

    let tool_indices = find_trim_candidates(messages);
    let trimmed_total = trim_tool_messages(messages, &tool_indices, budget);

    let final_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
    tracing::warn!(
        trimmed_total,
        final_estimated_tokens = final_chars / 2 + 2000,
        budget,
        context_length,
        "Context budget enforcement complete"
    );
}

/// Collect indices of tool messages, oldest first.
fn find_trim_candidates(messages: &[crate::provider::Message]) -> Vec<usize> {
    messages.iter().enumerate()
        .filter(|(_, m)| m.role == "tool")
        .map(|(i, _)| i)
        .collect()
}

/// Trim tool messages from oldest to newest until context fits.
fn trim_tool_messages(
    messages: &mut Vec<crate::provider::Message>,
    tool_indices: &[usize],
    context_length: usize,
) -> usize {
    let mut trimmed_total = 0usize;

    for &idx in tool_indices {
        let total_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
        let estimated = total_chars / 2 + 2000;
        if estimated <= context_length {
            break;
        }

        let content = messages[idx].text_content();
        let content_len = content.len();

        // For older tool results (not the last one), compress
        if idx != *tool_indices.last().unwrap_or(&usize::MAX) {
            let compressed = compress_tool_result(&content);
            let saved = content_len.saturating_sub(compressed.len());
            messages[idx].content = serde_json::Value::String(compressed);
            trimmed_total += saved;
            tracing::info!(idx, content_len, compressed_len = content_len - saved, "Compressed old tool result");
            continue;
        }

        // For the most recent tool result, trim with a bookmark
        let overshoot_now = estimated - context_length;
        let trim_chars = overshoot_now * 4 + 2000;
        if content_len > trim_chars + 500 {
            let keep = content_len - trim_chars;
            let truncated = match content[..keep].rfind('\n') {
                Some(pos) => &content[..pos],
                None => &content[..keep],
            };
            let shown_lines = truncated.lines().count();
            let total_lines = content.lines().count();
            let new_content = format!(
                "[Lines 1-{} of {} — trimmed to fit context window]\n{}\n\n[BOOKMARK: line {} — use file_read with start_line={} to continue]",
                shown_lines, total_lines, truncated, shown_lines + 1, shown_lines + 1
            );
            trimmed_total += content_len - new_content.len();
            messages[idx].content = serde_json::Value::String(new_content);
            tracing::info!(idx, shown_lines, total_lines, "Trimmed latest tool result with bookmark");
        }
    }

    trimmed_total
}

/// Compress a tool result to preserve key content while reducing size.
/// Keeps pagination markers, head/tail ~2000 chars, and section headings.
fn compress_tool_result(content: &str) -> String {
    let total_lines = content.lines().count();
    let total_chars = content.len();

    if total_chars <= 8000 {
        return content.to_string();
    }

    let (pagination_header, bookmark) = extract_pagination_markers(content);
    let inner = strip_pagination(content, &pagination_header);
    let head = extract_head(inner, 2000);
    let tail = extract_tail(inner, &bookmark, 2000);
    let headings = extract_middle_headings(inner, head, tail);

    build_compressed_output(
        &pagination_header, &bookmark,
        head, tail, &headings,
        total_chars, total_lines,
        inner.lines().count(),
    )
}

/// Extract pagination header and bookmark from content.
fn extract_pagination_markers(content: &str) -> (String, String) {
    let first_line = content.lines().next().unwrap_or("");
    let header = if first_line.starts_with("[Lines ") || first_line.starts_with("[FILE SAVED") {
        first_line.to_string()
    } else {
        String::new()
    };

    let last_line = content.lines().last().unwrap_or("");
    let bookmark = if last_line.contains("[BOOKMARK:") || last_line.contains("END OF FILE") {
        last_line.to_string()
    } else {
        String::new()
    };

    (header, bookmark)
}

/// Strip pagination header from content to get inner body.
fn strip_pagination<'a>(content: &'a str, header: &str) -> &'a str {
    if !header.is_empty() {
        let skip = header.len() + 1;
        &content[skip.min(content.len())..]
    } else {
        content
    }
}

/// Extract first ~N chars, aligned to line boundary.
fn extract_head(inner: &str, max_chars: usize) -> &str {
    let head_end = inner.char_indices()
        .take_while(|(i, _)| *i < max_chars)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(inner.len().min(max_chars));
    match inner[..head_end].rfind('\n') {
        Some(pos) => &inner[..pos],
        None => &inner[..head_end],
    }
}

/// Extract last ~N chars, aligned to line boundary.
fn extract_tail<'a>(inner: &'a str, bookmark: &str, max_chars: usize) -> &'a str {
    let inner_for_tail = if !bookmark.is_empty() {
        let end = inner.len().saturating_sub(bookmark.len() + 1);
        &inner[..end]
    } else {
        inner
    };
    let tail_start = inner_for_tail.len().saturating_sub(max_chars);
    match inner_for_tail[tail_start..].find('\n') {
        Some(pos) => &inner_for_tail[tail_start + pos + 1..],
        None => &inner_for_tail[tail_start..],
    }
}

/// Extract section headings from the middle region.
fn extract_middle_headings<'a>(inner: &'a str, head: &str, tail: &str) -> Vec<&'a str> {
    let head_lines = head.lines().count();
    let inner_total_lines = inner.lines().count();
    let tail_line_start = inner_total_lines.saturating_sub(tail.lines().count());

    inner.lines()
        .enumerate()
        .filter(|(i, _)| *i >= head_lines && *i < tail_line_start)
        .filter(|(_, line)| {
            let trimmed = line.trim();
            trimmed.starts_with('#') || trimmed.starts_with("---") || trimmed.starts_with("***")
        })
        .map(|(_, line)| line)
        .collect()
}

/// Assemble the compressed output.
fn build_compressed_output(
    pagination_header: &str, bookmark: &str,
    head: &str, tail: &str, headings: &[&str],
    total_chars: usize, total_lines: usize,
    inner_total_lines: usize,
) -> String {
    let head_lines = head.lines().count();
    let tail_line_start = inner_total_lines.saturating_sub(tail.lines().count());

    let headings_section = if headings.is_empty() {
        String::new()
    } else {
        format!("\n\n[Section headings from compressed region:]\n{}", headings.join("\n"))
    };

    let mut output = String::new();
    if !pagination_header.is_empty() {
        output.push_str(pagination_header);
        output.push('\n');
    }

    output.push_str(&format!(
        "[COMPRESSED — {} total chars, {} lines — reading position preserved]\n\
         \n--- BEGIN (lines 1-{}) ---\n{}\n--- END OF HEAD ---\n\
         \n[... {} chars / {} lines compressed ...]{}\n\
         \n--- TAIL (lines {}-{}) ---\n{}\n--- END ---",
        total_chars, total_lines,
        head_lines, head,
        total_chars - head.len() - tail.len(),
        tail_line_start - head_lines,
        headings_section,
        tail_line_start, inner_total_lines, tail,
    ));

    if !bookmark.is_empty() {
        output.push('\n');
        output.push_str(bookmark);
    }

    output
}

/// Build tool context summary for observer audit.
pub fn build_tool_context(messages: &[crate::provider::Message]) -> String {
    let mut entries: Vec<String> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == "tool" {
            let tool_name = find_tool_name(messages, i);
            let result_text = msg.text_content();
            entries.push(format!("[{}] {} → {}", entries.len() + 1, tool_name, result_text));
        }
    }
    if entries.is_empty() {
        String::new()
    } else {
        format!("Platform tools executed ({} calls):\n{}", entries.len(), entries.join("\n"))
    }
}

/// Look backwards through messages to find the tool name for a tool result.
fn find_tool_name(messages: &[crate::provider::Message], tool_msg_idx: usize) -> String {
    let tool_call_id = messages[tool_msg_idx].tool_call_id.as_deref().unwrap_or("");
    for j in (0..tool_msg_idx).rev() {
        if messages[j].role == "assistant" {
            if let Some(tcs) = &messages[j].tool_calls {
                for tc in tcs {
                    if tc["id"].as_str() == Some(tool_call_id) {
                        return tc["function"]["name"]
                            .as_str()
                            .unwrap_or("unknown")
                            .to_string();
                    }
                }
            }
        }
    }
    "unknown".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_tool_context_empty() {
        let messages: Vec<crate::provider::Message> = Vec::new();
        assert!(build_tool_context(&messages).is_empty());
    }

    #[test]
    fn test_find_tool_name_no_match() {
        let messages = vec![
            crate::provider::Message::text("user", "hello"),
        ];
        assert_eq!(find_tool_name(&messages, 0), "unknown");
    }

    #[test]
    fn test_extract_pagination_markers() {
        let content = "[Lines 1-100 of 500]\ncontent\n[BOOKMARK: line 101 — continue]";
        let (header, bookmark) = extract_pagination_markers(content);
        assert_eq!(header, "[Lines 1-100 of 500]");
        assert!(bookmark.contains("[BOOKMARK:"));
    }

    #[test]
    fn test_extract_pagination_markers_none() {
        let content = "just plain content\nno markers here";
        let (header, bookmark) = extract_pagination_markers(content);
        assert!(header.is_empty());
        assert!(bookmark.is_empty());
    }

    #[test]
    fn test_compress_small_content() {
        let content = "small content under 8000 chars";
        assert_eq!(compress_tool_result(content), content);
    }

    #[test]
    fn test_find_trim_candidates() {
        let messages = vec![
            crate::provider::Message::text("user", "hello"),
            crate::provider::Message::tool_result("id1", "result1"),
            crate::provider::Message::text("assistant", "response"),
            crate::provider::Message::tool_result("id2", "result2"),
        ];
        let candidates = find_trim_candidates(&messages);
        assert_eq!(candidates, vec![1, 3]);
    }
}
