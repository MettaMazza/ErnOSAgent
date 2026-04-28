//! Platform execution engines — L1 tool chain and L2 ReAct loop for platforms.
//!
//! These run the full tool execution pipeline for platform-ingested messages,
//! collecting ToolEvent metadata for thinking thread observability.

use crate::web::state::AppState;
use super::platform_ingest::{ToolEvent, AuditSummary, audit_and_capture};

/// Execute a tool and capture a ToolEvent with timing.
pub async fn execute_and_capture(
    state: &AppState,
    tc: &crate::tools::schema::ToolCall,
) -> (crate::tools::schema::ToolResult, ToolEvent) {
    let start = std::time::Instant::now();
    let result = crate::web::tool_dispatch::execute_tool_with_state(state, tc).await;
    let elapsed = start.elapsed().as_millis() as u64;

    let preview = if result.output.len() > 300 {
        // Find the char boundary at or before byte 300 to avoid slicing mid-character
        let boundary = result.output.char_indices()
            .take_while(|(i, _)| *i <= 300)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        format!("{}…", &result.output[..boundary])
    } else {
        result.output.clone()
    };

    let event = ToolEvent {
        name: tc.name.clone(),
        success: result.success,
        elapsed_ms: elapsed,
        output_preview: preview,
    };

    tracing::info!(
        tool = %tc.name,
        success = result.success,
        elapsed_ms = elapsed,
        "Platform tool executed"
    );

    (result, event)
}

/// Run L1 tool chain. Returns (reply, tool_events, audit_summary).
/// If `sse_tx` is provided, tool events are emitted live to keep the SSE stream alive.
pub async fn run_platform_tool_chain(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    messages: &mut Vec<crate::provider::Message>,
    tools: &serde_json::Value,
    user_query: &str,
    session_id: &str,
    first_tc: crate::tools::schema::ToolCall,
    sse_tx: Option<&tokio::sync::mpsc::Sender<Result<axum::response::sse::Event, std::convert::Infallible>>>,
) -> (String, Vec<ToolEvent>, Option<AuditSummary>) {
    let max_iterations = 50;
    let mut current_tc = first_tc;
    let mut tool_events: Vec<ToolEvent> = Vec::new();

    for iteration in 0..max_iterations {
        tracing::info!(tool = %current_tc.name, iteration, "Platform L1 tool execution");

        // Emit tool_start live
        if let Some(tx) = sse_tx {
            let _ = tx.send(Ok(axum::response::sse::Event::default()
                .event("tool_start")
                .data(serde_json::json!({"name": &current_tc.name}).to_string())
            )).await;
        }

        let (result, event) = execute_and_capture(state, &current_tc).await;

        // Emit tool_result live
        if let Some(tx) = sse_tx {
            let _ = tx.send(Ok(axum::response::sse::Event::default()
                .event("tool_result")
                .data(serde_json::json!({
                    "name": event.name, "success": event.success,
                    "elapsed_ms": event.elapsed_ms, "preview": event.output_preview,
                }).to_string())
            )).await;
        }

        tool_events.push(event);
        append_tool_messages(messages, &current_tc, &result);

        let total_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
        let tool_msg_count = messages.iter().filter(|m| m.role == "tool").count();
        tracing::info!(
            iteration,
            tool = %current_tc.name,
            msg_count = messages.len(),
            tool_msgs = tool_msg_count,
            total_chars,
            estimated_tokens = total_chars / 4,
            context_length = state.model_spec.context_length,
            "Tool chain: context state before budget enforcement"
        );
        enforce_context_budget(messages, state.model_spec.context_length);

        // Pre-infer budget check: if context is still dangerously close to the limit
        // after trimming, fail fast with a useful message instead of sending a
        // doomed inference request that will process for a very long time.
        let post_trim_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
        let post_trim_tokens = post_trim_chars / 4 + 2000;
        let budget_ratio = post_trim_tokens as f64 / state.model_spec.context_length as f64;
        if budget_ratio > 0.90 {
            tracing::warn!(
                post_trim_tokens,
                context_length = state.model_spec.context_length,
                budget_ratio = format!("{:.1}%", budget_ratio * 100.0),
                "Context at {:.0}% of window after trimming — stopping tool chain to avoid excessive inference time",
                budget_ratio * 100.0,
            );
            let reply = format!(
                "I've read through the file but the accumulated context is at {:.0}% of the model's window. \
                 To continue reading, please ask me to read the next section in a new message — \
                 I'll use the bookmarks to pick up where I left off.",
                budget_ratio * 100.0,
            );
            return (reply, tool_events, None);
        }

        match reinfer_and_dispatch(state, provider, messages, tools, user_query, session_id).await {
            LoopAction::Reply(text, audit) => return (text, tool_events, Some(audit)),
            LoopAction::NextTool(tc) => current_tc = tc,
            LoopAction::MultiTool(tcs) => {
                // Execute all but the last tool inline, then set current_tc to the last
                let mut remaining = tcs;
                let last_tc = remaining.pop();
                for tc in remaining {
                    tracing::info!(tool = %tc.name, iteration, "Platform L1 parallel tool execution");
                    if let Some(tx) = sse_tx {
                        let _ = tx.send(Ok(axum::response::sse::Event::default()
                            .event("tool_start")
                            .data(serde_json::json!({"name": &tc.name}).to_string())
                        )).await;
                    }
                    let (result, event) = execute_and_capture(state, &tc).await;
                    if let Some(tx) = sse_tx {
                        let _ = tx.send(Ok(axum::response::sse::Event::default()
                            .event("tool_result")
                            .data(serde_json::json!({
                                "name": event.name, "success": event.success,
                                "elapsed_ms": event.elapsed_ms, "preview": event.output_preview,
                            }).to_string())
                        )).await;
                    }
                    tool_events.push(event);
                    append_tool_messages(messages, &tc, &result);
                }
                match last_tc {
                    Some(tc) => current_tc = tc,
                    None => return ("No tool calls to execute.".to_string(), tool_events, None),
                }
            }
            LoopAction::Escalate(reply, events, audit) => {
                tool_events.extend(events);
                return (reply, tool_events, audit);
            }
            LoopAction::Error(msg) => return (msg, tool_events, None),
        }
    }

    ("Tool chain reached maximum iterations.".to_string(), tool_events, None)
}

/// Run L2 ReAct loop. Returns (reply, tool_events, audit_summary).
/// If `sse_tx` is provided, tool events are emitted live to the SSE stream
/// so platform clients (Discord thinking thread) can see progress.
pub async fn run_platform_react(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    mut messages: Vec<crate::provider::Message>,
    objective: &str,
    plan: Option<&str>,
    user_query: &str,
    session_id: &str,
    sse_tx: Option<&tokio::sync::mpsc::Sender<Result<axum::response::sse::Event, std::convert::Infallible>>>,
) -> (String, Vec<ToolEvent>, Option<AuditSummary>) {
    let tools = crate::tools::schema::layer2_tools();
    let mut tool_events: Vec<ToolEvent> = Vec::new();
    inject_react_instruction(&mut messages, objective, plan);

    for turn in 0..50 {
        let rx = match provider.chat(&messages, Some(&tools), true).await {
            Ok(rx) => rx,
            Err(e) => return (format!("ReAct error: {}", e), tool_events, None),
        };
        use crate::inference::stream_consumer::{self, ConsumeResult, NullSink};
        let mut sink = NullSink;
        let result = stream_consumer::consume_stream(rx, &mut sink).await;
        match result {
            ConsumeResult::Spiral { .. } => {
                let recovered = stream_consumer::reprompt_after_spiral(
                    provider, &mut messages, Some(&tools), &mut sink,
                ).await;
                if let ConsumeResult::Reply { text, .. } = recovered {
                    let (audited, audit) = audit_and_capture(
                        state, provider, &mut messages, &tools, user_query, &text, session_id,
                    ).await;
                    return (audited, tool_events, Some(audit));
                }
            }
            ConsumeResult::Reply { text, .. } => {
                let (audited, audit) = audit_and_capture(
                    state, provider, &mut messages, &tools, user_query, &text, session_id,
                ).await;
                return (audited, tool_events, Some(audit));
            }
            ConsumeResult::ToolCall { id, name, arguments } => {
                let tc = crate::tools::schema::ToolCall { id, name, arguments };
                if let Some(result) = handle_react_tool(
                    state, &mut messages, &mut tool_events, &tc, provider, &tools, user_query, session_id, turn, sse_tx,
                ).await {
                    return result;
                }
            }
            ConsumeResult::Escalate { objective: nested, .. } => {
                tracing::info!(nested = %nested, turn, "Platform ReAct nested escalation");
                messages.push(crate::provider::Message::text("system", &format!("Additional objective: {}", nested)));
            }
            ConsumeResult::ToolCalls(calls) => {
                for (id, name, arguments) in calls {
                    let tc = crate::tools::schema::ToolCall { id, name, arguments };
                    if let Some(result) = handle_react_tool(
                        state, &mut messages, &mut tool_events, &tc, provider, &tools, user_query, session_id, turn, sse_tx,
                    ).await {
                        return result;
                    }
                }
            }
            ConsumeResult::Error(e) => return (format!("ReAct error: {}", e), tool_events, None),
            _ => {}
        }
    }
    ("ReAct loop reached maximum turns.".to_string(), tool_events, None)
}

/// Handle a tool call within the ReAct loop.
/// Returns Some if the loop should exit, None to continue.
async fn handle_react_tool(
    state: &AppState,
    messages: &mut Vec<crate::provider::Message>,
    tool_events: &mut Vec<ToolEvent>,
    tc: &crate::tools::schema::ToolCall,
    provider: &dyn crate::provider::Provider,
    tools: &serde_json::Value,
    user_query: &str,
    session_id: &str,
    turn: usize,
    sse_tx: Option<&tokio::sync::mpsc::Sender<Result<axum::response::sse::Event, std::convert::Infallible>>>,
) -> Option<(String, Vec<ToolEvent>, Option<AuditSummary>)> {
    // Check for reply_request (loop terminator)
    if let Some(reply_text) = crate::tools::schema::extract_reply_text(tc) {
        let (audited, audit) = audit_and_capture(
            state, provider, messages, tools, user_query, &reply_text, session_id,
        ).await;
        crate::web::ws_learning::ingest_assistant_turn(state, &audited, session_id).await;
        return Some((audited, tool_events.clone(), Some(audit)));
    }

    // Emit tool_start live to SSE if available
    if let Some(tx) = sse_tx {
        let _ = tx.send(Ok(axum::response::sse::Event::default()
            .event("tool_start")
            .data(serde_json::json!({"name": &tc.name}).to_string())
        )).await;
    }

    tracing::info!(tool = %tc.name, turn, "Platform ReAct tool execution");
    let (result, event) = execute_and_capture(state, tc).await;

    // Emit tool_result live to SSE if available
    if let Some(tx) = sse_tx {
        let _ = tx.send(Ok(axum::response::sse::Event::default()
            .event("tool_result")
            .data(serde_json::json!({
                "name": event.name, "success": event.success,
                "elapsed_ms": event.elapsed_ms, "preview": event.output_preview,
            }).to_string())
        )).await;
    }

    tool_events.push(event);
    append_tool_messages(messages, tc, &result);
    enforce_context_budget(messages, state.model_spec.context_length);
    None
}

/// Result of re-inference after tool execution.
enum LoopAction {
    Reply(String, AuditSummary),
    NextTool(crate::tools::schema::ToolCall),
    MultiTool(Vec<crate::tools::schema::ToolCall>),
    Escalate(String, Vec<ToolEvent>, Option<AuditSummary>),
    Error(String),
}

/// Re-infer after tool execution and classify the result.
async fn reinfer_and_dispatch(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    messages: &mut Vec<crate::provider::Message>,
    tools: &serde_json::Value,
    user_query: &str,
    session_id: &str,
) -> LoopAction {
    let rx = match provider.chat(messages, Some(tools), true).await {
        Ok(rx) => rx,
        Err(e) => {
            tracing::error!(error = %e, "Platform L1 re-inference failed");
            return LoopAction::Error(format!("Tool execution error: {}", e));
        }
    };

    use crate::inference::stream_consumer::{self, ConsumeResult, NullSink};
    let mut sink = NullSink;
    let result = stream_consumer::consume_stream(rx, &mut sink).await;
    match result {
        ConsumeResult::Spiral { .. } => {
            let recovered = stream_consumer::reprompt_after_spiral(
                provider, messages, Some(tools), &mut sink,
            ).await;
            match recovered {
                ConsumeResult::Reply { text, .. } => {
                    let (audited, audit) = audit_and_capture(
                        state, provider, messages, tools, user_query, &text, session_id,
                    ).await;
                    LoopAction::Reply(audited, audit)
                }
                ConsumeResult::ToolCall { id, name, arguments } => {
                    LoopAction::NextTool(crate::tools::schema::ToolCall { id, name, arguments })
                }
                ConsumeResult::Error(e) => LoopAction::Error(format!("Spiral recovery: {}", e)),
                _ => LoopAction::Error("Spiral recovery: unexpected result".to_string()),
            }
        }
        ConsumeResult::Reply { text, .. } if text.trim().is_empty() => {
            // The model returned finish_reason=stop with ZERO tokens generated.
            // This is NOT a model decision — it's context overflow. The prompt
            // filled the entire context window, leaving no room for decode.
            // Trim the heaviest tool results and retry once.
            let total_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
            tracing::error!(
                total_chars,
                msg_count = messages.len(),
                estimated_tokens = total_chars / 4 + 2000,
                context_length = state.model_spec.context_length,
                "Model returned empty response — context overflow detected, trimming and retrying"
            );
            enforce_context_budget(messages, state.model_spec.context_length / 2);

            // Retry once with trimmed context
            let retry_rx = match provider.chat(messages, Some(tools), false).await {
                Ok(rx) => rx,
                Err(e) => return LoopAction::Error(format!("Retry after empty response failed: {}", e)),
            };
            let mut retry_sink = NullSink;
            match stream_consumer::consume_stream(retry_rx, &mut retry_sink).await {
                ConsumeResult::Reply { text, .. } if !text.trim().is_empty() => {
                    let (audited, audit) = audit_and_capture(
                        state, provider, messages, tools, user_query, &text, session_id,
                    ).await;
                    LoopAction::Reply(audited, audit)
                }
                ConsumeResult::ToolCall { id, name, arguments } => {
                    LoopAction::NextTool(crate::tools::schema::ToolCall { id, name, arguments })
                }
                _ => {
                    // Retry also empty — return a diagnostic message instead of silence
                    let reply = "The tool results were too large for the context window. \
                                 Please ask me to continue in a new message — I'll use \
                                 the tool results I've already collected.".to_string();
                    LoopAction::Error(reply)
                }
            }
        }
        ConsumeResult::Reply { text, .. } => {
            let (audited, audit) = audit_and_capture(
                state, provider, messages, tools, user_query, &text, session_id,
            ).await;
            LoopAction::Reply(audited, audit)
        }
        ConsumeResult::ToolCall { id, name, arguments } => {
            LoopAction::NextTool(crate::tools::schema::ToolCall { id, name, arguments })
        }
        ConsumeResult::ToolCalls(calls) => {
            let tcs: Vec<crate::tools::schema::ToolCall> = calls.into_iter()
                .map(|(id, name, arguments)| crate::tools::schema::ToolCall { id, name, arguments })
                .collect();
            LoopAction::MultiTool(tcs)
        }
        ConsumeResult::Escalate { objective, plan, .. } => {
            let (reply, events, audit) = run_platform_react(
                state, provider, messages.clone(), &objective, plan.as_deref(),
                user_query, session_id, None,
            ).await;
            LoopAction::Escalate(reply, events, audit)
        }
        ConsumeResult::Error(e) => {
            tracing::error!(error = %e, "Platform L1 stream error");
            LoopAction::Error(format!("Error: {}", e))
        }
        _ => LoopAction::Error("Unexpected stream result".to_string()),
    }
}

/// Append tool call and result messages to the conversation.
pub(crate) fn append_tool_messages(
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
pub(crate) fn enforce_context_budget(
    messages: &mut Vec<crate::provider::Message>,
    context_length: usize,
) {
    // Estimate total tokens: ~4 chars per token, plus overhead for tool schemas (~2000 tokens)
    let total_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
    let estimated_tokens = total_chars / 4 + 2000;

    if estimated_tokens <= context_length {
        return;
    }

    tracing::warn!(
        estimated_tokens,
        context_length,
        overshoot = estimated_tokens - context_length,
        "Context budget exceeded — trimming tool results"
    );

    // Collect indices of tool messages, oldest first
    let tool_indices: Vec<usize> = messages.iter().enumerate()
        .filter(|(_, m)| m.role == "tool")
        .map(|(i, _)| i)
        .collect();

    // Trim from oldest to newest, keeping the most recent tool result as large as possible
    let mut trimmed_total = 0usize;
    for &idx in &tool_indices {
        let total_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
        let estimated = total_chars / 4 + 2000;
        if estimated <= context_length {
            break;
        }

        let content = messages[idx].text_content();
        let content_len = content.len();

        // For older tool results (not the last one), compress to preserve key content
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

    let final_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
    tracing::warn!(
        trimmed_total,
        final_estimated_tokens = final_chars / 4 + 2000,
        context_length,
        "Context budget enforcement complete"
    );
}

/// Compress a tool result to preserve key content while reducing size.
/// Keeps pagination markers (file_read headers/bookmarks), the first ~2000 chars,
/// last ~2000 chars, all section headings from the middle, and structural metadata.
fn compress_tool_result(content: &str) -> String {
    let total_lines = content.lines().count();
    let total_chars = content.len();

    // If already small, don't compress
    if total_chars <= 8000 {
        return content.to_string();
    }

    // Extract file_read pagination markers BEFORE compressing.
    // These are critical — without them the model loses its reading position
    // and re-reads the same pages in a loop.
    let mut pagination_header = String::new();
    let mut bookmark = String::new();

    let first_line = content.lines().next().unwrap_or("");
    if first_line.starts_with("[Lines ") || first_line.starts_with("[FILE SAVED") {
        pagination_header = first_line.to_string();
    }

    let last_line = content.lines().last().unwrap_or("");
    if last_line.contains("[BOOKMARK:") || last_line.contains("END OF FILE") {
        bookmark = last_line.to_string();
    }

    // Strip the pagination markers from content before extracting head/tail
    let inner = if !pagination_header.is_empty() {
        let skip = pagination_header.len() + 1; // +1 for newline
        &content[skip.min(content.len())..]
    } else {
        content
    };

    // Extract first ~2000 chars (preserve beginning context)
    let head_end = inner.char_indices()
        .take_while(|(i, _)| *i < 2000)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(inner.len().min(2000));
    let head = match inner[..head_end].rfind('\n') {
        Some(pos) => &inner[..pos],
        None => &inner[..head_end],
    };

    // Extract last ~2000 chars (preserve ending context, excluding bookmark)
    let inner_for_tail = if !bookmark.is_empty() {
        let end = inner.len().saturating_sub(bookmark.len() + 1);
        &inner[..end]
    } else {
        inner
    };
    let tail_start = inner_for_tail.len().saturating_sub(2000);
    let tail = match inner_for_tail[tail_start..].find('\n') {
        Some(pos) => &inner_for_tail[tail_start + pos + 1..],
        None => &inner_for_tail[tail_start..],
    };

    // Extract section headings from the middle
    let head_lines = head.lines().count();
    let inner_total_lines = inner.lines().count();
    let tail_line_start = inner_total_lines.saturating_sub(tail.lines().count());
    let middle_headings: Vec<&str> = inner.lines()
        .enumerate()
        .filter(|(i, _)| *i >= head_lines && *i < tail_line_start)
        .filter(|(_, line)| {
            let trimmed = line.trim();
            trimmed.starts_with('#') || trimmed.starts_with("---") || trimmed.starts_with("***")
        })
        .map(|(_, line)| line)
        .collect();

    let headings_section = if middle_headings.is_empty() {
        String::new()
    } else {
        format!("\n\n[Section headings from compressed region:]\n{}", middle_headings.join("\n"))
    };

    // Build compressed output — pagination markers always preserved
    let mut output = String::new();

    if !pagination_header.is_empty() {
        output.push_str(&pagination_header);
        output.push('\n');
    }

    output.push_str(&format!(
        "[COMPRESSED — {} total chars, {} lines — reading position preserved]\n\
         \n--- BEGIN (lines 1-{}) ---\n{}\n--- END OF HEAD ---\n\
         \n[... {} chars / {} lines compressed ...]{}\n\
         \n--- TAIL (lines {}-{}) ---\n{}\n--- END ---",
        total_chars, total_lines,
        head.lines().count(), head,
        total_chars - head.len() - tail.len(),
        tail_line_start - head_lines,
        headings_section,
        tail_line_start, inner_total_lines, tail,
    ));

    if !bookmark.is_empty() {
        output.push('\n');
        output.push_str(&bookmark);
    }

    output
}

/// Inject ReAct system instruction into messages.
fn inject_react_instruction(
    messages: &mut Vec<crate::provider::Message>,
    objective: &str,
    plan: Option<&str>,
) {
    let instruction = format!(
        "You are now in ReAct (Reason-Act-Observe) mode.\n\
         Objective: {}\n{}\n\
         Use tools to accomplish the objective. When done, call `reply_request` with your final response.",
        objective,
        plan.map(|p| format!("Plan:\n{}", p)).unwrap_or_default(),
    );
    messages.push(crate::provider::Message::text("system", &instruction));
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
    fn test_inject_react_instruction() {
        let mut messages = Vec::new();
        inject_react_instruction(&mut messages, "Test objective", Some("Step 1"));
        assert_eq!(messages.len(), 1);
        let content = messages[0].text_content();
        assert!(content.contains("Test objective"));
        assert!(content.contains("Step 1"));
    }

    #[test]
    fn test_inject_react_no_plan() {
        let mut messages = Vec::new();
        inject_react_instruction(&mut messages, "Simple task", None);
        assert_eq!(messages.len(), 1);
        assert!(messages[0].text_content().contains("Simple task"));
    }
}
