// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! SSE stream parser — handles OpenAI-compatible SSE streams with
//! Gemma 4 thinking block extraction and tool call accumulation.

use crate::provider::StreamEvent;
use anyhow::Result;
use futures_util::StreamExt;
use reqwest::Response;
use serde::Deserialize;
use tokio::sync::mpsc;

/// Accumulated tool call from streaming deltas.
#[derive(Debug, Default, Clone)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
}

/// OpenAI SSE delta chunk structure.
#[derive(Debug, Deserialize)]
struct SseChunk {
    choices: Option<Vec<SseChoice>>,
}

#[derive(Debug, Deserialize)]
struct SseChoice {
    delta: Option<SseDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SseDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
    tool_calls: Option<Vec<SseToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct SseToolCallDelta {
    index: Option<usize>,
    id: Option<String>,
    function: Option<SseFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct SseFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

/// State machine for extracting thinking blocks from content stream.
#[derive(Debug, PartialEq)]
enum ThinkingState {
    /// Normal text output
    Normal,
    /// Inside a `<|channel>thought` ... `<channel|>` block (Gemma 4)
    InsideGemma4,
    /// Inside a `<think>` ... `</think>` block (legacy)
    InsideLegacy,
}

/// Parse an SSE stream from an OpenAI-compatible endpoint.
///
/// Handles:
/// - Text content deltas
/// - Gemma 4 thinking: `<|channel>thought` ... `<channel|>` extraction
/// - Legacy thinking: `<think>` ... `</think>` extraction
/// - `reasoning_content` field (direct thinking support)
/// - Tool call delta accumulation by index
/// - Stream completion
pub async fn parse_sse_stream(
    response: Response,
    tx: mpsc::Sender<StreamEvent>,
) -> Result<()> {
    let mut tool_calls: Vec<ToolCallAccumulator> = Vec::new();
    let mut thinking_state = ThinkingState::Normal;
    let mut content_buffer = String::new();

    let mut stream = response.bytes_stream();
    let mut chunk_count: u64 = 0;
    let start = std::time::Instant::now();

    while let Some(chunk_result) = stream.next().await {
        let chunk = match chunk_result {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(error = %e, chunks = chunk_count, "SSE stream: chunk read error");
                let _ = tx.send(StreamEvent::Error(e.to_string())).await;
                break;
            }
        };

        chunk_count += 1;
        let text = String::from_utf8_lossy(&chunk);

        for line in text.lines() {
            let line = line.trim();

            if line.is_empty() || line.starts_with(':') {
                continue;
            }

            if line == "data: [DONE]" {
                // CRITICAL: flush remaining content buffer before Done
                if !content_buffer.is_empty() {
                    let _ = tx.send(StreamEvent::TextDelta(content_buffer.clone())).await;
                    content_buffer.clear();
                }
                emit_accumulated_tools(&tool_calls, &tx).await;
                let _ = tx.send(StreamEvent::Done).await;
                tracing::debug!(
                    chunks = chunk_count,
                    tool_calls = tool_calls.len(),
                    elapsed_ms = start.elapsed().as_millis() as u64,
                    "SSE stream: [DONE] received"
                );
                return Ok(());
            }

            if let Some(json_str) = line.strip_prefix("data: ") {
                if let Ok(chunk) = serde_json::from_str::<SseChunk>(json_str) {
                    process_chunk(
                        &chunk,
                        &mut tool_calls,
                        &mut thinking_state,
                        &mut content_buffer,
                        &tx,
                    )
                    .await;
                }
            }
        }
    }

    // Stream ended without [DONE] — flush remaining content buffer
    tracing::warn!(
        chunks = chunk_count,
        content_len = content_buffer.len(),
        tool_calls = tool_calls.len(),
        elapsed_ms = start.elapsed().as_millis() as u64,
        "SSE stream: ended WITHOUT [DONE]"
    );
    if !content_buffer.is_empty() {
        let _ = tx.send(StreamEvent::TextDelta(content_buffer)).await;
    }
    emit_accumulated_tools(&tool_calls, &tx).await;
    let _ = tx.send(StreamEvent::Done).await;
    Ok(())
}

/// Process a single SSE chunk.
async fn process_chunk(
    chunk: &SseChunk,
    tool_calls: &mut Vec<ToolCallAccumulator>,
    thinking_state: &mut ThinkingState,
    content_buffer: &mut String,
    tx: &mpsc::Sender<StreamEvent>,
) {
    let choices = match &chunk.choices {
        Some(c) => c,
        None => return,
    };

    for choice in choices {
        let delta = match &choice.delta {
            Some(d) => d,
            None => {
                // No delta — check if this is a finish-only chunk
                if let Some(reason) = &choice.finish_reason {
                    if reason == "length" {
                        tracing::warn!("Response truncated — model hit max_tokens");
                    }
                }
                continue;
            }
        };

        // Direct reasoning_content field (some providers)
        if let Some(reasoning) = &delta.reasoning_content {
            if !reasoning.is_empty() {
                let _ = tx.send(StreamEvent::ThinkingDelta(reasoning.clone())).await;
            }
        }

        // Content with thinking block extraction
        if let Some(content) = &delta.content {
            process_content_delta(
                content,
                thinking_state,
                content_buffer,
                tx,
            )
            .await;
        }

        // Tool call delta accumulation
        if let Some(tc_deltas) = &delta.tool_calls {
            for tc_delta in tc_deltas {
                accumulate_tool_call(tool_calls, tc_delta);
            }
        }

        // Check finish_reason for truncation detection
        if let Some(reason) = &choice.finish_reason {
            if reason == "length" {
                tracing::warn!("Response truncated — model hit max_tokens (finish_reason=length)");
            }
        }
    }
}

/// Process content delta with thinking block extraction.
///
/// Handles both Gemma 4 (`<|channel>thought` ... `<channel|>`) and
/// legacy (`<think>` ... `</think>`) thinking formats.
async fn process_content_delta(
    content: &str,
    state: &mut ThinkingState,
    buffer: &mut String,
    tx: &mpsc::Sender<StreamEvent>,
) {
    buffer.push_str(content);

    loop {
        match state {
            ThinkingState::Normal => {
                // Check for Gemma 4 thinking start
                if let Some(pos) = buffer.find("<|channel>thought") {
                    let before = &buffer[..pos];
                    if !before.is_empty() {
                        let _ = tx.send(StreamEvent::TextDelta(before.to_string())).await;
                    }
                    *buffer = buffer[pos + "<|channel>thought".len()..].to_string();
                    *state = ThinkingState::InsideGemma4;
                    continue;
                }

                // Check for legacy thinking start
                if let Some(pos) = buffer.find("<think>") {
                    let before = &buffer[..pos];
                    if !before.is_empty() {
                        let _ = tx.send(StreamEvent::TextDelta(before.to_string())).await;
                    }
                    *buffer = buffer[pos + "<think>".len()..].to_string();
                    *state = ThinkingState::InsideLegacy;
                    continue;
                }

                // No thinking tags — check if buffer might contain partial tag
                let safe_len = safe_emit_length(buffer);
                if safe_len > 0 {
                    let emit = buffer[..safe_len].to_string();
                    let _ = tx.send(StreamEvent::TextDelta(emit)).await;
                    *buffer = buffer[safe_len..].to_string();
                }
                break;
            }

            ThinkingState::InsideGemma4 => {
                if let Some(pos) = buffer.find("<channel|>") {
                    let thinking = &buffer[..pos];
                    if !thinking.is_empty() {
                        let _ = tx.send(StreamEvent::ThinkingDelta(thinking.to_string())).await;
                    }
                    *buffer = buffer[pos + "<channel|>".len()..].to_string();
                    *state = ThinkingState::Normal;
                    continue;
                }

                // Emit partial thinking, keep last 15 chars for tag detection
                let safe = buffer.len().saturating_sub(15);
                if safe > 0 {
                    let emit = buffer[..safe].to_string();
                    let _ = tx.send(StreamEvent::ThinkingDelta(emit)).await;
                    *buffer = buffer[safe..].to_string();
                }
                break;
            }

            ThinkingState::InsideLegacy => {
                if let Some(pos) = buffer.find("</think>") {
                    let thinking = &buffer[..pos];
                    if !thinking.is_empty() {
                        let _ = tx.send(StreamEvent::ThinkingDelta(thinking.to_string())).await;
                    }
                    *buffer = buffer[pos + "</think>".len()..].to_string();
                    *state = ThinkingState::Normal;
                    continue;
                }

                let safe = buffer.len().saturating_sub(10);
                if safe > 0 {
                    let emit = buffer[..safe].to_string();
                    let _ = tx.send(StreamEvent::ThinkingDelta(emit)).await;
                    *buffer = buffer[safe..].to_string();
                }
                break;
            }
        }
    }
}

/// Calculate how many bytes can be safely emitted without splitting a potential tag.
fn safe_emit_length(buffer: &str) -> usize {
    // Keep enough chars to detect partial `<|channel>thought` (17 chars) or `<think>` (7 chars)
    let reserve = 20;
    let raw = buffer.len().saturating_sub(reserve);
    // Snap to a valid char boundary — never slice mid-codepoint.
    let mut pos = raw;
    while pos > 0 && !buffer.is_char_boundary(pos) {
        pos -= 1;
    }
    pos
}

/// Accumulate tool call deltas by index.
fn accumulate_tool_call(
    tool_calls: &mut Vec<ToolCallAccumulator>,
    delta: &SseToolCallDelta,
) {
    let idx = delta.index.unwrap_or(0);

    // Extend the vector if needed
    while tool_calls.len() <= idx {
        tool_calls.push(ToolCallAccumulator::default());
    }

    let tc = &mut tool_calls[idx];

    if let Some(id) = &delta.id {
        tc.id.clone_from(id);
    }
    if let Some(func) = &delta.function {
        if let Some(name) = &func.name {
            tc.name.clone_from(name);
        }
        if let Some(args) = &func.arguments {
            tc.arguments.push_str(args);
        }
    }
}

/// Emit all accumulated tool calls as StreamEvents.
async fn emit_accumulated_tools(
    tool_calls: &[ToolCallAccumulator],
    tx: &mpsc::Sender<StreamEvent>,
) {
    for tc in tool_calls {
        if !tc.name.is_empty() {
            let _ = tx
                .send(StreamEvent::ToolCall {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
                    arguments: tc.arguments.clone(),
                })
                .await;
        }
    }
}

/// Detect thought spirals — repetitive thinking patterns.
/// Returns true if the thinking content shows repetition.
pub fn detect_thought_spiral(thinking: &str) -> bool {
    if thinking.len() < 200 {
        return false;
    }

    let lines: Vec<&str> = thinking.lines().collect();
    if lines.len() < 6 {
        return false;
    }

    // Check for duplicate consecutive lines
    let mut consecutive_dupes = 0;
    for window in lines.windows(2) {
        if window[0].trim() == window[1].trim() && !window[0].trim().is_empty() {
            consecutive_dupes += 1;
        }
    }

    // 3+ consecutive duplicates = spiral
    consecutive_dupes >= 3
}

include!("stream_parser_tests.rs");
