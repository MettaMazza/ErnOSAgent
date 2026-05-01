//! Ern-OS — Unified Stream Consumer
//!
//! ONE consumer for ALL stream consumption paths. Side effects (WebSocket
//! forwarding, SSE emission, cancel checks) are pluggable via `StreamSink`.
//!
//! Spiral detection, cancel support, and classification live here — once.
//! No more duplicated consumers, no more missed code paths.

use crate::provider::stream_parser::detect_thought_spiral;
use crate::provider::StreamEvent;
use tokio::sync::mpsc;

// ── Result type ──────────────────────────────────────────────────────

/// Unified result from stream consumption. All paths return this.
pub enum ConsumeResult {
    /// Direct text response.
    Reply { text: String, thinking: Option<String> },
    /// Single tool call.
    ToolCall { id: String, name: String, arguments: String },
    /// Multiple parallel tool calls.
    ToolCalls(Vec<(String, String, String)>),
    /// Escalation to ReAct loop.
    Escalate { objective: String, plan: Option<String>, planned_turns: usize },
    /// Plan proposal awaiting user approval.
    PlanProposal { title: String, plan_markdown: String, estimated_turns: usize },
    /// Thinking spiral detected — caller should re-prompt.
    Spiral { text: String, thinking: String },
    /// User cancelled mid-stream.
    Cancelled { text: String, thinking: String },
    /// Stream error.
    Error(String),
}

// ── StreamSink trait ─────────────────────────────────────────────────

/// Side-effect handler for stream consumption.
/// Implementations control where/how events are forwarded.
///
/// Default implementations are no-ops so sinks only override what they need.
#[allow(unused_variables)]
pub trait StreamSink: Send {
    /// Called on each text content delta.
    fn on_text(&mut self, delta: &str) -> impl std::future::Future<Output = ()> + Send { async {} }

    /// Called on each thinking/reasoning delta.
    fn on_thinking(&mut self, delta: &str) -> impl std::future::Future<Output = ()> + Send { async {} }

    /// Called when a tool call is received.
    fn on_tool_call(&mut self, id: &str, name: &str, args: &str) -> impl std::future::Future<Output = ()> + Send { async {} }

    /// Called on stream completion.
    fn on_done(&mut self) -> impl std::future::Future<Output = ()> + Send { async {} }

    /// Called on stream error.
    fn on_error(&mut self, error: &str) -> impl std::future::Future<Output = ()> + Send { async {} }

    /// Called when a thinking spiral is detected (before stream is killed).
    fn on_spiral_detected(&mut self, thinking_len: usize) -> impl std::future::Future<Output = ()> + Send { async {} }

    /// Called when cancellation triggers.
    fn on_cancelled(&mut self) -> impl std::future::Future<Output = ()> + Send { async {} }

    /// Check if the consumer should cancel. Return true to stop.
    fn should_cancel(&self) -> bool { false }
}

// ── Sink implementations ─────────────────────────────────────────────

/// Silent sink — no side effects. For platform_exec tool chains and
/// any path that doesn't need live event forwarding.
pub struct NullSink;
impl StreamSink for NullSink {}

/// Cancellable null sink — checks cancel flag but no forwarding.
pub struct CancellableNullSink {
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl StreamSink for CancellableNullSink {
    fn should_cancel(&self) -> bool {
        self.cancel.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// WebSocket sink — forwards thinking deltas to WebUI, checks cancel flag.
pub struct WebSocketSink<'a> {
    pub sender: &'a mut futures_util::stream::SplitSink<
        axum::extract::ws::WebSocket,
        axum::extract::ws::Message,
    >,
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl<'a> StreamSink for WebSocketSink<'a> {
    async fn on_text(&mut self, delta: &str) {
        let msg = serde_json::json!({"type": "text_delta", "content": delta});
        let _ = futures_util::SinkExt::send(
            self.sender,
            axum::extract::ws::Message::Text(msg.to_string().into()),
        ).await;
    }

    async fn on_thinking(&mut self, delta: &str) {
        let msg = serde_json::json!({"type": "thinking_delta", "content": delta});
        let _ = futures_util::SinkExt::send(
            self.sender,
            axum::extract::ws::Message::Text(msg.to_string().into()),
        ).await;
    }

    async fn on_spiral_detected(&mut self, _thinking_len: usize) {
        let msg = serde_json::json!({"type": "spiral_detected"});
        let _ = futures_util::SinkExt::send(
            self.sender,
            axum::extract::ws::Message::Text(msg.to_string().into()),
        ).await;
    }

    fn should_cancel(&self) -> bool {
        self.cancel.load(std::sync::atomic::Ordering::Relaxed)
    }
}

/// SSE sink — emits Server-Sent Events for Discord thinking thread.
/// Buffers thinking tokens and only emits every ~200 chars to avoid
/// flooding the Discord API with per-token messages.
pub struct SseSink<'a> {
    pub tx: &'a tokio::sync::mpsc::Sender<
        Result<axum::response::sse::Event, std::convert::Infallible>,
    >,
    thinking_buf: String,
}

impl<'a> SseSink<'a> {
    pub fn new(tx: &'a tokio::sync::mpsc::Sender<
        Result<axum::response::sse::Event, std::convert::Infallible>,
    >) -> Self {
        Self { tx, thinking_buf: String::new() }
    }

    /// Flush any accumulated thinking tokens as a single SSE event.
    async fn flush_thinking(&mut self) {
        if self.thinking_buf.is_empty() {
            return;
        }
        let chunk = std::mem::take(&mut self.thinking_buf);
        let _ = self.tx.send(Ok(
            axum::response::sse::Event::default()
                .event("thinking")
                .data(serde_json::json!({"chunk": chunk}).to_string()),
        )).await;
    }
}

/// Minimum characters to accumulate before emitting a thinking SSE event.
const THINKING_BATCH_SIZE: usize = 200;

impl<'a> StreamSink for SseSink<'a> {
    async fn on_thinking(&mut self, delta: &str) {
        self.thinking_buf.push_str(delta);
        if self.thinking_buf.len() >= THINKING_BATCH_SIZE {
            self.flush_thinking().await;
        }
    }

    async fn on_text(&mut self, _delta: &str) {
        // Flush any pending thinking before text arrives
        self.flush_thinking().await;
    }

    async fn on_tool_call(&mut self, _id: &str, name: &str, _args: &str) {
        self.flush_thinking().await;
        let _ = self.tx.send(Ok(
            axum::response::sse::Event::default()
                .event("tool_call")
                .data(serde_json::json!({"name": name}).to_string()),
        )).await;
    }

    async fn on_done(&mut self) {
        self.flush_thinking().await;
    }

    async fn on_spiral_detected(&mut self, thinking_len: usize) {
        self.flush_thinking().await;
        let _ = self.tx.send(Ok(
            axum::response::sse::Event::default()
                .event("spiral_detected")
                .data(serde_json::json!({"thinking_len": thinking_len}).to_string()),
        )).await;
    }

    async fn on_error(&mut self, error: &str) {
        self.flush_thinking().await;
        let _ = self.tx.send(Ok(
            axum::response::sse::Event::default()
                .event("error")
                .data(serde_json::json!({"error": error}).to_string()),
        )).await;
    }
}

// ── The ONE consumer ─────────────────────────────────────────────────

/// Consume a provider stream. ALL paths use this. ONE place for:
/// - Accumulation of text/thinking/tool_calls
/// - Spiral detection
/// - Cancel support
/// - Side-effect forwarding via `StreamSink`
pub async fn consume_stream<S: StreamSink>(
    mut rx: mpsc::Receiver<StreamEvent>,
    sink: &mut S,
) -> ConsumeResult {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut tool_calls: Vec<(String, String, String)> = Vec::new();

    while let Some(event) = rx.recv().await {
        // Cancel check — every event
        if sink.should_cancel() {
            tracing::info!(
                text_len = text.len(), thinking_len = thinking.len(),
                "Stream consumer: CANCELLED"
            );
            sink.on_cancelled().await;
            drop(rx);
            return ConsumeResult::Cancelled {
                text: if text.is_empty() { String::new() } else {
                    format!("{}\n\n*(Generation stopped by user)*", text.trim())
                },
                thinking: if thinking.is_empty() { String::new() } else { thinking },
            };
        }

        match event {
            StreamEvent::TextDelta(delta) => {
                text.push_str(&delta);
                sink.on_text(&delta).await;
            }
            StreamEvent::ThinkingDelta(delta) => {
                thinking.push_str(&delta);
                sink.on_thinking(&delta).await;

                // Spiral detection — ONE place, covers ALL paths
                if detect_thought_spiral(&thinking) {
                    tracing::warn!(
                        thinking_len = thinking.len(),
                        "THINKING SPIRAL DETECTED — killing stream"
                    );
                    sink.on_spiral_detected(thinking.len()).await;
                    drop(rx);
                    return ConsumeResult::Spiral { text, thinking };
                }
            }
            StreamEvent::ToolCall { id, name, arguments } => {
                sink.on_tool_call(&id, &name, &arguments).await;
                tool_calls.push((id, name, arguments));
            }
            StreamEvent::Done => {
                sink.on_done().await;
                break;
            }
            StreamEvent::Error(e) => {
                sink.on_error(&e).await;
                return ConsumeResult::Error(e);
            }
        }
    }

    classify(text, thinking, tool_calls)
}

/// Classify accumulated stream data into a ConsumeResult.
fn classify(
    text: String,
    thinking: String,
    tool_calls: Vec<(String, String, String)>,
) -> ConsumeResult {
    let thinking_opt = if thinking.is_empty() { None } else { Some(thinking) };

    // Check for propose_plan
    if let Some((_, _, args)) = tool_calls.iter().find(|(_, n, _)| n == "propose_plan") {
        let parsed: serde_json::Value = serde_json::from_str(args).unwrap_or_default();
        let turns = parsed["estimated_turns"].as_u64().unwrap_or(10) as usize;
        return ConsumeResult::PlanProposal {
            title: parsed["title"].as_str().unwrap_or("Plan").to_string(),
            plan_markdown: parsed["plan_markdown"].as_str().unwrap_or("").to_string(),
            estimated_turns: turns.max(3).min(50),
        };
    }

    // Check for start_react_system escalation
    if let Some((_, _, args)) = tool_calls.iter().find(|(_, n, _)| n == "start_react_system") {
        let parsed: serde_json::Value = serde_json::from_str(args).unwrap_or_default();
        let planned = parsed["planned_turns"].as_u64().unwrap_or(10) as usize;
        return ConsumeResult::Escalate {
            objective: parsed["objective"].as_str().unwrap_or("").to_string(),
            plan: parsed["plan"].as_str().map(|s| s.to_string()),
            planned_turns: planned.max(3).min(50),
        };
    }

    // Multiple tool calls
    if tool_calls.len() > 1 {
        return ConsumeResult::ToolCalls(tool_calls);
    }

    // Single tool call
    if let Some((id, name, arguments)) = tool_calls.into_iter().next() {
        return ConsumeResult::ToolCall { id, name, arguments };
    }

    // Text reply
    ConsumeResult::Reply { text, thinking: thinking_opt }
}

// ── Spiral recovery ──────────────────────────────────────────────────

/// Re-prompt after spiral detection. Trims poisoned context, injects
/// a completion directive, and re-infers with thinking disabled.
///
/// Called by any code that receives `ConsumeResult::Spiral`.
pub async fn reprompt_after_spiral<S: StreamSink>(
    provider: &dyn crate::provider::Provider,
    messages: &mut Vec<crate::provider::Message>,
    tools: Option<&serde_json::Value>,
    sink: &mut S,
) -> ConsumeResult {
    tracing::info!("Spiral recovery: trimming context and re-prompting");

    // Trim the largest tool result to prevent session poisoning
    trim_largest_tool_result(messages);

    messages.push(crate::provider::Message::text(
        "system",
        "STOP. Your previous thinking entered an infinite loop and was terminated. \
         Do NOT think further. Reply to the user's message NOW with a direct, \
         complete response. Keep it concise and focused.",
    ));

    // Re-infer with thinking DISABLED to prevent another spiral
    match provider.chat(messages, tools, false).await {
        Ok(rx) => consume_stream(rx, sink).await,
        Err(e) => {
            tracing::error!(error = %e, "Spiral re-prompt inference failed");
            ConsumeResult::Error(format!("Spiral recovery failed: {}", e))
        }
    }
}

/// Trim the largest tool result from message history to prevent session
/// poisoning after a spiral. The 2MB book result that killed all follow-up
/// messages is the canonical example of why this exists.
fn trim_largest_tool_result(messages: &mut Vec<crate::provider::Message>) {
    if let Some((idx, _)) = messages.iter().enumerate()
        .filter(|(_, m)| m.role == "tool")
        .max_by_key(|(_, m)| m.text_content().len())
    {
        let len = messages[idx].text_content().len();
        if len > 10_000 {
            tracing::warn!(idx, len, "Trimming large tool result to recover from spiral");
            messages[idx].content = serde_json::Value::String(
                format!("[Tool result trimmed — {} chars removed to recover from spiral]", len),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_consume_text_stream() {
        let (tx, rx) = mpsc::channel(32);
        tx.send(StreamEvent::TextDelta("Hello ".to_string())).await.unwrap();
        tx.send(StreamEvent::TextDelta("world".to_string())).await.unwrap();
        tx.send(StreamEvent::Done).await.unwrap();
        drop(tx);

        let result = consume_stream(rx, &mut NullSink).await;
        assert!(matches!(result, ConsumeResult::Reply { text, .. } if text == "Hello world"));
    }

    #[tokio::test]
    async fn test_consume_tool_call() {
        let (tx, rx) = mpsc::channel(32);
        tx.send(StreamEvent::ToolCall {
            id: "c1".into(), name: "shell".into(), arguments: "{}".into(),
        }).await.unwrap();
        tx.send(StreamEvent::Done).await.unwrap();
        drop(tx);

        let result = consume_stream(rx, &mut NullSink).await;
        assert!(matches!(result, ConsumeResult::ToolCall { name, .. } if name == "shell"));
    }

    #[tokio::test]
    async fn test_consume_multiple_tool_calls() {
        let (tx, rx) = mpsc::channel(32);
        tx.send(StreamEvent::ToolCall {
            id: "c1".into(), name: "shell".into(), arguments: "{}".into(),
        }).await.unwrap();
        tx.send(StreamEvent::ToolCall {
            id: "c2".into(), name: "file_read".into(), arguments: "{}".into(),
        }).await.unwrap();
        tx.send(StreamEvent::Done).await.unwrap();
        drop(tx);

        let result = consume_stream(rx, &mut NullSink).await;
        assert!(matches!(result, ConsumeResult::ToolCalls(calls) if calls.len() == 2));
    }

    #[tokio::test]
    async fn test_consume_escalation() {
        let (tx, rx) = mpsc::channel(32);
        tx.send(StreamEvent::ToolCall {
            id: "c1".into(),
            name: "start_react_system".into(),
            arguments: r#"{"objective":"deploy app","planned_turns":10}"#.into(),
        }).await.unwrap();
        tx.send(StreamEvent::Done).await.unwrap();
        drop(tx);

        let result = consume_stream(rx, &mut NullSink).await;
        assert!(matches!(result, ConsumeResult::Escalate { objective, .. } if objective == "deploy app"));
    }

    #[tokio::test]
    async fn test_consume_plan_proposal() {
        let (tx, rx) = mpsc::channel(32);
        tx.send(StreamEvent::ToolCall {
            id: "c1".into(),
            name: "propose_plan".into(),
            arguments: r#"{"title":"My Plan","plan_markdown":"step 1","estimated_turns":5}"#.into(),
        }).await.unwrap();
        tx.send(StreamEvent::Done).await.unwrap();
        drop(tx);

        let result = consume_stream(rx, &mut NullSink).await;
        assert!(matches!(result, ConsumeResult::PlanProposal { title, .. } if title == "My Plan"));
    }

    #[tokio::test]
    async fn test_consume_error() {
        let (tx, rx) = mpsc::channel(32);
        tx.send(StreamEvent::Error("test error".to_string())).await.unwrap();
        drop(tx);

        let result = consume_stream(rx, &mut NullSink).await;
        assert!(matches!(result, ConsumeResult::Error(e) if e == "test error"));
    }

    #[tokio::test]
    async fn test_spiral_detection() {
        let (tx, rx) = mpsc::channel(1024);
        // Create a repeating multi-line block that triggers chunk detection
        let block = "The SM axioms are NOT low complexity.\n\
                     The SM axioms are SU(3) x SU(2) x U(1) gauge theories.\n\
                     These are incredibly complex structures.\n\
                     So I will acknowledge that fact in my response.\n\
                     The user's prompt about lowest complexity is false.\n";
        // Send same block many times — this creates matching 150-char chunks
        for _ in 0..20 {
            tx.send(StreamEvent::ThinkingDelta(block.to_string())).await.unwrap();
        }
        drop(tx);

        let result = consume_stream(rx, &mut NullSink).await;
        assert!(matches!(result, ConsumeResult::Spiral { .. }));
    }

    #[tokio::test]
    async fn test_cancel() {
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let mut sink = CancellableNullSink { cancel };
        let (tx, rx) = mpsc::channel(32);
        tx.send(StreamEvent::TextDelta("hello".to_string())).await.unwrap();
        // Don't send Done — cancel should fire
        drop(tx);

        let result = consume_stream(rx, &mut sink).await;
        assert!(matches!(result, ConsumeResult::Cancelled { .. }));
    }

    #[test]
    fn test_trim_largest_tool_result() {
        let mut messages = vec![
            crate::provider::Message::text("user", "hello"),
            crate::provider::Message::tool_result("t1", &"x".repeat(20_000)),
            crate::provider::Message::tool_result("t2", "small result"),
        ];
        trim_largest_tool_result(&mut messages);
        assert!(messages[1].text_content().contains("trimmed"));
        assert_eq!(messages[2].text_content(), "small result");
    }
}
