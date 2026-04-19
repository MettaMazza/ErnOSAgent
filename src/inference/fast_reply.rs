// Ern-OS — Layer 1: Fast Reply — standard tool-equipped inference
//! Single-turn inference with tool support. If the model calls
//! `start_react_system`, the router escalates to Layer 2.

use crate::provider::{Message, Provider, StreamEvent};
use crate::tools::schema;
use anyhow::{Context, Result};
use tokio::sync::mpsc;

/// Result of a fast reply inference.
pub enum FastReplyResult {
    /// Direct text response — no escalation needed.
    Reply {
        text: String,
        thinking: Option<String>,
    },
    /// Model wants to escalate to the ReAct loop.
    Escalate {
        objective: String,
        plan: Option<String>,
    },
    /// Model made a tool call that needs execution + re-inference.
    ToolCall {
        id: String,
        name: String,
        arguments: String,
    },
}

/// Run Layer 1 inference — single streaming call with Layer 1 tools.
pub async fn run(
    provider: &dyn Provider,
    messages: &[Message],
    thinking_enabled: bool,
) -> Result<(FastReplyResult, mpsc::Receiver<StreamEvent>)> {
    let tools = schema::layer1_tools();

    let rx = provider
        .chat(messages, Some(&tools), thinking_enabled)
        .await
        .context("Layer 1 inference failed")?;

    Ok((FastReplyResult::Reply {
        text: String::new(),
        thinking: None,
    }, rx))
}

/// Consume the stream and determine the result.
pub async fn consume_stream(
    mut rx: mpsc::Receiver<StreamEvent>,
    tx_ws: Option<&mpsc::Sender<StreamEvent>>,
) -> Result<FastReplyResult> {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut tool_calls: Vec<(String, String, String)> = Vec::new();

    while let Some(event) = rx.recv().await {
        match &event {
            StreamEvent::TextDelta(delta) => {
                text.push_str(delta);
            }
            StreamEvent::ThinkingDelta(delta) => {
                thinking.push_str(delta);
            }
            StreamEvent::ToolCall { id, name, arguments } => {
                tool_calls.push((id.clone(), name.clone(), arguments.clone()));
            }
            StreamEvent::Done => break,
            StreamEvent::Error(e) => {
                anyhow::bail!("Stream error: {}", e);
            }
        }

        // Forward to WebSocket if connected
        if let Some(ws_tx) = tx_ws {
            let _ = ws_tx.send(event).await;
        }
    }

    // Check for start_react_system escalation
    if let Some((_, _, args)) = tool_calls.iter().find(|(_, name, _)| name == "start_react_system") {
        let parsed: serde_json::Value = serde_json::from_str(args).unwrap_or_default();
        return Ok(FastReplyResult::Escalate {
            objective: parsed["objective"].as_str().unwrap_or("").to_string(),
            plan: parsed["plan"].as_str().map(|s| s.to_string()),
        });
    }

    // Check for other tool calls
    if let Some((id, name, arguments)) = tool_calls.into_iter().next() {
        return Ok(FastReplyResult::ToolCall { id, name, arguments });
    }

    Ok(FastReplyResult::Reply {
        text,
        thinking: if thinking.is_empty() { None } else { Some(thinking) },
    })
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

        let result = consume_stream(rx, None).await.unwrap();
        assert!(matches!(result, FastReplyResult::Reply { text, .. } if text == "Hello world"));
    }

    #[tokio::test]
    async fn test_consume_escalation() {
        let (tx, rx) = mpsc::channel(32);
        tx.send(StreamEvent::ToolCall {
            id: "c1".into(),
            name: "start_react_system".into(),
            arguments: r#"{"objective":"deploy app"}"#.into(),
        }).await.unwrap();
        tx.send(StreamEvent::Done).await.unwrap();
        drop(tx);

        let result = consume_stream(rx, None).await.unwrap();
        assert!(matches!(result, FastReplyResult::Escalate { objective, .. } if objective == "deploy app"));
    }
}
