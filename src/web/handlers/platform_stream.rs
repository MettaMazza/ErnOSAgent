//! Streaming platform inference endpoint — SSE variant of platform_ingest.
//!
//! Returns Server-Sent Events as inference progresses, enabling the platform
//! router to post thinking/tool/audit updates to the Discord thinking thread
//! in real-time rather than all-at-once.

use crate::web::state::AppState;
use axum::extract::State;
use axum::response::sse::{Event, Sse};
use futures_util::stream::Stream;
use std::convert::Infallible;
use std::pin::Pin;

/// SSE streaming platform ingest endpoint.
pub async fn platform_ingest_stream(
    State(state): State<AppState>,
    axum::Json(msg): axum::Json<crate::platform::adapter::PlatformMessage>,
) -> Sse<Pin<Box<dyn Stream<Item = Result<Event, Infallible>> + Send>>> {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        run_streaming_pipeline(state, msg, tx).await;
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(Box::pin(stream))
}

/// Run the full pipeline, emitting SSE events as each stage completes.
async fn run_streaming_pipeline(
    state: AppState,
    msg: crate::platform::adapter::PlatformMessage,
    tx: tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) {
    // Bug fix: session ID must match platform_ingest.rs (3-part format)
    let session_id = format!("{}_{}_{}", msg.platform, msg.user_id, msg.channel_id);

    // Bug fix: create/load session — was completely missing from streaming path
    super::platform_ingest::ensure_session(&state, &session_id, &msg).await;

    // Process platform attachments (security-scoped: admin=disk, non-admin=memory)
    let processed = crate::web::attachment_ingest::process_attachments(
        &msg.attachments, msg.is_admin,
    ).await;
    let (images, attachment_text) = crate::web::attachment_ingest::split_processed(
        &processed, state.model_spec.context_length,
    );
    let content_with_attachments = if attachment_text.is_empty() {
        msg.content.clone()
    } else {
        format!("{}\n\n[ATTACHED FILES]\n{}", msg.content, attachment_text)
    };

    let ctx = crate::web::ws_context::build_chat_context(
        &state, &content_with_attachments, &session_id, None, images, &msg.platform,
    ).await;
    let mut messages = ctx.messages;
    let tools = super::platform_ingest::select_tools(msg.is_admin);
    let provider = state.provider.as_ref();

    // Start inference
    tracing::info!(
        msg_count = messages.len(),
        tools_count = tools.as_array().map(|a| a.len()).unwrap_or(0),
        thinking = state.config.prompt.thinking_enabled,
        session = %session_id,
        "Platform stream: calling provider.chat()"
    );
    let chat_start = std::time::Instant::now();
    let rx_stream = match provider.chat(
        &messages, Some(&tools), state.config.prompt.thinking_enabled,
    ).await {
        Ok(rx) => {
            tracing::info!(
                elapsed_ms = chat_start.elapsed().as_millis() as u64,
                "Platform stream: provider.chat() returned OK"
            );
            rx
        }
        Err(e) => {
            tracing::error!(
                elapsed_ms = chat_start.elapsed().as_millis() as u64,
                error = %e,
                "Platform stream: provider.chat() FAILED"
            );
            let _ = emit(&tx, "error", &serde_json::json!({"error": e.to_string()})).await;
            return;
        }
    };


    // Consume stream using the unified consumer with SSE sink
    use crate::inference::stream_consumer::{self, ConsumeResult, SseSink};
    let mut sink = SseSink { tx: &tx };
    let result = stream_consumer::consume_stream(rx_stream, &mut sink).await;

    // Handle spiral: re-prompt with thinking disabled
    let result = match result {
        ConsumeResult::Spiral { .. } => {
            tracing::info!("Platform stream: re-prompting after spiral");
            let _ = emit(&tx, "spiral_reprompt", &serde_json::json!({})).await;
            stream_consumer::reprompt_after_spiral(
                provider, &mut messages, Some(&tools), &mut sink,
            ).await
        }
        other => other,
    };

    // Dispatch result
    match result {
        ConsumeResult::Reply { ref text, ref thinking } => {
            if let Some(ref t) = thinking {
                let _ = emit(&tx, "thinking_complete", &serde_json::json!({"length": t.len()})).await;
            }
            if text.trim().is_empty() {
                // Empty reply = context overflow. Deliver a diagnostic instead of silence.
                tracing::error!(
                    msg_count = messages.len(),
                    "Platform stream: model returned empty response — context overflow"
                );
                let diagnostic = "The context window is full. Please start a new conversation \
                                  or ask a shorter question to continue.";
                let _ = emit(&tx, "response", &serde_json::json!({
                    "text": diagnostic, "session_id": session_id, "has_plan": false,
                })).await;
            } else {
                emit_reply(&state, provider, &mut messages, &tools, &msg, &session_id, text, &tx).await;
            }
        }
        ConsumeResult::PlanProposal { title, plan_markdown, estimated_turns } => {
            emit_plan(&session_id, &title, &plan_markdown, estimated_turns, &tx).await;
        }
        ConsumeResult::ToolCall { id, name, arguments } => {
            emit_tool_chain(&state, provider, &mut messages, &tools, &msg, &session_id, id, name, arguments, &tx).await;
        }
        ConsumeResult::Escalate { objective, plan, .. } => {
            emit_escalation(&state, provider, messages, &msg, &session_id, &objective, plan, &tx).await;
        }
        ConsumeResult::Error(e) => {
            let _ = emit(&tx, "error", &serde_json::json!({"error": e})).await;
        }
        _ => {}
    }

    let _ = emit(&tx, "done", &serde_json::json!({})).await;
}


// ── Emitters ──────────────────────────────────────────────────────

async fn emit(tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>, event: &str, data: &serde_json::Value) {
    let _ = tx.send(Ok(Event::default().event(event).data(data.to_string()))).await;
}

async fn emit_reply(
    state: &AppState, provider: &dyn crate::provider::Provider,
    messages: &mut Vec<crate::provider::Message>, tools: &serde_json::Value,
    msg: &crate::platform::adapter::PlatformMessage, session_id: &str,
    text: &str, tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) {
    let (audited, audit) = super::platform_ingest::audit_and_capture(
        state, provider, messages, tools, &msg.content, text, session_id,
    ).await;
    let _ = emit(tx, "audit", &serde_json::json!({
        "verdict": audit.verdict, "confidence": audit.confidence,
    })).await;
    let _ = emit(tx, "response", &serde_json::json!({
        "text": audited, "session_id": session_id, "has_plan": false,
    })).await;

    // Bug fix: persist assistant response to session so next turn has history
    crate::web::ws_learning::ingest_assistant_turn(state, &audited, session_id).await;

    // SAE: extract activations from response
    super::sae_capture::spawn_activation_capture(state, &audited);
}

async fn emit_plan(
    session_id: &str, title: &str, plan_md: &str, turns: usize,
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) {
    let plan = crate::web::ws_plans::save_pending_plan(session_id, title, plan_md, turns);
    let _ = emit(tx, "plan", &serde_json::json!({
        "title": plan.title, "plan_markdown": plan.plan_markdown,
    })).await;
    let response = format!(
        "📋 **{}**\n\nI've prepared a plan. Review the details in the thinking thread.",
        plan.title,
    );
    let _ = emit(tx, "response", &serde_json::json!({
        "text": response, "session_id": session_id,
        "has_plan": true, "plan_markdown": plan.plan_markdown,
    })).await;
}

async fn emit_tool_chain(
    state: &AppState, provider: &dyn crate::provider::Provider,
    messages: &mut Vec<crate::provider::Message>, tools: &serde_json::Value,
    msg: &crate::platform::adapter::PlatformMessage, session_id: &str,
    id: String, name: String, arguments: String,
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) {
    let _ = emit(tx, "tool_start", &serde_json::json!({"name": &name})).await;
    let tc = crate::tools::schema::ToolCall { id, name: name.clone(), arguments };

    // Use the proper tool chain that loops until the model produces a reply
    let (reply, _events, audit) = super::platform_exec::run_platform_tool_chain(
        state, provider, messages, tools, &msg.content, session_id, tc, Some(tx),
    ).await;
    // Tool events already emitted live via sse_tx during the loop

    // Emit audit if present
    if let Some(ref audit) = audit {
        let _ = emit(tx, "audit", &serde_json::json!({
            "verdict": audit.verdict, "confidence": audit.confidence,
        })).await;
    }

    // Emit the actual reply
    let _ = emit(tx, "response", &serde_json::json!({
        "text": reply, "session_id": session_id, "has_plan": false,
    })).await;

    // Persist to session
    crate::web::ws_learning::ingest_assistant_turn(state, &reply, session_id).await;

    // SAE: extract activations from response
    super::sae_capture::spawn_activation_capture(state, &reply);
}

async fn emit_escalation(
    state: &AppState, provider: &dyn crate::provider::Provider,
    messages: Vec<crate::provider::Message>,
    msg: &crate::platform::adapter::PlatformMessage, session_id: &str,
    objective: &str, plan: Option<String>,
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) {
    let _ = emit(tx, "escalate", &serde_json::json!({"objective": objective})).await;
    let (reply, _events, _audit) = super::platform_exec::run_platform_react(
        state, provider, messages, objective, plan.as_deref(), &msg.content, session_id, Some(tx),
    ).await;
    // Tool events already emitted live via sse_tx during the loop
    let _ = emit(tx, "response", &serde_json::json!({
        "text": reply, "session_id": session_id, "has_plan": false,
    })).await;

    // Persist escalation reply to session
    crate::web::ws_learning::ingest_assistant_turn(state, &reply, session_id).await;

    // SAE: extract activations from response
    super::sae_capture::spawn_activation_capture(state, &reply);
}
