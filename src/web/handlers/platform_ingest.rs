//! Platform inference pipeline — processes messages from platform adapters.
//!
//! Runs the full inference pipeline (context assembly, L1/L2 inference,
//! tool execution, observer audit) for messages ingested from Discord,
//! Telegram, or any other platform adapter. Returns JSON including
//! tool execution metadata and audit results for thinking thread display.

use crate::web::state::AppState;
use axum::{extract::State, Json};
use serde::Serialize;

/// A single tool execution event, captured for thinking thread display.
#[derive(Debug, Clone, Serialize)]
pub struct ToolEvent {
    pub name: String,
    pub success: bool,
    pub elapsed_ms: u64,
    pub output_preview: String,
}

/// Observer audit summary, captured for thinking thread display.
#[derive(Debug, Clone, Serialize)]
pub struct AuditSummary {
    pub verdict: String,
    pub confidence: f32,
    pub retries: usize,
    pub active_topic: String,
}

impl AuditSummary {
    fn skipped() -> Self {
        Self { verdict: "Skipped".into(), confidence: 0.0, retries: 0, active_topic: String::new() }
    }
    fn error(retries: usize) -> Self {
        Self { verdict: "Error".into(), confidence: 0.0, retries, active_topic: String::new() }
    }
}

/// POST /api/chat/platform — ingest a message from a platform adapter.
/// Full inference pipeline: context assembly, tool execution, observer audit.
/// Returns tool events and audit metadata for thinking thread observability.
pub async fn platform_ingest(
    State(state): State<AppState>,
    Json(msg): Json<crate::platform::adapter::PlatformMessage>,
) -> Json<serde_json::Value> {
    tracing::info!(
        platform = %msg.platform,
        user = %msg.user_name,
        is_admin = msg.is_admin,
        content_len = msg.content.len(),
        "Platform message ingested"
    );

    let session_id = format!("{}_{}_{}", msg.platform, msg.user_id, msg.channel_id);
    ensure_session(&state, &session_id, &msg).await;

    // Process platform attachments (security-scoped: admin=disk, non-admin=memory)
    let processed = crate::web::attachment_ingest::process_attachments(
        &msg.attachments, msg.is_admin,
    ).await;
    let (images, attachment_text) = crate::web::attachment_ingest::split_processed(
        &processed, state.model_spec.context_length,
    );
    // Deep-read: if admin attachment exceeds inline budget, summarise page-by-page.
    // Track which attachments get deep-read so we can REPLACE their inline text with the digest.
    let mut deep_read_digests: Vec<(String, String)> = Vec::new();
    if msg.is_admin {
        for att in &processed {
            if let (Some(ref path), true) = (&att.saved_path, att.exceeds_budget(state.model_spec.context_length)) {
                let config = crate::web::attachment_reader::DeepReadConfig {
                    path: path.clone(),
                    filename: att.filename.clone(),
                    context_length: state.model_spec.context_length,
                };
                let digest = crate::web::attachment_reader::deep_read(
                    config, state.provider.as_ref(), &state.memory, None,
                ).await;
                deep_read_digests.push((att.filename.clone(), digest));
            }
        }
    }

    // Build the final content: user message + attachments.
    // For deep-read files, the digest REPLACES inline text (not stacked on top).
    let content_with_attachments = if deep_read_digests.is_empty() {
        if attachment_text.is_empty() {
            msg.content.clone()
        } else {
            format!("{}\n\n[ATTACHED FILES]\n{}", msg.content, attachment_text)
        }
    } else {
        let mut content = msg.content.clone();
        for (filename, digest) in &deep_read_digests {
            content.push_str(&format!("\n\n[FILE: {} — processed via deep-read]\n{}", filename, digest));
        }
        content
    };

    // Select tools BEFORE building context so consolidation can account for tool overhead
    let tools = select_tools(msg.is_admin);
    let tools_chars = tools.to_string().len();

    let ctx = crate::web::ws_context::build_chat_context(
        &state, &content_with_attachments, &session_id, None, images, &msg.platform, tools_chars,
    ).await;
    let mut messages = ctx.messages;
    let provider = state.provider.as_ref();

    let rx = match provider.chat(&messages, Some(&tools), state.config.prompt.thinking_enabled).await {
        Ok(rx) => rx,
        Err(e) => return build_error_response(&msg.platform, &e),
    };

    use crate::inference::stream_consumer::{self as sc, NullSink};
    let mut sink = NullSink;
    let result = sc::consume_stream(rx, &mut sink).await;

    // Handle spiral: re-prompt
    let result = match result {
        sc::ConsumeResult::Spiral { .. } => {
            sc::reprompt_after_spiral(provider, &mut messages, Some(&tools), &mut sink).await
        }
        other => other,
    };

    let (response, thinking_content, tool_events, audit_summary, has_plan, plan_markdown) = dispatch_result(
        &state, provider, &mut messages, &tools, &msg, &session_id, result,
    ).await;

    Json(serde_json::json!({
        "success": true,
        "response": response,
        "thinking": thinking_content,
        "tool_events": tool_events,
        "audit": audit_summary,
        "session_id": session_id,
        "has_plan": has_plan,
        "plan_markdown": plan_markdown,
        "platform": msg.platform,
        "channel_id": msg.channel_id,
        "message_id": msg.message_id,
    }))
}

/// Build error response when initial inference fails.
fn build_error_response(platform: &str, e: &anyhow::Error) -> Json<serde_json::Value> {
    tracing::error!(error = %e, platform = %platform, "Platform inference failed");
    Json(serde_json::json!({
        "success": false,
        "error": e.to_string(),
    }))
}

/// Dispatch the L1 inference result to the appropriate handler.
async fn dispatch_result(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    messages: &mut Vec<crate::provider::Message>,
    tools: &serde_json::Value,
    msg: &crate::platform::adapter::PlatformMessage,
    session_id: &str,
    result: crate::inference::stream_consumer::ConsumeResult,
) -> (String, Option<String>, Vec<ToolEvent>, Option<AuditSummary>, bool, Option<String>) {
    use crate::inference::stream_consumer::ConsumeResult;
    match result {
        ConsumeResult::Reply { text, thinking } => {
            let (audited, audit) = audit_and_capture(
                state, provider, messages, tools, &msg.content, &text, session_id,
            ).await;
            crate::web::ws_learning::ingest_assistant_turn(state, &audited, session_id).await;
            crate::web::ws_learning::spawn_insight_extraction(state, &msg.content, &audited);
            (audited, thinking, Vec::new(), Some(audit), false, None)
        }
        ConsumeResult::Escalate { objective, plan, .. } => {
            let (reply, thinking, events, audit) = handle_escalation(
                state, provider, messages.clone(), msg, &objective, plan.as_deref(), session_id,
            ).await;
            (reply, thinking, events, audit, plan.is_some(), None)
        }
        ConsumeResult::ToolCall { id, name, arguments } => {
            // Intercept propose_plan: save the plan and surface it for the platform
            if name == "propose_plan" {
                return handle_plan_proposal(state, &arguments, session_id).await;
            }
            let tc = crate::tools::schema::ToolCall { id, name, arguments };
            let (reply, events, audit) = super::platform_exec::run_platform_tool_chain(
                state, provider, messages, tools, &msg.content, session_id, tc, None,
            ).await;
            crate::web::ws_learning::ingest_assistant_turn(state, &reply, session_id).await;
            (reply, None, events, audit, false, None)
        }
        ConsumeResult::ToolCalls(calls) => {
            // Execute all tool calls, then continue with tool chain from the last one
            let mut all_events = Vec::new();
            let mut tcs: Vec<crate::tools::schema::ToolCall> = calls.into_iter()
                .map(|(id, name, arguments)| crate::tools::schema::ToolCall { id, name, arguments })
                .collect();
            let last_tc = match tcs.pop() {
                Some(tc) => tc,
                None => return ("No tool calls.".to_string(), None, Vec::new(), None, false, None),
            };
            // Execute all but the last
            for tc in &tcs {
                let (result, event) = super::platform_exec::execute_and_capture(state, tc).await;
                all_events.push(event);
                super::platform_exec::append_tool_messages(messages, tc, &result);
            }
            // Run tool chain starting from the last tool call
            let (reply, events, audit) = super::platform_exec::run_platform_tool_chain(
                state, provider, messages, tools, &msg.content, session_id, last_tc, None,
            ).await;
            all_events.extend(events);
            crate::web::ws_learning::ingest_assistant_turn(state, &reply, session_id).await;
            (reply, None, all_events, audit, false, None)
        }
        ConsumeResult::Error(e) => {
            tracing::error!(error = %e, platform = %msg.platform, "Stream consumption failed");
            (format!("An error occurred: {}", e), None, Vec::new(), None, false, None)
        }
        _ => {
            ("Unexpected result.".to_string(), None, Vec::new(), None, false, None)
        }
    }
}

/// Handle a propose_plan tool call: save the plan and return it for the platform.
async fn handle_plan_proposal(
    _state: &AppState,
    arguments: &str,
    session_id: &str,
) -> (String, Option<String>, Vec<ToolEvent>, Option<AuditSummary>, bool, Option<String>) {
    let parsed: serde_json::Value = serde_json::from_str(arguments).unwrap_or_default();
    let title = parsed["title"].as_str().unwrap_or("Plan");
    let plan_md = parsed["plan_markdown"].as_str().unwrap_or("");
    let turns = parsed["estimated_turns"].as_u64().unwrap_or(10) as usize;

    let plan = crate::web::ws_plans::save_pending_plan(
        session_id, title, plan_md, turns.max(3).min(50),
    );
    tracing::info!(
        title = %plan.title, turns = plan.estimated_turns,
        "Platform plan proposal saved — awaiting approval"
    );

    let response = format!(
        "📋 **{}**\n\nI've prepared a plan. Review the details in the thinking thread.",
        plan.title,
    );
    (response, None, Vec::new(), None, true, Some(plan.plan_markdown))
}

/// Handle L2 escalation from L1 inference.
async fn handle_escalation(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    messages: Vec<crate::provider::Message>,
    msg: &crate::platform::adapter::PlatformMessage,
    objective: &str,
    plan: Option<&str>,
    session_id: &str,
) -> (String, Option<String>, Vec<ToolEvent>, Option<AuditSummary>) {
    if !msg.is_admin {
        return ("I can't perform complex multi-step tasks for non-admin users. Please ask an admin.".to_string(),
                None, Vec::new(), None);
    }
    let (reply, events, audit) = super::platform_exec::run_platform_react(
        state, provider, messages, objective, plan, &msg.content, session_id, None,
    ).await;
    crate::web::ws_learning::ingest_assistant_turn(state, &reply, session_id).await;
    (reply, None, events, audit)
}

/// Ensure a session exists for this platform + user + channel combination.
pub(crate) async fn ensure_session(
    state: &AppState,
    session_id: &str,
    msg: &crate::platform::adapter::PlatformMessage,
) {
    let mut sessions = state.sessions.write().await;
    if sessions.get(session_id).is_none() {
        let mut session = crate::session::Session::new();
        session.id = session_id.to_string();
        session.title = format!("{} — {}", msg.platform, msg.user_name);
        if let Err(e) = sessions.update(&session) {
            tracing::warn!(error = %e, "Failed to persist new platform session");
        }
        sessions.list(); // ensure loaded
    }
}

/// Select tool schema based on admin status.
pub(crate) fn select_tools(is_admin: bool) -> serde_json::Value {
    if is_admin {
        crate::tools::schema::layer1_tools()
    } else {
        crate::tools::schema::platform_safe_tools()
    }
}

/// Run observer audit and capture training signals + conversation stack.
/// Returns the audited text and an audit summary for the thinking thread.
pub async fn audit_and_capture(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    messages: &mut Vec<crate::provider::Message>,
    tools: &serde_json::Value,
    user_query: &str,
    initial_text: &str,
    session_id: &str,
) -> (String, AuditSummary) {
    if !state.config.observer.enabled || initial_text.is_empty() {
        return (initial_text.to_string(), AuditSummary::skipped());
    }

    let tool_context = super::platform_exec::build_tool_context(messages);
    let mut current_text = initial_text.to_string();
    let mut retries: usize = 0;

    // No cap — the model MUST produce an approved response.
    // If this loops, it means the observer feedback isn't being followed,
    // which is a deeper bug to fix — not mask with a bailout.
    loop {
        match crate::observer::audit_response(
            provider, messages, &current_text, &tool_context, user_query,
        ).await {
            Ok(output) if output.result.verdict.is_allowed() => {
                crate::observer::persist_audit_result(&state.config.general.data_dir, &output.result);
                return handle_approved(state, user_query, &current_text, session_id, &output.result, retries);
            }
            Ok(output) => {
                retries += 1;
                let rejected = current_text.clone();
                tracing::warn!(
                    retries,
                    category = %output.result.failure_category,
                    what_went_wrong = %output.result.what_went_wrong,
                    how_to_fix = %output.result.how_to_fix,
                    "Observer BLOCKED — re-inferring with feedback"
                );
                crate::observer::persist_audit_result(&state.config.general.data_dir, &output.result);
                current_text = retry_after_rejection(
                    state, provider, messages, tools, user_query, session_id, &rejected, &output.result,
                ).await;
            }
            Err(e) => {
                // Infrastructure error (observer itself is down) — fail-open.
                // This is NOT a quality issue — the response was never evaluated.
                tracing::warn!(error = %e, "Platform observer failed — fail-open");
                return (current_text, AuditSummary::error(retries));
            }
        }
    }
}

/// Handle approved audit — capture training signal and save conversation stack.
fn handle_approved(
    state: &AppState,
    user_query: &str,
    text: &str,
    session_id: &str,
    result: &crate::observer::AuditResult,
    retries: usize,
) -> (String, AuditSummary) {
    crate::web::training_capture::capture_approved_with_flags(
        state, user_query, text, result.confidence, &result.positive_flags,
    );
    crate::web::ws_stream::save_conversation_stack(state, session_id, result);
    (text.to_string(), AuditSummary {
        verdict: "Allowed".to_string(),
        confidence: result.confidence,
        retries,
        active_topic: result.active_topic.clone(),
    })
}


/// Retry inference after observer rejection.
/// Handles tool calls in the retry response — the model may follow observer
/// guidance by calling file_read/codebase_search instead of generating text.
///
/// Uses Box::pin indirection because this is part of a recursive async cycle:
/// retry_after_rejection → run_platform_tool_chain → audit_and_capture → retry_after_rejection
fn retry_after_rejection<'a>(
    state: &'a AppState,
    provider: &'a dyn crate::provider::Provider,
    messages: &'a mut Vec<crate::provider::Message>,
    tools: &'a serde_json::Value,
    user_query: &'a str,
    session_id: &'a str,
    rejected_text: &'a str,
    result: &'a crate::observer::AuditResult,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = String> + Send + 'a>> {
    Box::pin(async move {
        let feedback = crate::observer::format_rejection_feedback(result);
        messages.push(crate::provider::Message::text("assistant", rejected_text));
        messages.push(crate::provider::Message::text("system", &feedback));
        super::platform_exec::enforce_context_budget(provider, messages, Some(tools), state.model_spec.context_length, true).await;

        if let Ok(rx) = provider.chat(messages, Some(tools), true).await {
            use crate::inference::stream_consumer::{self as sc, NullSink};
            let mut sink = NullSink;
            match sc::consume_stream(rx, &mut sink).await {
                sc::ConsumeResult::Reply { text, .. } => {
                    crate::web::training_capture::capture_rejection(
                        state, user_query, rejected_text, &text, &result.what_went_wrong,
                    );
                    return text;
                }
                sc::ConsumeResult::ToolCall { id, name, arguments } => {
                    // Model is following observer guidance — execute the tool chain
                    tracing::info!(
                        tool = %name, retries = ?result.failure_category,
                        "Observer retry: model called tool (following feedback) — executing"
                    );
                    let tc = crate::tools::schema::ToolCall { id, name, arguments };
                    let (reply, _events, _audit) = super::platform_exec::run_platform_tool_chain(
                        state, provider, messages, tools, user_query, session_id, tc, None,
                    ).await;
                    return reply;
                }
                sc::ConsumeResult::ToolCalls(calls) => {
                    tracing::info!(
                        count = calls.len(),
                        "Observer retry: model called multiple tools (following feedback) — executing"
                    );
                    let mut tcs: Vec<crate::tools::schema::ToolCall> = calls.into_iter()
                        .map(|(id, name, arguments)| crate::tools::schema::ToolCall { id, name, arguments })
                        .collect();
                    if let Some(last_tc) = tcs.pop() {
                        for tc in &tcs {
                            let (result, _event) = super::platform_exec::execute_and_capture(state, tc).await;
                            super::platform_exec::append_tool_messages(messages, tc, &result);
                        }
                        let (reply, _events, _audit) = super::platform_exec::run_platform_tool_chain(
                            state, provider, messages, tools, user_query, session_id, last_tc, None,
                        ).await;
                        return reply;
                    }
                }
                _ => {} // Empty/Error — fall through to rejected_text
            }
        }

        rejected_text.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_id_scoping() {
        let session_id = format!("{}_{}_{}", "discord", "user123", "channel456");
        assert_eq!(session_id, "discord_user123_channel456");
    }

    #[test]
    fn test_select_tools_admin() {
        let tools = select_tools(true);
        assert!(tools.is_array());
    }

    #[test]
    fn test_select_tools_non_admin() {
        let tools = select_tools(false);
        assert!(tools.is_array());
    }

    #[test]
    fn test_tool_event_serialize() {
        let event = ToolEvent {
            name: "web_search".to_string(),
            success: true,
            elapsed_ms: 1234,
            output_preview: "Found 5 results".to_string(),
        };
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["name"], "web_search");
        assert_eq!(json["success"], true);
        assert_eq!(json["elapsed_ms"], 1234);
    }

    #[test]
    fn test_audit_summary_serialize() {
        let summary = AuditSummary {
            verdict: "Allowed".to_string(),
            confidence: 8.5,
            retries: 0,
            active_topic: "Testing serialization".to_string(),
        };
        let json = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["verdict"], "Allowed");
        assert_eq!(json["confidence"], 8.5);
    }
}
