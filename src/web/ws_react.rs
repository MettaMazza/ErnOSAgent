//! WebSocket ReAct loop execution — runs the full agentic tool loop.

use crate::inference::react_loop::{self, ReactContext, IterationResult};
use crate::observer;
use crate::provider::Message;
use crate::tools::schema;
use crate::web::state::AppState;
use crate::web::ws_learning::{ingest_assistant_turn, spawn_insight_extraction};
use crate::web::ws_stream::send_ws;
use axum::extract::ws::{Message as WsMessage, WebSocket};

/// Run the full ReAct loop with tool execution and observer audit.
pub async fn run_react_loop(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    base_messages: Vec<Message>,
    objective: &str,
    plan: Option<&str>,
    planned_turns: usize,
    user_query: &str,
    session_id: &str,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    stop_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let mut ctx = ReactContext::new(objective, plan, base_messages);
    let mut remaining_turns = planned_turns;
    let mut total_iterations = 0usize;
    let mut budget_exhausted_prompted = false;
    let mut consecutive_rejections: usize = 0;
    // Spiral detection: track consecutive identical failures
    let mut last_fail_signature: Option<String> = None;
    let mut consecutive_fails: usize = 0;

    loop {
        // ─── User Stop Check ───
        if stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(iteration = total_iterations, "ReAct loop stopped by user");
            ctx.messages.push(Message::text("system",
                "[USER INTERRUPT] The user has stopped this loop. \
                 Summarize everything you have gathered so far and deliver your \
                 best response using reply_request. Do NOT call any more tools."
            ));
            match react_loop::run_iteration(provider, &ctx, true).await {
                Ok(IterationResult::Reply(reply, thinking)) | Ok(IterationResult::ImplicitReply(reply, thinking)) => {
                    if let Some(ref t) = thinking {
                        send_ws(sender, "thinking_delta", &serde_json::json!({"content": t})).await;
                    }
                    send_ws(sender, "text_delta", &serde_json::json!({"content": &reply})).await;
                    ingest_assistant_turn(state, &reply, session_id).await;
                }
                _ => {
                    send_ws(sender, "text_delta", &serde_json::json!({"content": "(Loop stopped by user)"})).await;
                }
            }
            send_ws(sender, "done", &serde_json::json!({})).await;
            return;
        }

        // ─── Turn Budget Exhausted ───
        if remaining_turns == 0 && !budget_exhausted_prompted {
            budget_exhausted_prompted = true;
            tracing::info!(total = total_iterations, "ReAct turns exhausted — forcing assessment");
            ctx.messages.push(Message::text("system",
                &format!(
                    "[TURN BUDGET EXHAUSTED] You have used all {} planned turns.\n\n\
                     STOP. Assess your progress:\n\
                     - What have you accomplished so far?\n\
                     - Do you have enough information to answer the user?\n\n\
                     You have exactly TWO options:\n\
                     1. Call `reply_request` with your complete response if you have enough.\n\
                     2. Call `extend_turns` with a progress summary, what work remains, and a NEW turn estimate.\n\n\
                     You MUST NOT call any other tool. Assess and decide.",
                    total_iterations
                )
            ));
        }

        tracing::info!(iteration = total_iterations, remaining = remaining_turns, "ReAct iteration");

        match react_loop::run_iteration(provider, &ctx, true).await {
            Ok(IterationResult::Reply(reply, thinking)) => {
                if let Some(ref t) = thinking {
                    send_ws(sender, "thinking_delta", &serde_json::json!({"content": t})).await;
                }
                if state.config.observer.enabled {
                    send_ws(sender, "audit_running", &serde_json::json!({})).await;
                    let tool_context = build_tool_context(&ctx.messages);
                    match observer::audit_response(provider, &ctx.messages, &reply, &tool_context, user_query).await {
                        Ok(output) if !output.result.verdict.is_allowed() => {
                            consecutive_rejections += 1;
                            tracing::info!(
                                category = %output.result.failure_category,
                                reason = %output.result.what_went_wrong,
                                consecutive = consecutive_rejections,
                                "ReAct reply rejected — retrying"
                            );

                            // Bailout after 2 consecutive rejections
                            if consecutive_rejections >= 2 {
                                tracing::warn!(rejections = consecutive_rejections, "ReAct observer bailout — forcing response");
                                let bailout = observer::format_bailout_override(consecutive_rejections);
                                ctx.messages.push(Message::text("system", &bailout));
                                // Fall through to deliver the reply
                            } else {
                                // Push the rejected reply so the model can see what it said
                                ctx.messages.push(Message::text("assistant", &reply));
                                ctx.add_rejection_feedback(&output.result);
                                continue;
                            }
                        }
                        Ok(output) => {
                            send_ws(sender, "audit_completed", &serde_json::json!({
                                "approved": true,
                                "confidence": output.result.confidence,
                                "category": &output.result.failure_category,
                            })).await;
                        }
                        Err(e) => tracing::warn!(error = %e, "Observer failed — fail-open"),
                    }
                }
                send_ws(sender, "text_delta", &serde_json::json!({"content": &reply})).await;
                ingest_assistant_turn(state, &reply, session_id).await;
                spawn_insight_extraction(state, user_query, &reply);
                send_ws(sender, "done", &serde_json::json!({})).await;
                return;
            }
            Ok(IterationResult::Refuse(reason)) => {
                send_ws(sender, "text_delta", &serde_json::json!({"content": format!("I cannot complete this: {}", reason)})).await;
                send_ws(sender, "done", &serde_json::json!({})).await;
                return;
            }
            Ok(IterationResult::ExtendTurns { additional, progress, remaining_work }) => {
                let granted = additional;
                remaining_turns = granted;
                budget_exhausted_prompted = false;
                tracing::info!(
                    granted, progress = %progress,
                    remaining_work = %remaining_work,
                    "ReAct extension granted after assessment"
                );
                send_ws(sender, "status", &serde_json::json!({
                    "message": format!("ReAct extended (+{} turns after assessment)", granted)
                })).await;
                ctx.messages.push(Message::text("system",
                    &format!(
                        "[EXTENSION GRANTED] You have been granted {} additional turns. \
                         Continue with your plan. Remaining work: {}",
                        granted, remaining_work
                    )
                ));
            }
            Ok(IterationResult::ToolCall(tc)) => {
                if remaining_turns == 0 {
                    tracing::warn!(tool = %tc.name, "Model tried to call tool with 0 budget — rejecting");
                    ctx.messages.push(Message::assistant_tool_call(&tc.id, &tc.name, &tc.arguments));
                    ctx.messages.push(Message::tool_result(&tc.id,
                        "[REJECTED] Your turn budget is exhausted. You must call `reply_request` \
                         to deliver your response, or `extend_turns` to request more turns. \
                         No other tools are available until you assess and decide."
                    ));
                    continue;
                }
                send_ws(sender, "tool_executing", &serde_json::json!({"name": &tc.name, "id": &tc.id})).await;
                let result = if tc.name == "spawn_sub_agent" {
                    execute_sub_agent(state, provider, &tc, sender).await
                } else {
                    execute_tool_with_state(state, &tc).await
                };
                send_ws(sender, "tool_completed", &serde_json::json!({
                    "id": &tc.id, "name": &tc.name,
                    "result": &result.output, "success": result.success,
                })).await;
                // Emit artifact card if this was a successful create_artifact call
                if tc.name == "create_artifact" && result.success {
                    if let Ok(artifact) = serde_json::from_str::<serde_json::Value>(&result.output) {
                        send_ws(sender, "artifact_created", &artifact).await;
                    }
                }
                // Spiral detection: track consecutive identical failures
                if !result.success {
                    let sig = format!("{}:{}", tc.name, result.output);
                    if last_fail_signature.as_deref() == Some(&sig) {
                        consecutive_fails += 1;
                    } else {
                        last_fail_signature = Some(sig);
                        consecutive_fails = 1;
                    }
                    if consecutive_fails >= 3 {
                        tracing::warn!(
                            tool = %tc.name, consecutive = consecutive_fails,
                            "Degenerate tool spiral detected — injecting feedback"
                        );
                        ctx.messages.push(Message::text("system", &format!(
                            "[PATTERN DETECTED] You have called `{}` {} times in a row with the same \
                             error: \"{}\". Stop calling it. Reassess your approach and try a different \
                             strategy, or call `reply_request` with what you have.",
                            tc.name, consecutive_fails, result.output
                        )));
                    }
                } else {
                    last_fail_signature = None;
                    consecutive_fails = 0;
                }
                ctx.add_tool_result(&tc, result);
                remaining_turns = remaining_turns.saturating_sub(1);
                total_iterations += 1;
            }
            Ok(IterationResult::ToolCalls(tcs)) => {
                if remaining_turns == 0 {
                    tracing::warn!(count = tcs.len(), "Model tried parallel calls with 0 budget — rejecting all");
                    let call_refs: Vec<_> = tcs.iter()
                        .map(|tc| (tc.id.as_str(), tc.name.as_str(), tc.arguments.as_str()))
                        .collect();
                    ctx.messages.push(Message::assistant_tool_calls(&call_refs));
                    for tc in &tcs {
                        ctx.messages.push(Message::tool_result(&tc.id,
                            "[REJECTED] Turn budget exhausted. Call reply_request or extend_turns."
                        ));
                    }
                    continue;
                }
                tracing::info!(
                    count = tcs.len(),
                    tools = %tcs.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "),
                    "Executing parallel tool calls"
                );
                for tc in &tcs {
                    send_ws(sender, "tool_executing", &serde_json::json!({"name": &tc.name, "id": &tc.id})).await;
                }
                let futs: Vec<_> = tcs.iter()
                    .map(|tc| execute_tool_with_state(state, tc))
                    .collect();
                let results = futures::future::join_all(futs).await;
                for (i, result) in results.iter().enumerate() {
                    send_ws(sender, "tool_completed", &serde_json::json!({
                        "id": &result.tool_call_id, "name": &result.name,
                        "result": &result.output, "success": result.success,
                    })).await;
                    if tcs[i].name == "create_artifact" && result.success {
                        if let Ok(artifact) = serde_json::from_str::<serde_json::Value>(&result.output) {
                            send_ws(sender, "artifact_created", &artifact).await;
                        }
                    }
                }
                // Reset spiral tracker on parallel calls (mixed success/fail)
                last_fail_signature = None;
                consecutive_fails = 0;
                let pairs: Vec<_> = tcs.iter().zip(results.into_iter()).collect();
                ctx.add_tool_results(pairs);
                remaining_turns = remaining_turns.saturating_sub(1);
                total_iterations += 1;
            }
            Ok(IterationResult::ImplicitReply(text, thinking)) => {
                if let Some(ref t) = thinking {
                    send_ws(sender, "thinking_delta", &serde_json::json!({"content": t})).await;
                }
                send_ws(sender, "text_delta", &serde_json::json!({"content": &text})).await;
                ingest_assistant_turn(state, &text, session_id).await;
                spawn_insight_extraction(state, user_query, &text);
                send_ws(sender, "done", &serde_json::json!({})).await;
                return;
            }
            Err(e) => {
                tracing::error!(error = ?e, iteration = total_iterations, "ReAct iteration inference failed (full chain)");
                send_ws(sender, "error", &serde_json::json!({"message": format!("ReAct error: {}", e)})).await;
                send_ws(sender, "done", &serde_json::json!({})).await;
                return;
            }
        }
    }
}

/// Execute a tool call, dispatching memory tools through AppState.
async fn execute_tool_with_state(
    state: &AppState,
    tc: &schema::ToolCall,
) -> schema::ToolResult {
    crate::web::tool_dispatch::execute_tool_with_state(state, tc).await
}

/// Execute a spawn_sub_agent tool call — runs an isolated ReAct loop.
async fn execute_sub_agent(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    tc: &schema::ToolCall,
    sender: &mut futures_util::stream::SplitSink<axum::extract::ws::WebSocket, axum::extract::ws::Message>,
) -> schema::ToolResult {
    let args = tc.args();
    let task = args["task"].as_str().unwrap_or("").to_string();
    let tools: Vec<String> = args["tools"].as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let max_turns = args["max_turns"].as_u64().unwrap_or(5) as usize;

    if task.is_empty() || tools.is_empty() {
        return schema::ToolResult {
            tool_call_id: tc.id.clone(),
            name: tc.name.clone(),
            output: "Error: sub-agent requires non-empty 'task' and 'tools' array".to_string(),
            success: false,
        };
    }

    send_ws(sender, "status", &serde_json::json!({
        "message": format!("Sub-agent spawned: {} (tools: {}, max {} turns)", task, tools.join(", "), max_turns)
    })).await;

    let config = crate::inference::sub_agent::SubAgentConfig {
        task: task.clone(),
        allowed_tools: tools,
        max_turns,
    };

    match crate::inference::sub_agent::run_sub_agent(provider, config, state).await {
        Ok(result) => {
            tracing::info!(
                task = %task,
                success = result.success,
                turns = result.turns_used,
                tools_called = %result.tool_calls_made.join(", "),
                "Sub-agent completed"
            );
            schema::ToolResult {
                tool_call_id: tc.id.clone(),
                name: tc.name.clone(),
                output: format!(
                    "[Sub-Agent Result]\nSuccess: {}\nTurns: {}\nTools: {}\n\n{}",
                    result.success, result.turns_used,
                    result.tool_calls_made.join(", "),
                    result.summary
                ),
                success: result.success,
            }
        }
        Err(e) => {
            tracing::error!(error = %e, task = %task, "Sub-agent failed");
            schema::ToolResult {
                tool_call_id: tc.id.clone(),
                name: tc.name.clone(),
                output: format!("Sub-agent error: {}", e),
                success: false,
            }
        }
    }
}

/// Build a concise tool context string from the message history for the observer.
/// Scans for tool result messages and matches them to their assistant tool_call,
/// producing a summary like: "[1] tool_name → result_preview"
fn build_tool_context(messages: &[Message]) -> String {
    let mut entries: Vec<String> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == "tool" {
            let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("");
            let result_text = msg.text_content();
            let result_preview = if result_text.len() > 200 {
                format!("{}...", &result_text[..200])
            } else {
                result_text.to_string()
            };

            // Find the matching tool call in preceding assistant messages
            let mut tool_name = "unknown".to_string();
            for j in (0..i).rev() {
                if messages[j].role == "assistant" {
                    if let Some(tcs) = &messages[j].tool_calls {
                        for tc in tcs {
                            if tc["id"].as_str() == Some(tool_call_id) {
                                tool_name = tc["function"]["name"]
                                    .as_str()
                                    .unwrap_or("unknown")
                                    .to_string();
                                break;
                            }
                        }
                        if tool_name != "unknown" { break; }
                    }
                }
            }

            entries.push(format!("[{}] {} → {}", entries.len() + 1, tool_name, result_preview));
        }
    }

    if entries.is_empty() {
        String::new()
    } else {
        format!("Tools executed this session ({} calls):\n{}", entries.len(), entries.join("\n"))
    }
}

