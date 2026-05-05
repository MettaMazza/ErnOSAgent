// Ern-OS — Tool dispatch for WebSocket handler.
//! Dispatches tool calls through AppState, routing memory-dependent tools
//! to the shared MemoryManager via Arc<RwLock>.

use crate::tools::schema;
use crate::web::state::AppState;

/// Execute a tool call, dispatching memory tools through AppState.
pub async fn execute_tool_with_state(
    state: &AppState,
    tc: &schema::ToolCall,
) -> schema::ToolResult {
    // Validate arguments are parseable JSON before dispatching.
    // Defense-in-depth: the stream parser should strip control tokens and
    // skip corrupt calls, but if one still gets through, reject here with
    // a structured error so the model can re-emit with valid arguments.
    if !tc.arguments.trim().is_empty()
        && serde_json::from_str::<serde_json::Value>(&tc.arguments).is_err()
    {
        tracing::error!(
            tool = %tc.name, id = %tc.id,
            args_preview = %tc.arguments.chars().take(200).collect::<String>(),
            "Malformed tool call arguments — returning structured error"
        );
        return schema::ToolResult {
            tool_call_id: tc.id.clone(),
            name: tc.name.clone(),
            output: format!(
                "ERROR: Your tool call arguments were malformed JSON. \
                 Re-emit this tool call with valid JSON arguments. \
                 Parse error: {}",
                serde_json::from_str::<serde_json::Value>(&tc.arguments)
                    .unwrap_err()
            ),
            success: false,
            images: vec![],
        };
    }

    let args = tc.args();
    let start = std::time::Instant::now();
    tracing::info!(
        tool = %tc.name, id = %tc.id,
        args = %serde_json::to_string(&args).unwrap_or_default(),
        "Tool dispatch START"
    );

    // Browser tool: screenshot images are returned separately for multimodal context.
    if tc.name == "browser" {
        let mut images = Vec::new();
        let mut result = match tc.name.as_str() {
            _ => dispatch_desktop_tool(state, &tc.name, &args).await,
        };
        maybe_auto_screenshot_to_images(state, &args, &mut result, &mut images).await;
        return format_tool_result_with_images(&tc.id, &tc.name, result, start.elapsed(), images);
    }

    // Image generation: feed the generated image back to the vision model.
    if tc.name == "generate_image" {
        let result = dispatch_desktop_tool(state, &tc.name, &args).await;
        let images = if state.model_spec.supports_vision {
            extract_generated_image_data(&result)
        } else {
            Vec::new()
        };
        return format_tool_result_with_images(&tc.id, &tc.name, result, start.elapsed(), images);
    }

    let result = match tc.name.as_str() {
        "run_bash_command" => dispatch_shell(&args).await,
        "web_search" => dispatch_web_search(&args).await,
        "memory" => dispatch_memory(state, &args).await,
        "scratchpad" => dispatch_scratchpad(state, &args).await,
        "synaptic" => dispatch_synaptic(state, &args).await,
        "timeline" => dispatch_timeline(state, &args).await,
        "lessons" => dispatch_lessons(state, &args).await,
        "self_skills" => dispatch_self_skills(state, &args).await,
        "learning" => crate::tools::learning_tool::execute(&args, state).await,
        "steering" => crate::tools::steering_tool::execute(&args).await,
        "interpretability" => crate::tools::interpretability_tool::execute(&args, state).await,
        "codebase_search" => crate::tools::codebase_search::execute(&args).await,
        "file_read" => crate::tools::file_read::execute(&args, state.model_spec.context_length).await,
        "file_write" => crate::tools::file_write::execute(&args).await,
        "browse_url" | "screenshot_url" | "system_recompile" =>
            dispatch_desktop_tool(state, &tc.name, &args).await,
        "create_artifact" => crate::tools::artifact_tool::execute(&args).await,
        "codebase_edit" => dispatch_codebase_edit(state, &args).await,
        "checkpoint" => dispatch_checkpoint(state, &args).await,
        "system_logs" => crate::tools::system_logs::execute(&args, &state.config.general.data_dir),
        "verify_code" => crate::web::dispatch_planning::dispatch_verify_code(&args).await,
        "plan_and_execute" => crate::web::dispatch_planning::dispatch_plan_and_execute(state, &args).await,
        "session_recall" => crate::tools::session_recall_tool::execute(&args, state).await,
        "introspect" => crate::tools::introspect_tool::execute(&args, state).await,
        "spawn_sub_agent" => dispatch_spawn_sub_agent(state, &args).await,
        other => Ok(format!("Unknown tool: {}", other)),
    };

    let elapsed = start.elapsed();
    let result = format_tool_result(&tc.id, &tc.name, result, elapsed);
    log_agent_activity(&state.config.general.data_dir, &tc.name, result.success, elapsed);
    result
}

fn format_tool_result(
    id: &str,
    name: &str,
    result: anyhow::Result<String>,
    elapsed: std::time::Duration,
) -> schema::ToolResult {
    format_tool_result_with_images(id, name, result, elapsed, Vec::new())
}

/// Append a tool execution event to `data/agent_activity.jsonl`.
/// Uses JSONL append-only format — no read/parse/serialize cycle on the hot path.
/// Wires the `introspect(action='agent_activity')` tool to real data.
fn log_agent_activity(
    data_dir: &std::path::Path, tool_name: &str, success: bool, elapsed: std::time::Duration,
) {
    let path = data_dir.join("agent_activity.jsonl");
    if let Ok(line) = serde_json::to_string(&serde_json::json!({
        "tool": tool_name,
        "success": success,
        "elapsed_ms": elapsed.as_millis() as u64,
        "ts": chrono::Utc::now().to_rfc3339(),
    })) {
        let mut full = line;
        full.push('\n');
        let _ = std::fs::OpenOptions::new()
            .create(true).append(true).open(&path)
            .and_then(|mut f| { use std::io::Write; f.write_all(full.as_bytes()) });
    }
}

/// Format a tool execution result into a ToolResult with optional images.
fn format_tool_result_with_images(
    id: &str,
    name: &str,
    result: anyhow::Result<String>,
    elapsed: std::time::Duration,
    images: Vec<String>,
) -> schema::ToolResult {
    match result {
        Ok(output) => {
            tracing::info!(
                tool = %name, id = %id,
                elapsed_ms = elapsed.as_millis() as u64,
                output_len = output.len(),
                images = images.len(),
                "Tool dispatch OK"
            );
            schema::ToolResult {
                tool_call_id: id.to_string(), name: name.to_string(), output, success: true, images,
            }
        }
        Err(e) => {
            tracing::error!(
                tool = %name, id = %id,
                elapsed_ms = elapsed.as_millis() as u64,
                error = %e,
                "Tool dispatch FAILED"
            );
            let error_context = format!(
                "[TOOL FAILURE: {}]\n\
                 Error: {}\n\
                 Elapsed: {}ms\n\
                 Action: This tool call failed. Do NOT retry the same tool with the same arguments. \
                 Either try a different approach, use a different tool, or respond to the user \
                 explaining that this capability is currently unavailable.",
                name, e, elapsed.as_millis()
            );
            schema::ToolResult {
                tool_call_id: id.to_string(), name: name.to_string(),
                output: error_context, success: false, images: Vec::new(),
            }
        }
    }
}

async fn dispatch_shell(args: &serde_json::Value) -> anyhow::Result<String> {
    let cmd = args["command"].as_str().unwrap_or("");
    let wd = args["working_dir"].as_str();

    // Containment gate — block destructive/dangerous commands
    if let Some(reason) = crate::tools::containment::check_command(cmd) {
        tracing::warn!(command = %cmd, reason = %reason, "Shell: BLOCKED by containment");
        anyhow::bail!("{}", reason);
    }

    // Auto-exclude heavy binary directories from recursive file-scanning commands.
    // Prevents runaway greps through multi-GB model/build artifacts.
    let cmd = inject_exclusions(cmd);

    tracing::debug!(command = %cmd, working_dir = ?wd, "Shell: executing");
    let result = crate::tools::shell::run_command(&cmd, wd).await;
    tracing::debug!(success = result.is_ok(), "Shell: finished");
    result
}

/// Inject `--exclude-dir` flags for heavy directories into recursive grep/rg/find commands.
fn inject_exclusions(cmd: &str) -> String {
    const HEAVY_DIRS: &[&str] = &["models", "target", ".git", "node_modules"];
    let is_recursive_grep = (cmd.contains("grep") && (cmd.contains("-r") || cmd.contains("-R")))
        || cmd.starts_with("rg ");

    if is_recursive_grep {
        let mut patched = cmd.to_string();
        for dir in HEAVY_DIRS {
            let flag = format!("--exclude-dir={}", dir);
            if !patched.contains(&flag) {
                // Insert exclusions right after the grep/rg command token
                if let Some(pos) = patched.find("grep") {
                    let after = patched[pos..].find(' ').map(|i| pos + i).unwrap_or(patched.len());
                    patched.insert_str(after, &format!(" {}", flag));
                } else if patched.starts_with("rg ") {
                    patched = format!("rg {} {}", flag, &patched[3..]);
                }
            }
        }
        if patched != cmd {
            tracing::info!(original = %cmd, patched = %patched, "Shell: auto-injected exclusions");
        }
        return patched;
    }

    cmd.to_string()
}

async fn dispatch_web_search(args: &serde_json::Value) -> anyhow::Result<String> {
    let action = args["action"].as_str().unwrap_or("search");
    match action {
        "visit" => {
            let url = args["url"].as_str().unwrap_or("");
            tracing::debug!(url = %url, "WebSearch: visiting URL");
            crate::tools::web_search::visit(url).await
        }
        _ => {
            let query = args["query"].as_str().unwrap_or("");
            tracing::debug!(query = %query, "WebSearch: 8-tier waterfall search");
            crate::tools::web_search::search(query).await
        }
    }
}

use super::dispatch_memory::{
    dispatch_memory, dispatch_scratchpad, dispatch_synaptic,
    dispatch_timeline, dispatch_lessons, dispatch_self_skills,
};

async fn dispatch_codebase_edit(state: &AppState, args: &serde_json::Value) -> anyhow::Result<String> {
    let action = args["action"].as_str().unwrap_or("");
    let path = args["path"].as_str().unwrap_or("");
    let data_dir = &state.config.general.data_dir;

    match action {
        "patch" => {
            let find = args["find"].as_str().unwrap_or("");
            let replace = args["replace"].as_str().unwrap_or("");
            crate::tools::codebase_edit::patch_file(data_dir, path, find, replace)
        }
        "insert" => {
            let anchor = args["anchor"].as_str().unwrap_or("");
            let content = args["content"].as_str().unwrap_or("");
            let position = args["position"].as_str().unwrap_or("after");
            crate::tools::codebase_edit::insert_content(data_dir, path, anchor, content, position)
        }
        "multi_patch" => {
            let patches = args["patches"].as_array()
                .map(|a| a.to_vec())
                .unwrap_or_default();
            crate::tools::codebase_edit::multi_patch(data_dir, path, &patches)
        }
        "delete" => crate::tools::codebase_edit::delete_file(data_dir, path),
        other => Ok(format!("Unknown codebase_edit action: {}", other)),
    }
}

/// Execute a sub-agent spawn through the unified dispatch.
/// Parses config from args and delegates to `sub_agent::run_sub_agent`.
async fn dispatch_spawn_sub_agent(state: &AppState, args: &serde_json::Value) -> anyhow::Result<String> {
    let task = args["task"].as_str().unwrap_or("").to_string();
    let tools: Vec<String> = args["tools"].as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let max_turns = args["max_turns"].as_u64().unwrap_or(5) as usize;

    if task.is_empty() || tools.is_empty() {
        anyhow::bail!("spawn_sub_agent requires non-empty 'task' and 'tools' array");
    }

    let config = crate::inference::sub_agent::SubAgentConfig {
        task: task.clone(), allowed_tools: tools, max_turns,
    };
    let result = crate::inference::sub_agent::run_sub_agent(
        state.provider.as_ref(), config, state,
    ).await?;

    Ok(format!(
        "[Sub-Agent Result]\nSuccess: {}\nTurns: {}\nTools: {}\n\n{}",
        result.success, result.turns_used,
        result.tool_calls_made.join(", "), result.summary,
    ))
}

async fn dispatch_system_recompile(state: &AppState) -> anyhow::Result<String> {
    // Read the most recently active session to thread into resume state.
    let session_id = {
        let sessions = state.sessions.read().await;
        sessions.list().first().map(|s| s.id.clone()).unwrap_or_default()
    };
    // Infer platform from session_id format: "discord_uid_chid" or "telegram_uid_chid" → platform name.
    // Web sessions use UUIDs so they won't match.
    let platform = if session_id.starts_with("discord_") {
        "discord"
    } else if session_id.starts_with("telegram_") {
        "telegram"
    } else {
        "web"
    };
    crate::tools::compiler::run_recompile(&session_id, platform)
        .await
        .map_err(|e| anyhow::anyhow!("{}", e))
}

async fn dispatch_checkpoint(state: &AppState, args: &serde_json::Value) -> anyhow::Result<String> {
    let action = args["action"].as_str().unwrap_or("");
    let data_dir = &state.config.general.data_dir;
    let mgr = crate::tools::checkpoint::CheckpointManager::new(data_dir);

    match action {
        "list" => {
            let entries = mgr.list();
            if entries.is_empty() {
                Ok("No checkpoints found.".to_string())
            } else {
                let lines: Vec<String> = entries.iter().map(|e| {
                    format!("{} | {} | {} bytes | {}", e.id, e.original_path, e.size_bytes, e.created_at)
                }).collect();
                Ok(format!("{} checkpoint(s):\n{}", entries.len(), lines.join("\n")))
            }
        }
        "rollback" => {
            let id = args["id"].as_str().unwrap_or("");
            if id.is_empty() { anyhow::bail!("Missing checkpoint id"); }
            mgr.rollback(id)
        }
        "prune" => {
            let hours = args["max_age_hours"].as_i64().unwrap_or(48);
            let pruned = mgr.prune(hours);
            Ok(format!("Pruned {} checkpoint(s) older than {}h", pruned, hours))
        }
        other => Ok(format!("Unknown checkpoint action: {}", other)),
    }
}

/// Dispatch tools that require the desktop feature (browser, image gen, recompile).
async fn dispatch_desktop_tool(
    state: &AppState,
    name: &str,
    args: &serde_json::Value,
) -> anyhow::Result<String> {
    match name {
        "browse_url" => {
            let url = args["url"].as_str().unwrap_or("");
            crate::tools::browser_tool::browse_url(&state.browser, url).await
        }
        "screenshot_url" => {
            let url = args["url"].as_str().unwrap_or("");
            crate::tools::browser_tool::screenshot_url(&state.browser, url).await
        }
        "browser" => {
            crate::tools::browser_tool::execute_action(&state.browser, args).await
        }
        "generate_image" => {
            let port = state.config.general.flux_port.unwrap_or(8890);
            crate::tools::image_gen_tool::execute(args, port).await
        }
        "system_recompile" => dispatch_system_recompile(state).await,
        _ => Ok(format!("Unknown tool: {}", name)),
    }
}

/// Save a viewport screenshot to disk and push its data URI into images vec
/// for proper multimodal tool results (image_url content blocks, not inline base64 text).
async fn maybe_auto_screenshot_to_images(
    state: &AppState,
    args: &serde_json::Value,
    result: &mut anyhow::Result<String>,
    images: &mut Vec<String>,
) {
    let action = args["action"].as_str().unwrap_or("");
    if action != "open" && action != "navigate" { return; }
    if !state.model_spec.supports_vision { return; }

    let page_id = args["page_id"].as_str().unwrap_or("page_0");
    if result.is_ok() {
        let s = state.browser.read().await;
        if let Some(page) = s.get_page_or_latest(page_id) {
            let screenshot_fut = page.screenshot(
                chromiumoxide::page::ScreenshotParams::builder().build(),
            );
            match tokio::time::timeout(std::time::Duration::from_secs(3), screenshot_fut).await {
                Ok(Ok(screenshot)) => {
                    // Save to disk for API access
                    let id = uuid::Uuid::new_v4().to_string();
                    let filename = format!("{}.png", id);
                    let dir = std::path::PathBuf::from("data/images");
                    std::fs::create_dir_all(&dir).ok();
                    std::fs::write(dir.join(&filename), &screenshot).ok();

                    // Push data URI for multimodal context (proper image_url block)
                    use base64::Engine;
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&screenshot);
                    images.push(format!("data:image/png;base64,{}", b64));

                    // Append only the file reference to text (not 277KB of base64)
                    if let Ok(ref mut text) = result {
                        text.push_str(&format!("\n\n![viewport screenshot](/api/images/{})", filename));
                    }
                    tracing::info!(action, page_id, "Auto-screenshot saved ({} bytes)", screenshot.len());
                }
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, "Auto-screenshot failed — continuing without");
                }
                Err(_) => {
                    tracing::warn!(action, page_id, "Auto-screenshot timed out (3s) — continuing without");
                }
            }
        }
    }
}

/// Read the generated image from disk and return its base64 data URI.
/// Extracts the filename from the tool result text (looks for /api/images/xxx.png).
fn extract_generated_image_data(result: &anyhow::Result<String>) -> Vec<String> {
    let text = match result {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let re = regex::Regex::new(r"/api/images/([^)\s]+)").unwrap();
    let Some(cap) = re.captures(text) else { return Vec::new() };
    let filename = &cap[1];
    let path = format!("data/images/{}", filename);
    match std::fs::read(&path) {
        Ok(bytes) => {
            use base64::Engine;
            let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
            vec![format!("data:image/png;base64,{}", b64)]
        }
        Err(e) => {
            tracing::warn!(path = %path, error = %e, "Failed to read generated image for vision feedback");
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dispatch_unknown_tool() {
        let tc = schema::ToolCall {
            id: "1".into(), name: "nonexistent".into(), arguments: "{}".into(),
        };
        let rt = tokio::runtime::Runtime::new().expect("Failed to create runtime");
        let tmp = tempfile::TempDir::new().unwrap();
        let state = crate::web::state::AppState {
            config: std::sync::Arc::new(crate::config::AppConfig::default()),
            model_spec: std::sync::Arc::new(crate::model::ModelSpec::default()),
            memory: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::memory::MemoryManager::new(tmp.path()).unwrap()
            )),
            sessions: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::session::SessionManager::new(&tmp.path().join("s")).unwrap()
            )),
            provider: std::sync::Arc::new(crate::provider::llamacpp::LlamaCppProvider::new(
                &crate::config::LlamaCppConfig::default()
            )),
            golden_buffer: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::learning::buffers::GoldenBuffer::new(500)
            )),
            rejection_buffer: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::learning::buffers_rejection::RejectionBuffer::new()
            )),
            scheduler: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::scheduler::store::JobStore::load(tmp.path()).unwrap()
            )),
            agents: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::agents::AgentRegistry::new(tmp.path()).unwrap()
            )),
            teams: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::agents::teams::TeamRegistry::new(tmp.path()).unwrap()
            )),
            browser: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::tools::browser_tool::BrowserState::new()
            )),
            platforms: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::platform::registry::PlatformRegistry::new()
            )),
            mutable_config: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::config::AppConfig::default()
            )),
            resume_message: std::sync::Arc::new(tokio::sync::RwLock::new(None)),
            sae: std::sync::Arc::new(tokio::sync::RwLock::new(None)),
            live_monitor: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::interpretability::live::LiveMonitor::new(50),
            )),
            snapshot_store: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::interpretability::snapshot::SnapshotStore::new(
                    &tmp.path().join("snapshots"),
                ).unwrap(),
            )),
            cancel_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            curriculum: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::learning::curriculum::CurriculumStore::open(
                    &tmp.path().join("curriculum"),
                ).unwrap(),
            )),
            quarantine: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::learning::verification::QuarantineBuffer::new(),
            )),
            review_deck: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::learning::review::ReviewDeck::new(),
            )),
        };
        let result = rt.block_on(execute_tool_with_state(&state, &tc));
        assert!(result.output.contains("Unknown tool"));
    }
}
