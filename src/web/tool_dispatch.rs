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
    let args = tc.args();
    let start = std::time::Instant::now();
    tracing::info!(
        tool = %tc.name, id = %tc.id,
        args = %serde_json::to_string(&args).unwrap_or_default(),
        "Tool dispatch START"
    );

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
        "interpretability" => crate::tools::interpretability_tool::execute(&args).await,
        "codebase_search" => crate::tools::codebase_search::execute(&args).await,
        "file_read" => crate::tools::file_read::execute(&args).await,
        "file_write" => crate::tools::file_write::execute(&args).await,
        "browse_url" => {
            let url = args["url"].as_str().unwrap_or("");
            crate::tools::browser_tool::browse_url(&state.browser, url).await
        }
        "screenshot_url" => {
            let url = args["url"].as_str().unwrap_or("");
            crate::tools::browser_tool::screenshot_url(&state.browser, url).await
        }
        "browser" => crate::tools::browser_tool::execute_action(&state.browser, &args).await,
        "create_artifact" => crate::tools::artifact_tool::execute(&args).await,
        "generate_image" => crate::tools::image_gen_tool::execute(&args).await,
        "codebase_edit" => dispatch_codebase_edit(state, &args).await,
        "system_recompile" => dispatch_system_recompile().await,
        "checkpoint" => dispatch_checkpoint(state, &args).await,
        "system_logs" => {
            let data_dir = &state.config.general.data_dir;
            crate::tools::system_logs::execute(&args, data_dir)
        }
        other => Ok(format!("Unknown tool: {}", other)),
    };

    let elapsed = start.elapsed();

    match result {
        Ok(output) => {
            tracing::info!(
                tool = %tc.name, id = %tc.id,
                elapsed_ms = elapsed.as_millis() as u64,
                output_len = output.len(),
                "Tool dispatch OK"
            );
            schema::ToolResult {
                tool_call_id: tc.id.clone(), name: tc.name.clone(), output, success: true,
            }
        }
        Err(e) => {
            tracing::error!(
                tool = %tc.name, id = %tc.id,
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
                tc.name, e, elapsed.as_millis()
            );
            schema::ToolResult {
                tool_call_id: tc.id.clone(), name: tc.name.clone(),
                output: error_context, success: false,
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

async fn dispatch_memory(state: &AppState, args: &serde_json::Value) -> anyhow::Result<String> {
    let action = args["action"].as_str().unwrap_or("");
    let query = args["query"].as_str();
    match action {
        "recall" => {
            let memory = state.memory.read().await;
            Ok(memory.recall_context(query.unwrap_or("general"), 2000))
        }
        "status" => {
            let memory = state.memory.read().await;
            Ok(memory.status_summary())
        }
        "search" => {
            let memory = state.memory.read().await;
            Ok(memory.recall_context(query.unwrap_or(""), 1000))
        }
        "reset" => {
            let mut memory = state.memory.write().await;
            memory.clear();
            Ok("All memory tiers cleared.".to_string())
        }
        "consolidate" => {
            let mut memory = state.memory.write().await;
            let timeline_count = memory.timeline.entry_count();
            memory.consolidation.record_consolidation(
                timeline_count,
                "Manual consolidation via tool call",
                0,
            )?;
            Ok(format!("Memory consolidation recorded. Timeline entries: {}", timeline_count))
        }
        other => Ok(format!("Unknown memory action: '{}'. Valid actions: recall, status, search, reset, consolidate", other)),
    }
}

async fn dispatch_scratchpad(state: &AppState, args: &serde_json::Value) -> anyhow::Result<String> {
    let action = args["action"].as_str().unwrap_or("");
    let key = args["key"].as_str().unwrap_or("");
    let value = args["value"].as_str().unwrap_or("");
    let mut memory = state.memory.write().await;
    match action {
        "pin" => { let _ = memory.scratchpad.pin(key, value); Ok(format!("Pinned: {} = {}", key, value)) }
        "unpin" => { let _ = memory.scratchpad.unpin(key); Ok(format!("Unpinned: {}", key)) }
        "list" => {
            let all = memory.scratchpad.all();
            let entries: Vec<String> = all.iter().map(|e| format!("{}: {}", e.key, e.value)).collect();
            Ok(if entries.is_empty() { "Scratchpad is empty.".to_string() } else { entries.join("\n") })
        }
        "get" => Ok(memory.scratchpad.get(key).map(|s| s.to_string())
            .unwrap_or_else(|| format!("No entry for '{}'", key))),
        other => Ok(format!("Unknown scratchpad action: {}", other)),
    }
}

async fn dispatch_synaptic(state: &AppState, args: &serde_json::Value) -> anyhow::Result<String> {
    let action = args["action"].as_str().unwrap_or("");
    let mut memory = state.memory.write().await;
    match action {
        "store" => synaptic_store(&mut memory, args),
        "store_relationship" => synaptic_store_relationship(&mut memory, args),
        "search" => synaptic_search(&memory, args),
        "beliefs" => synaptic_beliefs(&memory, args),
        "recent" => synaptic_recent(&memory, args),
        "stats" => Ok(format!(
            "Nodes: {}, Edges: {}, Layers: {:?}",
            memory.synaptic.node_count(), memory.synaptic.edge_count(), memory.synaptic.layers()
        )),
        "layers" => Ok(format!("Layers: {:?}", memory.synaptic.layers())),
        "co_activate" => {
            let a = args["concept"].as_str().unwrap_or("");
            let b = args["target"].as_str().unwrap_or("");
            memory.synaptic.co_activate(a, b, 0.1);
            Ok(format!("Co-activated: {} <-> {}", a, b))
        }
        other => Ok(format!("Unknown synaptic action: {}", other)),
    }
}

fn synaptic_store(memory: &mut crate::memory::MemoryManager, args: &serde_json::Value) -> anyhow::Result<String> {
    let concept = args["concept"].as_str().unwrap_or("");
    let layer = args["layer"].as_str().unwrap_or("general");
    let mut data = std::collections::HashMap::new();
    if let Some(obj) = args["data"].as_object() {
        for (k, v) in obj { data.insert(k.clone(), v.as_str().unwrap_or("").to_string()); }
    }
    match memory.synaptic.upsert_node(concept, data, layer) {
        Ok(_) => Ok(format!("Stored concept '{}' in layer '{}'", concept, layer)),
        Err(e) => Ok(format!("Error: {}", e)),
    }
}

fn synaptic_store_relationship(memory: &mut crate::memory::MemoryManager, args: &serde_json::Value) -> anyhow::Result<String> {
    let source = args["concept"].as_str().unwrap_or("");
    let target = args["target"].as_str().unwrap_or("");
    let edge_type = args["edge_type"].as_str().unwrap_or("related_to");
    match memory.synaptic.add_edge(source, target, edge_type) {
        Ok(_) => Ok(format!("{} --{}-> {}", source, edge_type, target)),
        Err(e) => Ok(format!("Error: {}", e)),
    }
}

fn synaptic_search(memory: &crate::memory::MemoryManager, args: &serde_json::Value) -> anyhow::Result<String> {
    let q = args["concept"].as_str().unwrap_or("");
    let limit = args["limit"].as_u64().unwrap_or(10) as usize;
    let nodes = memory.synaptic.search_nodes(q, limit);
    let results: Vec<String> = nodes.iter()
        .map(|n| format!("{} [{}] (strength: {:.2})", n.id, n.layer, n.strength))
        .collect();
    Ok(if results.is_empty() { format!("No nodes matching '{}'", q) } else { results.join("\n") })
}

fn synaptic_beliefs(memory: &crate::memory::MemoryManager, args: &serde_json::Value) -> anyhow::Result<String> {
    let concept = args["concept"].as_str().unwrap_or("");
    match memory.synaptic.get_node(concept) {
        Some(node) => {
            let data: Vec<String> = node.data.iter().map(|(k, v)| format!("{}: {}", k, v)).collect();
            Ok(format!("{} [{}]\n{}", node.id, node.layer, data.join("\n")))
        }
        None => Ok(format!("No concept '{}'", concept)),
    }
}

fn synaptic_recent(memory: &crate::memory::MemoryManager, args: &serde_json::Value) -> anyhow::Result<String> {
    let n = args["limit"].as_u64().unwrap_or(10) as usize;
    let nodes = memory.synaptic.recent_nodes(n);
    let results: Vec<String> = nodes.iter().map(|n| format!("{} [{}]", n.id, n.layer)).collect();
    Ok(results.join("\n"))
}

async fn dispatch_timeline(state: &AppState, args: &serde_json::Value) -> anyhow::Result<String> {
    let action = args["action"].as_str().unwrap_or("");
    let memory = state.memory.read().await;
    match action {
        "recent" => {
            let n = args["limit"].as_u64().unwrap_or(10) as usize;
            let entries = memory.timeline.recent(n);
            let results: Vec<String> = entries.iter()
                .map(|e| format!("[{}] {}", e.timestamp.format("%Y-%m-%d %H:%M"), e.transcript))
                .collect();
            Ok(if results.is_empty() { "No timeline entries.".to_string() } else { results.join("\n") })
        }
        "search" => {
            let q = args["query"].as_str().unwrap_or("");
            let entries = memory.timeline.search(q, 20);
            let results: Vec<String> = entries.iter()
                .map(|e| format!("[{}] {}", e.timestamp.format("%Y-%m-%d %H:%M"), e.transcript))
                .collect();
            Ok(if results.is_empty() { format!("No matches for '{}'", q) } else { results.join("\n") })
        }
        "session" => {
            let sid = args["session_id"].as_str().unwrap_or("");
            let entries = memory.timeline.search(sid, 50);
            let results: Vec<String> = entries.iter()
                .map(|e| format!("[{}] {}", e.timestamp.format("%H:%M"), e.transcript))
                .collect();
            Ok(if results.is_empty() { format!("No entries for session '{}'", sid) } else { results.join("\n") })
        }
        other => Ok(format!("Unknown timeline action: {}", other)),
    }
}

async fn dispatch_lessons(state: &AppState, args: &serde_json::Value) -> anyhow::Result<String> {
    let action = args["action"].as_str().unwrap_or("");
    let mut memory = state.memory.write().await;
    match action {
        "add" => {
            let rule = args["rule"].as_str().unwrap_or("");
            let conf = args["confidence"].as_f64().unwrap_or(0.8) as f32;
            let _ = memory.lessons.add(rule, "agent", conf);
            Ok(format!("Learned: '{}' (confidence: {:.0}%)", rule, conf * 100.0))
        }
        "remove" => {
            let id = args["id"].as_str().unwrap_or("");
            if id.is_empty() {
                return Ok("Error: 'id' is required for remove. Use 'list' to see lesson IDs, then pass the ID.".to_string());
            }
            let _ = memory.lessons.remove(id);
            Ok(format!("Removed lesson: {}", id))
        }
        "list" => {
            let all = memory.lessons.all();
            let results: Vec<String> = all.iter()
                .map(|l| format!("[{:.0}%] {}", l.confidence * 100.0, l.rule))
                .collect();
            Ok(if results.is_empty() { "No lessons learned yet.".to_string() } else { results.join("\n") })
        }
        "search" => {
            let q = args["query"].as_str().unwrap_or("");
            let matches = memory.lessons.search(q, 20);
            let results: Vec<String> = matches.iter()
                .map(|l| format!("[{:.0}%] {}", l.confidence * 100.0, l.rule))
                .collect();
            Ok(if results.is_empty() { format!("No lessons matching '{}'", q) } else { results.join("\n") })
        }
        other => Ok(format!("Unknown lessons action: {}", other)),
    }
}

async fn dispatch_self_skills(state: &AppState, args: &serde_json::Value) -> anyhow::Result<String> {
    let mut memory = state.memory.write().await;
    crate::tools::self_skills_tool::execute(args, &mut memory.procedures).await
}

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

async fn dispatch_system_recompile() -> anyhow::Result<String> {
    crate::tools::compiler::run_recompile()
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
        };
        let result = rt.block_on(execute_tool_with_state(&state, &tc));
        assert!(result.output.contains("Unknown tool"));
    }
}
