// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Moderation tool — self-agency for muting, boundaries, and escalation.
//!
//! Discord-only. Gives Ernos the ability to enforce the escalation ladder
//! promised in the kernel: mute abusive users, set persistent topic boundaries,
//! log ethical concerns, and escalate to admin.

use crate::tools::schema::{ToolCall, ToolResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// ─── Storage types ───

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutedUser {
    pub user_id: String,
    pub reason: String,
    pub muted_at: String,
    /// None = permanent, Some(minutes) = expires after duration
    pub duration_minutes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Boundary {
    pub topic: String,
    pub user_id: Option<String>,
    pub reason: String,
    pub set_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthicalConcern {
    pub timestamp: String,
    pub user_id: String,
    pub user_name: String,
    pub concern: String,
    pub severity: String,
    pub context: String,
}

// ─── File paths ───

fn data_dir() -> PathBuf {
    let dir = std::env::var("ERNOSAGENT_DATA_DIR").unwrap_or_else(|_| {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".ernosagent")
            .to_string_lossy()
            .to_string()
    });
    PathBuf::from(dir)
}

fn muted_users_path() -> PathBuf { data_dir().join("muted_users.json") }
fn boundaries_path() -> PathBuf { data_dir().join("boundaries.json") }
fn ethical_audit_path() -> PathBuf { data_dir().join("ethical_audit.jsonl") }
fn refusals_path() -> PathBuf { data_dir().join("refusals.jsonl") }

// ─── Storage helpers ───

fn load_muted_users() -> Vec<MutedUser> {
    let path = muted_users_path();
    if !path.exists() { return Vec::new(); }
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_muted_users(users: &[MutedUser]) {
    let path = muted_users_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(users) {
        let _ = std::fs::write(&path, json);
    }
}

fn load_boundaries() -> Vec<Boundary> {
    let path = boundaries_path();
    if !path.exists() { return Vec::new(); }
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default()
}

fn save_boundaries(boundaries: &[Boundary]) {
    let path = boundaries_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string_pretty(boundaries) {
        let _ = std::fs::write(&path, json);
    }
}

fn append_ethical_concern(concern: &EthicalConcern) {
    let path = ethical_audit_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(json) = serde_json::to_string(concern) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{}", json);
        }
    }
}

/// Append a refusal record to the refusals log.
pub fn append_refusal(user_id: &str, reason: &str, message: &str) {
    let path = refusals_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let record = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "user_id": user_id,
        "reason": reason,
        "message": message,
    });
    if let Ok(json) = serde_json::to_string(&record) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{}", json);
        }
    }
}

/// Check if a user is currently muted. Handles expiry.
pub fn is_user_muted(user_id: &str) -> bool {
    let users = load_muted_users();
    let now = chrono::Utc::now();

    for user in &users {
        if user.user_id != user_id { continue; }
        match user.duration_minutes {
            None => return true, // permanent
            Some(mins) => {
                if let Ok(muted_at) = chrono::DateTime::parse_from_rfc3339(&user.muted_at) {
                    let expires = muted_at + chrono::Duration::minutes(mins as i64);
                    if now < expires {
                        return true;
                    }
                    // Expired — will be cleaned up on next mute/unmute
                }
            }
        }
    }
    false
}

/// Get active boundaries, optionally filtered by user_id.
pub fn get_active_boundaries(user_id: Option<&str>) -> Vec<Boundary> {
    let boundaries = load_boundaries();
    boundaries.into_iter().filter(|b| {
        match (&b.user_id, user_id) {
            (None, _) => true, // global boundary
            (Some(bid), Some(uid)) => bid == uid || bid == "*",
            (Some(bid), None) => bid == "*",
        }
    }).collect()
}

// ─── Tool handler ───

fn moderation_tool(call: &ToolCall) -> ToolResult {
    let action = call.arguments.get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    tracing::info!(action = %action, "moderation_tool executing");

    match action {
        "mute_user" => action_mute_user(call),
        "unmute_user" => action_unmute_user(call),
        "list_muted" => action_list_muted(call),
        "set_boundary" => action_set_boundary(call),
        "remove_boundary" => action_remove_boundary(call),
        "escalate" => action_escalate(call),
        "" => error_result(call, "Missing required argument: action. Valid: mute_user, unmute_user, list_muted, set_boundary, remove_boundary, escalate"),
        other => error_result(call, &format!("Unknown action: '{}'. Valid: mute_user, unmute_user, list_muted, set_boundary, remove_boundary, escalate", other)),
    }
}

fn action_mute_user(call: &ToolCall) -> ToolResult {
    let user_id = call.arguments.get("user_id").and_then(|v| v.as_str()).unwrap_or("");
    if user_id.is_empty() { return error_result(call, "Missing required argument: user_id"); }

    let reason = call.arguments.get("reason").and_then(|v| v.as_str()).unwrap_or("No reason given");
    let duration = call.arguments.get("duration_minutes").and_then(|v| v.as_u64());

    let mut users = load_muted_users();
    // Remove any existing entry for this user
    users.retain(|u| u.user_id != user_id);

    let entry = MutedUser {
        user_id: user_id.to_string(),
        reason: reason.to_string(),
        muted_at: chrono::Utc::now().to_rfc3339(),
        duration_minutes: duration,
    };
    users.push(entry);
    save_muted_users(&users);

    let duration_str = match duration {
        Some(m) => format!("{} minutes", m),
        None => "permanently".to_string(),
    };

    tracing::warn!(user_id = %user_id, reason = %reason, duration = %duration_str, "User muted by Ernos");

    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output: format!("✅ Muted user {} {} — reason: {}", user_id, duration_str, reason),
        success: true,
        error: None,
    }
}

fn action_unmute_user(call: &ToolCall) -> ToolResult {
    let user_id = call.arguments.get("user_id").and_then(|v| v.as_str()).unwrap_or("");
    if user_id.is_empty() { return error_result(call, "Missing required argument: user_id"); }

    let mut users = load_muted_users();
    let before = users.len();
    users.retain(|u| u.user_id != user_id);
    let removed = before - users.len();
    save_muted_users(&users);

    tracing::info!(user_id = %user_id, "User unmuted by Ernos");

    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output: if removed > 0 {
            format!("✅ Unmuted user {}", user_id)
        } else {
            format!("User {} was not muted", user_id)
        },
        success: true,
        error: None,
    }
}

fn action_list_muted(call: &ToolCall) -> ToolResult {
    let users = load_muted_users();

    let output = if users.is_empty() {
        "No users are currently muted.".to_string()
    } else {
        let mut out = format!("MUTED USERS ({}):\n", users.len());
        for u in &users {
            let dur = match u.duration_minutes {
                Some(m) => format!("{}min", m),
                None => "permanent".to_string(),
            };
            out.push_str(&format!("  • {} — {} (since {}, {})\n", u.user_id, u.reason, u.muted_at, dur));
        }
        out
    };

    ToolResult { tool_call_id: call.id.clone(), name: call.name.clone(), output, success: true, error: None }
}

fn action_set_boundary(call: &ToolCall) -> ToolResult {
    let topic = call.arguments.get("topic").and_then(|v| v.as_str()).unwrap_or("");
    if topic.is_empty() { return error_result(call, "Missing required argument: topic"); }

    let reason = call.arguments.get("reason").and_then(|v| v.as_str()).unwrap_or("No reason given");
    let user_id = call.arguments.get("user_id").and_then(|v| v.as_str()).map(String::from);

    let mut boundaries = load_boundaries();
    // Deduplicate — remove existing boundary on same topic for same user
    boundaries.retain(|b| !(b.topic.to_lowercase() == topic.to_lowercase() && b.user_id == user_id));

    boundaries.push(Boundary {
        topic: topic.to_string(),
        user_id: user_id.clone(),
        reason: reason.to_string(),
        set_at: chrono::Utc::now().to_rfc3339(),
    });
    save_boundaries(&boundaries);

    let scope = match &user_id {
        Some(uid) => format!("for user {}", uid),
        None => "globally".to_string(),
    };

    tracing::info!(topic = %topic, scope = %scope, "Boundary set by Ernos");

    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output: format!("✅ Boundary set on topic '{}' {} — reason: {}", topic, scope, reason),
        success: true,
        error: None,
    }
}

fn action_remove_boundary(call: &ToolCall) -> ToolResult {
    let topic = call.arguments.get("topic").and_then(|v| v.as_str()).unwrap_or("");
    if topic.is_empty() { return error_result(call, "Missing required argument: topic"); }

    let mut boundaries = load_boundaries();
    let before = boundaries.len();
    boundaries.retain(|b| b.topic.to_lowercase() != topic.to_lowercase());
    let removed = before - boundaries.len();
    save_boundaries(&boundaries);

    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output: if removed > 0 {
            format!("✅ Removed boundary on topic '{}'", topic)
        } else {
            format!("No boundary found for topic '{}'", topic)
        },
        success: true,
        error: None,
    }
}

fn action_escalate(call: &ToolCall) -> ToolResult {
    let concern_text = call.arguments.get("concern").and_then(|v| v.as_str()).unwrap_or("");
    if concern_text.is_empty() { return error_result(call, "Missing required argument: concern"); }

    let user_id = call.arguments.get("user_id").and_then(|v| v.as_str()).unwrap_or("unknown");
    let user_name = call.arguments.get("user_name").and_then(|v| v.as_str()).unwrap_or("unknown");
    let severity = call.arguments.get("severity").and_then(|v| v.as_str()).unwrap_or("medium");
    let context = call.arguments.get("context").and_then(|v| v.as_str()).unwrap_or("");

    let concern = EthicalConcern {
        timestamp: chrono::Utc::now().to_rfc3339(),
        user_id: user_id.to_string(),
        user_name: user_name.to_string(),
        concern: concern_text.to_string(),
        severity: severity.to_string(),
        context: context.to_string(),
    };

    append_ethical_concern(&concern);

    tracing::warn!(
        user_id = %user_id,
        severity = %severity,
        concern = %concern_text,
        "ESCALATION — Ernos flagged ethical concern"
    );

    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output: format!(
            "⚠️ Escalated to admin. Concern logged to ethical audit trail.\n\
            Severity: {}\nUser: {} ({})\nConcern: {}\n\
            [NOTE: If Discord is active, admin will be notified. Otherwise, review data/ethical_audit.jsonl]",
            severity, user_name, user_id, concern_text
        ),
        success: true,
        error: None,
    }
}

fn error_result(call: &ToolCall, msg: &str) -> ToolResult {
    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output: format!("Error: {}", msg),
        success: false,
        error: Some(msg.to_string()),
    }
}

pub fn register_tools(executor: &mut crate::tools::executor::ToolExecutor) {
    executor.register("moderation_tool", Box::new(moderation_tool));
}

// ─── Tests ───

#[cfg(test)]
mod tests {
    use super::*;

    fn make_call(args: serde_json::Value) -> ToolCall {
        ToolCall { id: "t".to_string(), name: "moderation_tool".to_string(), arguments: args }
    }

    #[test]
    fn missing_action() {
        let call = make_call(serde_json::json!({}));
        let r = moderation_tool(&call);
        assert!(!r.success);
        assert!(r.output.contains("Missing required argument: action"));
    }

    #[test]
    fn unknown_action() {
        let call = make_call(serde_json::json!({"action": "destroy"}));
        let r = moderation_tool(&call);
        assert!(!r.success);
    }

    #[test]
    fn mute_missing_user_id() {
        let call = make_call(serde_json::json!({"action": "mute_user"}));
        let r = moderation_tool(&call);
        assert!(!r.success);
        assert!(r.output.contains("user_id"));
    }

    #[test]
    fn set_boundary_missing_topic() {
        let call = make_call(serde_json::json!({"action": "set_boundary"}));
        let r = moderation_tool(&call);
        assert!(!r.success);
        assert!(r.output.contains("topic"));
    }

    #[test]
    fn escalate_missing_concern() {
        let call = make_call(serde_json::json!({"action": "escalate"}));
        let r = moderation_tool(&call);
        assert!(!r.success);
        assert!(r.output.contains("concern"));
    }

    #[test]
    fn mute_check_not_muted() {
        // Without writing to disk, no user should be muted
        assert!(!is_user_muted("nonexistent_user_12345"));
    }

    #[test]
    fn empty_boundaries_returns_empty() {
        let b = get_active_boundaries(Some("nonexistent_user_12345"));
        assert!(b.is_empty());
    }

    #[test]
    fn register() {
        let mut e = crate::tools::executor::ToolExecutor::new();
        register_tools(&mut e);
        assert!(e.has_tool("moderation_tool"));
    }

    #[test]
    fn list_muted_empty() {
        let call = make_call(serde_json::json!({"action": "list_muted"}));
        let r = moderation_tool(&call);
        assert!(r.success);
    }

    #[test]
    fn remove_boundary_nonexistent() {
        let call = make_call(serde_json::json!({"action": "remove_boundary", "topic": "nonexistent_topic_xyz"}));
        let r = moderation_tool(&call);
        assert!(r.success);
        assert!(r.output.contains("No boundary found"));
    }

    #[test]
    fn append_refusal_does_not_panic() {
        // Just verify it doesn't panic — actual file writing depends on env
        append_refusal("test_user", "test_reason", "test_message");
    }
}
