// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Memory, timeline, lessons, and scratchpad API routes.

use crate::tools::schema::ToolCall;
use crate::web::state::SharedState;
use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
pub struct SearchQuery {
    q: Option<String>,
    limit: Option<u64>,
}

// ── Memory Search ────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct MemorySearchResult {
    query: String,
    results: String,
}

pub async fn search_memory(
    State(state): State<SharedState>,
    Query(params): Query<SearchQuery>,
) -> Json<MemorySearchResult> {
    let query = params.q.unwrap_or_default();
    let call = make_tool_call(
        "memory_tool",
        serde_json::json!({
            "action": "recall",
            "query": query,
            "limit": params.limit.unwrap_or(10)
        }),
    );
    let s = state.read().await;
    let result = s.executor.execute(&call);
    Json(MemorySearchResult {
        query,
        results: result.output,
    })
}

// ── Timeline ─────────────────────────────────────────────────────────

pub async fn timeline_recent(
    State(state): State<SharedState>,
    Query(params): Query<SearchQuery>,
) -> Json<serde_json::Value> {
    let call = make_tool_call(
        "timeline_tool",
        serde_json::json!({
            "action": "recent",
            "limit": params.limit.unwrap_or(50)
        }),
    );
    let s = state.read().await;
    let result = s.executor.execute(&call);
    Json(serde_json::json!({ "output": result.output, "success": result.success }))
}

pub async fn timeline_search(
    State(state): State<SharedState>,
    Query(params): Query<SearchQuery>,
) -> Json<serde_json::Value> {
    let call = make_tool_call(
        "timeline_tool",
        serde_json::json!({
            "action": "search",
            "query": params.q.unwrap_or_default()
        }),
    );
    let s = state.read().await;
    let result = s.executor.execute(&call);
    Json(serde_json::json!({ "output": result.output, "success": result.success }))
}

// ── Lessons ──────────────────────────────────────────────────────────

pub async fn list_lessons(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let call = make_tool_call(
        "lessons_tool",
        serde_json::json!({
            "action": "list", "limit": 100
        }),
    );
    let s = state.read().await;
    let result = s.executor.execute(&call);
    Json(serde_json::json!({ "output": result.output, "success": result.success }))
}

pub async fn reinforce_lesson(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let call = make_tool_call(
        "lessons_tool",
        serde_json::json!({
            "action": "reinforce", "id": id
        }),
    );
    let s = state.read().await;
    let result = s.executor.execute(&call);
    let status = if result.success {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    };
    (status, Json(serde_json::json!({ "output": result.output })))
}

pub async fn weaken_lesson(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> (StatusCode, Json<serde_json::Value>) {
    let call = make_tool_call(
        "lessons_tool",
        serde_json::json!({
            "action": "weaken", "id": id
        }),
    );
    let s = state.read().await;
    let result = s.executor.execute(&call);
    let status = if result.success {
        StatusCode::OK
    } else {
        StatusCode::NOT_FOUND
    };
    (status, Json(serde_json::json!({ "output": result.output })))
}

// ── Scratchpad ───────────────────────────────────────────────────────

pub async fn read_scratchpad(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let call = make_tool_call(
        "scratchpad_tool",
        serde_json::json!({
            "action": "read"
        }),
    );
    let s = state.read().await;
    let result = s.executor.execute(&call);
    Json(serde_json::json!({ "output": result.output, "success": result.success }))
}

#[derive(Deserialize)]
pub struct ScratchpadWrite {
    key: String,
    content: String,
}

pub async fn write_scratchpad(
    State(state): State<SharedState>,
    Json(body): Json<ScratchpadWrite>,
) -> Json<serde_json::Value> {
    let call = make_tool_call(
        "scratchpad_tool",
        serde_json::json!({
            "action": "write", "key": body.key, "content": body.content
        }),
    );
    let s = state.read().await;
    let result = s.executor.execute(&call);
    Json(serde_json::json!({ "output": result.output, "success": result.success }))
}

// ── Consolidation ────────────────────────────────────────────────────

pub async fn consolidate_memory(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let call = make_tool_call(
        "memory_tool",
        serde_json::json!({
            "action": "consolidate"
        }),
    );
    let s = state.read().await;
    let result = s.executor.execute(&call);
    Json(serde_json::json!({ "output": result.output, "success": result.success }))
}

// ── Helper ───────────────────────────────────────────────────────────

fn make_tool_call(name: &str, args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: format!(
            "api_{}",
            uuid::Uuid::new_v4()
                .to_string()
                .split('-')
                .next()
                .unwrap_or("x")
        ),
        name: name.to_string(),
        arguments: args,
    }
}
