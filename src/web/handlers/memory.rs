//! Memory tier API handlers.

use crate::web::state::AppState;
use axum::{extract::State, response::IntoResponse, Json};

pub async fn timeline(State(state): State<AppState>) -> impl IntoResponse {
    let memory = state.memory.read().await;
    let entries: Vec<serde_json::Value> = memory.timeline.recent(50).iter().map(|e| {
        serde_json::json!({ "session_id": e.session_id, "transcript": e.transcript, "timestamp": e.timestamp })
    }).collect();
    Json(serde_json::json!({ "entries": entries, "total": memory.timeline.entry_count() }))
}

pub async fn lessons(State(state): State<AppState>) -> impl IntoResponse {
    let memory = state.memory.read().await;
    let lessons: Vec<serde_json::Value> = memory.lessons.all().iter().map(|l| {
        serde_json::json!({
            "id": l.id, "rule": l.rule, "source": l.source,
            "confidence": l.confidence, "times_applied": l.times_applied,
        })
    }).collect();
    Json(serde_json::json!({ "lessons": lessons, "total": memory.lessons.count() }))
}

pub async fn procedures(State(state): State<AppState>) -> impl IntoResponse {
    let memory = state.memory.read().await;
    let procs: Vec<serde_json::Value> = memory.procedures.all().iter().map(|p| {
        serde_json::json!({
            "id": p.id, "name": p.name, "description": p.description,
            "steps": p.steps.len(), "success_count": p.success_count, "last_used": p.last_used,
        })
    }).collect();
    Json(serde_json::json!({ "procedures": procs, "total": memory.procedures.count() }))
}

pub async fn scratchpad(State(state): State<AppState>) -> impl IntoResponse {
    let memory = state.memory.read().await;
    let entries: Vec<serde_json::Value> = memory.scratchpad.all().iter().map(|s| {
        serde_json::json!({ "key": s.key, "value": s.value, "pinned": s.pinned })
    }).collect();
    Json(serde_json::json!({ "entries": entries, "total": memory.scratchpad.count() }))
}

pub async fn synaptic(State(state): State<AppState>) -> impl IntoResponse {
    let memory = state.memory.read().await;
    let nodes: Vec<serde_json::Value> = memory.synaptic.recent_nodes(50).iter().map(|n| {
        serde_json::json!({ "id": n.id, "layer": n.layer, "data": n.data })
    }).collect();
    Json(serde_json::json!({
        "nodes": nodes, "node_count": memory.synaptic.node_count(),
        "edge_count": memory.synaptic.edge_count(), "layers": memory.synaptic.layers(),
    }))
}

pub async fn stats(State(state): State<AppState>) -> impl IntoResponse {
    let memory = state.memory.read().await;
    Json(serde_json::json!({
        "consolidations": memory.consolidation.consolidation_count(),
        "timeline": memory.timeline.entry_count(),
        "lessons": memory.lessons.count(),
        "procedures": memory.procedures.count(),
        "scratchpad": memory.scratchpad.count(),
        "embeddings": memory.embeddings.count(),
        "synaptic_nodes": memory.synaptic.node_count(),
        "synaptic_edges": memory.synaptic.edge_count(),
    }))
}
