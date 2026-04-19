//! Agent & team CRUD handlers.

use crate::web::state::AppState;
use axum::{extract::State, response::IntoResponse, Json};

pub async fn list_agents(State(state): State<AppState>) -> impl IntoResponse {
    let agents = state.agents.read().await;
    Json(serde_json::json!({ "agents": agents.list() }))
}

pub async fn create_agent(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let name = body["name"].as_str().unwrap_or("New Agent");
    let description = body["description"].as_str().unwrap_or("");
    let mut agent = crate::agents::AgentDefinition::new(name, description);

    if let Some(tools) = body["tools"].as_array() {
        agent.tools = tools.iter().filter_map(|t| t.as_str().map(|s| s.to_string())).collect();
    }
    if let Some(obs) = body["observer_enabled"].as_bool() {
        agent.observer_enabled = obs;
    }

    let mut agents = state.agents.write().await;
    match agents.create(agent) {
        Ok(created) => Json(serde_json::json!({ "ok": true, "agent": created })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

pub async fn get_agent(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let agents = state.agents.read().await;
    match agents.get(&id) {
        Some(agent) => Json(serde_json::json!({ "agent": agent })),
        None => Json(serde_json::json!({ "error": "Agent not found" })),
    }
}

pub async fn update_agent(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let mut agents = state.agents.write().await;
    let existing = match agents.get(&id) {
        Some(a) => a.clone(),
        None => return Json(serde_json::json!({ "error": "Agent not found" })),
    };

    let mut updated = existing;
    if let Some(name) = body["name"].as_str() { updated.name = name.to_string(); }
    if let Some(desc) = body["description"].as_str() { updated.description = desc.to_string(); }
    if let Some(tools) = body["tools"].as_array() {
        updated.tools = tools.iter().filter_map(|t| t.as_str().map(|s| s.to_string())).collect();
    }
    if let Some(obs) = body["observer_enabled"].as_bool() { updated.observer_enabled = obs; }
    updated.updated_at = chrono::Utc::now();

    match agents.update(updated) {
        Ok(()) => Json(serde_json::json!({ "ok": true })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

pub async fn delete_agent(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let mut agents = state.agents.write().await;
    match agents.delete(&id) {
        Ok(()) => Json(serde_json::json!({ "ok": true })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

pub async fn list_teams(State(state): State<AppState>) -> impl IntoResponse {
    let teams = state.teams.read().await;
    Json(serde_json::json!({ "teams": teams.list() }))
}

pub async fn create_team(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let name = body["name"].as_str().unwrap_or("New Team");
    let description = body["description"].as_str().unwrap_or("");
    let mode = match body["mode"].as_str() {
        Some("sequential") => crate::agents::teams::ExecutionMode::Sequential,
        _ => crate::agents::teams::ExecutionMode::Parallel,
    };
    let agent_ids: Vec<String> = body["agents"].as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
        .unwrap_or_default();

    let team = crate::agents::teams::TeamDefinition::new(name, description, mode, agent_ids);
    let mut teams = state.teams.write().await;
    match teams.create(team) {
        Ok(created) => Json(serde_json::json!({ "ok": true, "team": created })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

pub async fn delete_team(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let mut teams = state.teams.write().await;
    match teams.delete(&id) {
        Ok(()) => Json(serde_json::json!({ "ok": true })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}
