//! Session CRUD handlers — list, get, create, rename, delete, pin, archive, search, fork, export.

use crate::web::state::AppState;
use axum::{extract::State, response::IntoResponse, Json};

pub async fn list_sessions(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let sessions = state.sessions.read().await;
    let archived = params.get("archived").map_or(false, |v| v == "true");
    let list: Vec<serde_json::Value> = sessions.list().iter()
        .filter(|s| s.archived == archived)
        .map(|s| {
            serde_json::json!({
                "id": s.id, "title": s.title,
                "created_at": s.created_at, "updated_at": s.updated_at,
                "message_count": s.messages.len(),
                "pinned": s.pinned, "archived": s.archived,
                "preview": s.preview(),
                "relative_time": s.relative_time(),
                "date_group": s.date_group(),
            })
        }).collect();
    Json(list)
}

pub async fn get_session(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let sessions = state.sessions.read().await;
    match sessions.get(&id) {
        Some(session) => Json(serde_json::json!({
            "id": session.id, "title": session.title,
            "messages": session.messages,
            "created_at": session.created_at, "updated_at": session.updated_at,
            "pinned": session.pinned, "archived": session.archived,
        })),
        None => Json(serde_json::json!({"error": "Session not found"})),
    }
}

pub async fn create_session(State(state): State<AppState>) -> impl IntoResponse {
    let mut sessions = state.sessions.write().await;
    match sessions.create() {
        Ok(session) => Json(serde_json::json!({ "id": session.id, "title": session.title })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

pub async fn rename_session(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.write().await;
    let title = body["title"].as_str().unwrap_or("Untitled");
    if let Some(session) = sessions.get_mut(&id) {
        session.title = title.to_string();
        let updated = session.clone();
        let _ = sessions.update(&updated);
        Json(serde_json::json!({"ok": true}))
    } else {
        Json(serde_json::json!({"error": "Session not found"}))
    }
}

pub async fn delete_session(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.write().await;
    match sessions.delete(&id) {
        Ok(_) => Json(serde_json::json!({"ok": true})),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

pub async fn toggle_pin(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.write().await;
    if let Some(session) = sessions.get_mut(&id) {
        session.pinned = !session.pinned;
        let updated = session.clone();
        let _ = sessions.update(&updated);
        Json(serde_json::json!({"ok": true, "pinned": updated.pinned}))
    } else {
        Json(serde_json::json!({"error": "Session not found"}))
    }
}

pub async fn toggle_archive(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.write().await;
    if let Some(session) = sessions.get_mut(&id) {
        session.archived = !session.archived;
        let updated = session.clone();
        let _ = sessions.update(&updated);
        Json(serde_json::json!({"ok": true, "archived": updated.archived}))
    } else {
        Json(serde_json::json!({"error": "Session not found"}))
    }
}

pub async fn search_sessions(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let sessions = state.sessions.read().await;
    let query = params.get("q").map(|s| s.as_str()).unwrap_or("");
    if query.is_empty() {
        return Json(serde_json::json!([]));
    }
    let results = sessions.search(query);
    Json(serde_json::json!(results))
}

pub async fn fork_session(
    State(state): State<AppState>,
    axum::extract::Path((id, idx)): axum::extract::Path<(String, usize)>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.write().await;
    match sessions.fork(&id, idx) {
        Ok(forked) => Json(serde_json::json!({
            "ok": true,
            "id": forked.id,
            "title": forked.title,
        })),
        Err(e) => Json(serde_json::json!({"error": e.to_string()})),
    }
}

pub async fn export_session(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let sessions = state.sessions.read().await;
    match sessions.get(&id) {
        Some(session) => {
            let mut md = format!("# {}\n\nExported: {}\n\n---\n\n", session.title, chrono::Utc::now().format("%Y-%m-%d %H:%M UTC"));
            for msg in &session.messages {
                let role = match msg.role.as_str() {
                    "user" => "**You**",
                    "assistant" => "**Ern-OS**",
                    "system" => "**System**",
                    _ => &msg.role,
                };
                md.push_str(&format!("{}\n\n{}\n\n---\n\n", role, msg.text_content()));
            }
            Json(serde_json::json!({"ok": true, "markdown": md, "title": session.title}))
        }
        None => Json(serde_json::json!({"error": "Session not found"})),
    }
}

pub async fn delete_message(
    State(state): State<AppState>,
    axum::extract::Path((id, idx)): axum::extract::Path<(String, usize)>,
) -> impl IntoResponse {
    let mut sessions = state.sessions.write().await;
    if let Some(session) = sessions.get_mut(&id) {
        if idx < session.messages.len() {
            session.messages.remove(idx);
            let updated = session.clone();
            let _ = sessions.update(&updated);
            Json(serde_json::json!({"ok": true}))
        } else {
            Json(serde_json::json!({"error": "Message index out of bounds"}))
        }
    } else {
        Json(serde_json::json!({"error": "Session not found"}))
    }
}

pub async fn react_message(
    State(state): State<AppState>,
    axum::extract::Path((id, idx)): axum::extract::Path<(String, usize)>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let reaction = body["reaction"].as_str().unwrap_or("");
    let sessions = state.sessions.read().await;
    if let Some(session) = sessions.get(&id) {
        if idx < session.messages.len() {
            let msg = &session.messages[idx];
            let user_query = session.messages.iter()
                .take(idx)
                .filter(|m| m.role == "user")
                .last()
                .map(|m| m.text_content().to_string())
                .unwrap_or_default();

            // Feed into training buffers
            let data_dir = &state.config.general.data_dir;
            let entry = serde_json::json!({
                "query": user_query,
                "response": msg.text_content(),
                "reaction": reaction,
                "session_id": id,
                "message_index": idx,
                "timestamp": chrono::Utc::now(),
            });

            let buffer_file = match reaction {
                "up" => data_dir.join("golden_buffer.json"),
                "down" => data_dir.join("rejection_buffer.json"),
                _ => return Json(serde_json::json!({"error": "Invalid reaction. Use 'up' or 'down'."})),
            };

            // Append to JSONL buffer
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&buffer_file) {
                use std::io::Write;
                let _ = writeln!(file, "{}", serde_json::to_string(&entry).unwrap_or_default());
            }

            tracing::info!(reaction, session = %id, idx, "Message reaction recorded");
            Json(serde_json::json!({"ok": true, "reaction": reaction}))
        } else {
            Json(serde_json::json!({"error": "Message index out of bounds"}))
        }
    } else {
        Json(serde_json::json!({"error": "Session not found"}))
    }
}
