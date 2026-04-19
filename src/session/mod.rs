// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Session persistence — JSON-backed session CRUD.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::provider::Message;

/// A chat session with message history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub messages: Vec<Message>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    #[serde(default)]
    pub pinned: bool,
    #[serde(default)]
    pub archived: bool,
}

impl Session {
    pub fn new() -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            title: "New Chat".to_string(),
            messages: Vec::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            pinned: false,
            archived: false,
        }
    }

    /// Auto-title from first user message (first 50 chars).
    pub fn auto_title(&mut self) {
        if let Some(msg) = self.messages.iter().find(|m| m.role == "user") {
            let text = msg.text_content();
            self.title = text.chars().take(50).collect::<String>().trim().to_string();
            if self.title.is_empty() {
                self.title = "New Chat".to_string();
            }
        }
    }

    /// First 80 chars of the first user message as a preview snippet.
    pub fn preview(&self) -> String {
        self.messages.iter()
            .find(|m| m.role == "user")
            .map(|m| {
                let text = m.text_content();
                let preview: String = text.chars().take(80).collect();
                if text.len() > 80 { format!("{}…", preview.trim()) } else { preview.trim().to_string() }
            })
            .unwrap_or_default()
    }

    /// Human-readable relative time for the sidebar ("2m ago", "3h ago", "yesterday", "Apr 12").
    pub fn relative_time(&self) -> String {
        let now = Utc::now();
        let diff = now.signed_duration_since(self.updated_at);
        let secs = diff.num_seconds();
        if secs < 60 { return "just now".to_string(); }
        let mins = diff.num_minutes();
        if mins < 60 { return format!("{}m ago", mins); }
        let hours = diff.num_hours();
        if hours < 24 { return format!("{}h ago", hours); }
        let days = diff.num_days();
        if days == 1 { return "yesterday".to_string(); }
        if days < 7 { return format!("{}d ago", days); }
        self.updated_at.format("%b %d").to_string()
    }

    /// Date group key for sidebar grouping.
    pub fn date_group(&self) -> String {
        let now = Utc::now();
        let days = now.signed_duration_since(self.updated_at).num_days();
        if self.pinned { return "pinned".to_string(); }
        if days == 0 { return "today".to_string(); }
        if days == 1 { return "yesterday".to_string(); }
        if days < 7 { return "this_week".to_string(); }
        if days < 30 { return "this_month".to_string(); }
        "older".to_string()
    }
}

/// Manages session persistence on disk.
pub struct SessionManager {
    dir: PathBuf,
    sessions: Vec<Session>,
}

impl SessionManager {
    pub fn new(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create sessions dir: {}", dir.display()))?;

        let mut mgr = Self {
            dir: dir.to_path_buf(),
            sessions: Vec::new(),
        };
        mgr.load_all()?;

        tracing::info!(count = mgr.sessions.len(), "Sessions loaded");
        Ok(mgr)
    }

    fn load_all(&mut self) -> Result<()> {
        if !self.dir.exists() {
            return Ok(());
        }

        for entry in std::fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().map_or(false, |e| e == "json") {
                match std::fs::read_to_string(&path) {
                    Ok(content) => match serde_json::from_str::<Session>(&content) {
                        Ok(session) => self.sessions.push(session),
                        Err(e) => tracing::warn!(
                            path = %path.display(), error = %e,
                            "Skipped corrupt session file"
                        ),
                    },
                    Err(e) => tracing::warn!(
                        path = %path.display(), error = %e,
                        "Failed to read session file"
                    ),
                }
            }
        }

        self.sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(())
    }

    fn persist(&self, session: &Session) -> Result<()> {
        let path = self.dir.join(format!("{}.json", session.id));
        let content = serde_json::to_string_pretty(session)
            .context("Failed to serialize session")?;
        std::fs::write(&path, content)
            .with_context(|| format!("Failed to write session: {}", path.display()))?;
        Ok(())
    }

    pub fn create(&mut self) -> Result<Session> {
        let session = Session::new();
        self.persist(&session)?;
        self.sessions.insert(0, session.clone());
        Ok(session)
    }

    pub fn get(&self, id: &str) -> Option<&Session> {
        self.sessions.iter().find(|s| s.id == id)
    }

    pub fn get_mut(&mut self, id: &str) -> Option<&mut Session> {
        self.sessions.iter_mut().find(|s| s.id == id)
    }

    pub fn update(&mut self, session: &Session) -> Result<()> {
        self.persist(session)?;
        if let Some(s) = self.sessions.iter_mut().find(|s| s.id == session.id) {
            *s = session.clone();
        }
        Ok(())
    }

    pub fn delete(&mut self, id: &str) -> Result<()> {
        let path = self.dir.join(format!("{}.json", id));
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        self.sessions.retain(|s| s.id != id);
        Ok(())
    }

    pub fn list(&self) -> &[Session] {
        &self.sessions
    }

    /// Search sessions by query string (matches title and message content, case-insensitive).
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        let q = query.to_lowercase();
        let mut results = Vec::new();
        for session in &self.sessions {
            if session.archived { continue; }
            // Check title
            if session.title.to_lowercase().contains(&q) {
                results.push(SearchResult {
                    session_id: session.id.clone(),
                    title: session.title.clone(),
                    message_index: None,
                    snippet: session.preview(),
                    updated_at: session.updated_at,
                });
                continue;
            }
            // Check message content
            for (i, msg) in session.messages.iter().enumerate() {
                let text = msg.text_content();
                if text.to_lowercase().contains(&q) {
                    let start = text.to_lowercase().find(&q).unwrap_or(0);
                    let snippet_start = start.saturating_sub(40);
                    let snippet: String = text.chars().skip(snippet_start).take(120).collect();
                    results.push(SearchResult {
                        session_id: session.id.clone(),
                        title: session.title.clone(),
                        message_index: Some(i),
                        snippet: snippet.trim().to_string(),
                        updated_at: session.updated_at,
                    });
                    break; // One result per session
                }
            }
        }
        results
    }

    /// Fork a session up to a given message index, creating a new session that references the parent.
    pub fn fork(&mut self, id: &str, up_to_index: usize) -> Result<Session> {
        let parent = self.get(id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", id))?;
        let mut forked = Session::new();
        forked.title = format!("Fork of {}", parent.title);
        forked.messages = parent.messages[..=up_to_index.min(parent.messages.len().saturating_sub(1))].to_vec();
        self.persist(&forked)?;
        self.sessions.insert(0, forked.clone());
        Ok(forked)
    }
}

/// A search result with session context.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SearchResult {
    pub session_id: String,
    pub title: String,
    pub message_index: Option<usize>,
    pub snippet: String,
    pub updated_at: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_session_create() {
        let session = Session::new();
        assert_eq!(session.title, "New Chat");
        assert!(session.messages.is_empty());
    }

    #[test]
    fn test_auto_title() {
        let mut session = Session::new();
        session.messages.push(Message::text("user", "How do I build a Rust web server?"));
        session.auto_title();
        assert!(session.title.contains("Rust"));
    }

    #[test]
    fn test_session_manager_crud() {
        let tmp = TempDir::new().unwrap();
        let mut mgr = SessionManager::new(tmp.path()).unwrap();

        let session = mgr.create().unwrap();
        assert_eq!(mgr.list().len(), 1);

        assert!(mgr.get(&session.id).is_some());

        mgr.delete(&session.id).unwrap();
        assert_eq!(mgr.list().len(), 0);
    }

    #[test]
    fn test_persist_and_reload() {
        let tmp = TempDir::new().unwrap();
        let id;

        {
            let mut mgr = SessionManager::new(tmp.path()).unwrap();
            let session = mgr.create().unwrap();
            id = session.id.clone();
        }

        {
            let mgr = SessionManager::new(tmp.path()).unwrap();
            assert_eq!(mgr.list().len(), 1);
            assert!(mgr.get(&id).is_some());
        }
    }
}
