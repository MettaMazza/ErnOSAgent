//! Session store — JSON file persistence for sessions.

use crate::provider::Message;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A persistent chat session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub title: String,
    pub model: String,
    pub provider: String,
    pub messages: Vec<Message>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl Session {
    /// Create a new empty session.
    pub fn new(model: &str, provider: &str) -> Self {
        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();
        Self {
            id,
            title: "New Session".to_string(),
            model: model.to_string(),
            provider: provider.to_string(),
            messages: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }

    /// Add a message to the session.
    pub fn add_message(&mut self, message: Message) {
        self.messages.push(message);
        self.updated_at = Utc::now();
    }

    /// Auto-generate a title from the first user message if title is still default.
    pub fn auto_title(&mut self) {
        if self.title == "New Session" {
            if let Some(first_user_msg) = self.messages.iter().find(|m| m.role == "user") {
                let content = &first_user_msg.content;
                self.title = if content.len() > 50 {
                    format!("{}...", &content[..50])
                } else {
                    content.clone()
                };
            }
        }
    }

    /// Save session to disk.
    pub fn save(&self, sessions_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(sessions_dir)
            .with_context(|| format!("Failed to create sessions dir: {}", sessions_dir.display()))?;

        let path = sessions_dir.join(format!("{}.json", self.id));
        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize session")?;
        std::fs::write(&path, content)
            .with_context(|| format!("Failed to write session file: {}", path.display()))?;

        tracing::debug!(session_id = %self.id, title = %self.title, "Session saved");
        Ok(())
    }

    /// Load a session from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read session file: {}", path.display()))?;
        let session: Self = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse session file: {}", path.display()))?;
        Ok(session)
    }

    /// File path for this session within a sessions directory.
    pub fn file_path(&self, sessions_dir: &Path) -> PathBuf {
        sessions_dir.join(format!("{}.json", self.id))
    }
}

/// List all session files in a directory.
pub fn list_session_files(sessions_dir: &Path) -> Result<Vec<PathBuf>> {
    if !sessions_dir.exists() {
        return Ok(Vec::new());
    }

    let mut files: Vec<PathBuf> = std::fs::read_dir(sessions_dir)
        .with_context(|| format!("Failed to read sessions dir: {}", sessions_dir.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().map_or(false, |ext| ext == "json"))
        .collect();

    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_new_session() {
        let session = Session::new("gemma4:26b", "llamacpp");
        assert_eq!(session.title, "New Session");
        assert!(session.messages.is_empty());
        assert!(!session.id.is_empty());
    }

    #[test]
    fn test_add_message() {
        let mut session = Session::new("test", "test");
        session.add_message(Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
            images: Vec::new(),
        });
        assert_eq!(session.messages.len(), 1);
    }

    #[test]
    fn test_auto_title() {
        let mut session = Session::new("test", "test");
        session.add_message(Message {
            role: "user".to_string(),
            content: "How do I write Rust code?".to_string(),
            images: Vec::new(),
        });
        session.auto_title();
        assert_eq!(session.title, "How do I write Rust code?");
    }

    #[test]
    fn test_auto_title_long() {
        let mut session = Session::new("test", "test");
        session.add_message(Message {
            role: "user".to_string(),
            content: "x".repeat(100),
            images: Vec::new(),
        });
        session.auto_title();
        assert!(session.title.len() <= 54); // 50 + "..."
    }

    #[test]
    fn test_save_and_load() {
        let tmp = TempDir::new().unwrap();
        let mut session = Session::new("model", "provider");
        session.add_message(Message {
            role: "user".to_string(),
            content: "test content".to_string(),
            images: Vec::new(),
        });

        session.save(tmp.path()).unwrap();
        let loaded = Session::load(&session.file_path(tmp.path())).unwrap();

        assert_eq!(loaded.id, session.id);
        assert_eq!(loaded.messages.len(), 1);
        assert_eq!(loaded.messages[0].content, "test content");
    }

    #[test]
    fn test_list_session_files() {
        let tmp = TempDir::new().unwrap();
        Session::new("a", "b").save(tmp.path()).unwrap();
        Session::new("c", "d").save(tmp.path()).unwrap();
        std::fs::write(tmp.path().join("not_session.txt"), "nope").unwrap();

        let files = list_session_files(tmp.path()).unwrap();
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_list_session_files_empty_dir() {
        let tmp = TempDir::new().unwrap();
        let files = list_session_files(tmp.path()).unwrap();
        assert!(files.is_empty());
    }

    #[test]
    fn test_list_session_files_nonexistent() {
        let files = list_session_files(Path::new("/nonexistent")).unwrap();
        assert!(files.is_empty());
    }
}
