// Ern-OS — Multi-agent system
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Agent registry — create, manage, and orchestrate multiple AI agents.
//!
//! Each agent is a fully-equipped Ern-OS instance with its own:
//! - Identity prompt (personality, name, communication style)
//! - Core prompt (protocols, behaviour rules)
//! - Observer rules (quality audit)
//! - Tool whitelist (which tools the agent can access)
//! - Session history (per-agent conversations)

pub mod teams;
pub mod parallel;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A single agent definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDefinition {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Per-agent prompt overrides. `None` = use system defaults.
    pub prompts: AgentPrompts,
    /// Whitelist of tool names this agent can use. Empty = all tools.
    #[serde(default)]
    pub tools: Vec<String>,
    /// Whether the observer audit is enabled for this agent.
    #[serde(default = "default_true")]
    pub observer_enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

fn default_true() -> bool { true }

/// Per-agent prompt paths. Each is relative to the agent's directory.
/// `None` means "use the system default from data/prompts/".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPrompts {
    pub identity: Option<String>,
    pub core: Option<String>,
    pub observer: Option<String>,
}

impl Default for AgentPrompts {
    fn default() -> Self {
        Self {
            identity: None,
            core: None,
            observer: None,
        }
    }
}

impl AgentDefinition {
    /// Create a new agent with defaults (inherits all system prompts).
    pub fn new(name: &str, description: &str) -> Self {
        let id = slug::slugify(name);
        Self {
            id,
            name: name.to_string(),
            description: description.to_string(),
            prompts: AgentPrompts::default(),
            tools: Vec::new(),
            observer_enabled: true,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    /// Whether this agent has a custom prompt override for the given name.
    pub fn has_custom_prompt(&self, name: &str) -> bool {
        match name {
            "identity" => self.prompts.identity.is_some(),
            "core" => self.prompts.core.is_some(),
            "observer" => self.prompts.observer.is_some(),
            _ => false,
        }
    }
}

/// Registry that manages all agent definitions on disk.
pub struct AgentRegistry {
    dir: PathBuf,
    data_dir: PathBuf,
    agents: Vec<AgentDefinition>,
}

impl AgentRegistry {
    /// Load all agents from `data/agents/`.
    pub fn new(data_dir: &Path) -> Result<Self> {
        let dir = data_dir.join("agents");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create agents dir: {}", dir.display()))?;

        let mut registry = Self {
            dir,
            data_dir: data_dir.to_path_buf(),
            agents: Vec::new(),
        };
        registry.load_all()?;
        tracing::info!(count = registry.agents.len(), "Agent registry loaded");
        Ok(registry)
    }

    fn load_all(&mut self) -> Result<()> {
        for entry in std::fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().map_or(false, |e| e == "json") {
                match std::fs::read_to_string(&path) {
                    Ok(content) => match serde_json::from_str::<AgentDefinition>(&content) {
                        Ok(agent) => {
                            tracing::debug!(id = %agent.id, name = %agent.name, "Loaded agent");
                            self.agents.push(agent);
                        }
                        Err(e) => tracing::warn!(
                            path = %path.display(), error = %e,
                            "Skipped corrupt agent file"
                        ),
                    },
                    Err(e) => tracing::warn!(
                        path = %path.display(), error = %e,
                        "Failed to read agent file"
                    ),
                }
            }
        }
        self.agents.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(())
    }

    fn persist(&self, agent: &AgentDefinition) -> Result<()> {
        let path = self.dir.join(format!("{}.json", agent.id));
        let content = serde_json::to_string_pretty(agent)
            .context("Failed to serialize agent")?;
        std::fs::write(&path, content)
            .with_context(|| format!("Failed to write agent: {}", path.display()))?;
        tracing::info!(id = %agent.id, name = %agent.name, "Agent persisted");
        Ok(())
    }

    /// Create a new agent and persist it.
    pub fn create(&mut self, mut agent: AgentDefinition) -> Result<AgentDefinition> {
        // Ensure unique ID
        if self.agents.iter().any(|a| a.id == agent.id) {
            let suffix = &agent.created_at.timestamp().to_string()[..6];
            agent.id = format!("{}-{}", agent.id, suffix);
        }

        // Create agent's prompt directory
        let agent_dir = self.data_dir.join("agents").join(&agent.id);
        std::fs::create_dir_all(&agent_dir)?;

        self.persist(&agent)?;
        self.agents.push(agent.clone());
        self.agents.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(agent)
    }

    pub fn get(&self, id: &str) -> Option<&AgentDefinition> {
        self.agents.iter().find(|a| a.id == id)
    }

    pub fn update(&mut self, agent: AgentDefinition) -> Result<()> {
        self.persist(&agent)?;
        if let Some(existing) = self.agents.iter_mut().find(|a| a.id == agent.id) {
            *existing = agent;
        }
        Ok(())
    }

    pub fn delete(&mut self, id: &str) -> Result<()> {
        // Remove JSON file
        let path = self.dir.join(format!("{}.json", id));
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        // Remove agent prompt directory
        let agent_dir = self.data_dir.join("agents").join(id);
        if agent_dir.exists() {
            std::fs::remove_dir_all(&agent_dir)?;
        }
        self.agents.retain(|a| a.id != id);
        tracing::info!(id = %id, "Agent deleted");
        Ok(())
    }

    pub fn list(&self) -> &[AgentDefinition] {
        &self.agents
    }

    /// Resolve a prompt for a specific agent.
    /// If the agent has a custom prompt, load it from the agent's directory.
    /// Otherwise, fall back to the system default from data/prompts/.
    pub fn resolve_prompt(&self, agent_id: &str, prompt_name: &str) -> Result<String> {
        let agent = self.get(agent_id);

        // Check for agent-specific override
        let custom_path = agent.and_then(|a| {
            match prompt_name {
                "identity" => a.prompts.identity.as_ref(),
                "core" => a.prompts.core.as_ref(),
                "observer" => a.prompts.observer.as_ref(),
                _ => None,
            }
        });

        if let Some(rel_path) = custom_path {
            let path = self.data_dir.join(rel_path);
            return std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read agent prompt: {}", path.display()));
        }

        // Fall back to system default
        let default_path = self.data_dir.join("prompts").join(format!("{}.md", prompt_name));
        std::fs::read_to_string(&default_path)
            .with_context(|| format!("Failed to read system prompt: {}", default_path.display()))
    }

    /// Check if a tool is allowed for this agent.
    /// Empty whitelist = all tools allowed.
    pub fn is_tool_allowed(&self, agent_id: &str, tool_name: &str) -> bool {
        match self.get(agent_id) {
            Some(agent) => {
                if agent.tools.is_empty() {
                    true // empty whitelist = all tools
                } else {
                    agent.tools.iter().any(|t| t == tool_name)
                }
            }
            None => true, // unknown agent = allow all
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn setup() -> (TempDir, AgentRegistry) {
        let tmp = TempDir::new().unwrap();
        // Create system prompts dir
        let prompts_dir = tmp.path().join("prompts");
        std::fs::create_dir_all(&prompts_dir).unwrap();
        std::fs::write(prompts_dir.join("identity.md"), "Default identity").unwrap();
        std::fs::write(prompts_dir.join("core.md"), "Default core").unwrap();
        std::fs::write(prompts_dir.join("observer.md"), "Default observer").unwrap();

        let registry = AgentRegistry::new(tmp.path()).unwrap();
        (tmp, registry)
    }

    #[test]
    fn test_create_agent() {
        let (_tmp, mut registry) = setup();
        let agent = AgentDefinition::new("Code Reviewer", "Reviews code");
        let created = registry.create(agent).unwrap();
        assert_eq!(created.id, "code-reviewer");
        assert_eq!(registry.list().len(), 1);
    }

    #[test]
    fn test_get_agent() {
        let (_tmp, mut registry) = setup();
        let agent = AgentDefinition::new("Test Agent", "Test");
        registry.create(agent).unwrap();
        assert!(registry.get("test-agent").is_some());
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_delete_agent() {
        let (_tmp, mut registry) = setup();
        let agent = AgentDefinition::new("Deleteme", "Will be deleted");
        registry.create(agent).unwrap();
        assert_eq!(registry.list().len(), 1);
        registry.delete("deleteme").unwrap();
        assert_eq!(registry.list().len(), 0);
    }

    #[test]
    fn test_resolve_prompt_default() {
        let (_tmp, mut registry) = setup();
        let agent = AgentDefinition::new("Vanilla", "Uses defaults");
        registry.create(agent).unwrap();

        let identity = registry.resolve_prompt("vanilla", "identity").unwrap();
        assert_eq!(identity, "Default identity");
    }

    #[test]
    fn test_resolve_prompt_custom() {
        let (tmp, mut registry) = setup();
        let agent_dir = tmp.path().join("agents").join("custom-bot");
        std::fs::create_dir_all(&agent_dir).unwrap();
        std::fs::write(agent_dir.join("identity.md"), "I am custom").unwrap();

        let mut agent = AgentDefinition::new("Custom Bot", "Custom identity");
        agent.prompts.identity = Some("agents/custom-bot/identity.md".to_string());
        registry.create(agent).unwrap();

        let identity = registry.resolve_prompt("custom-bot", "identity").unwrap();
        assert_eq!(identity, "I am custom");
    }

    #[test]
    fn test_tool_whitelist() {
        let (_tmp, mut registry) = setup();
        let mut agent = AgentDefinition::new("Limited", "Limited tools");
        agent.tools = vec!["file_read".to_string(), "web_search".to_string()];
        registry.create(agent).unwrap();

        assert!(registry.is_tool_allowed("limited", "file_read"));
        assert!(registry.is_tool_allowed("limited", "web_search"));
        assert!(!registry.is_tool_allowed("limited", "run_bash_command"));
    }

    #[test]
    fn test_empty_whitelist_allows_all() {
        let (_tmp, mut registry) = setup();
        let agent = AgentDefinition::new("Full Access", "All tools");
        registry.create(agent).unwrap();
        assert!(registry.is_tool_allowed("full-access", "run_bash_command"));
        assert!(registry.is_tool_allowed("full-access", "anything"));
    }

    #[test]
    fn test_persist_and_reload() {
        let (tmp, mut registry) = setup();
        let agent = AgentDefinition::new("Persistent", "Survives reload");
        registry.create(agent).unwrap();

        let registry2 = AgentRegistry::new(tmp.path()).unwrap();
        assert_eq!(registry2.list().len(), 1);
        assert_eq!(registry2.list()[0].name, "Persistent");
    }
}
