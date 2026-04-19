// Ern-OS — Team orchestration and swarm mechanics
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Teams & Swarms — orchestrate multiple agents for collaborative tasks.
//!
//! Execution modes:
//! - **Parallel**: All agents work simultaneously, results collected
//! - **Sequential**: Pipeline — output of agent N feeds into agent N+1
//!
//! Swarm: Dynamic N-copy scaling of one agent template with split/aggregate.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// How agents in a team execute tasks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ExecutionMode {
    /// All agents work simultaneously on the same task
    Parallel,
    /// Pipeline — output of agent N becomes input to agent N+1
    Sequential,
}

impl Default for ExecutionMode {
    fn default() -> Self {
        Self::Parallel
    }
}

/// A team of agents that work together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamDefinition {
    pub id: String,
    pub name: String,
    pub description: String,
    pub mode: ExecutionMode,
    /// Ordered list of agent IDs in this team
    pub agents: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl TeamDefinition {
    pub fn new(name: &str, description: &str, mode: ExecutionMode, agents: Vec<String>) -> Self {
        Self {
            id: slug::slugify(name),
            name: name.to_string(),
            description: description.to_string(),
            mode,
            agents,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }
}

/// How to split work for a swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SplitStrategy {
    /// Divide the task into N sub-tasks
    Divide,
    /// Give the same task to all N agents
    Duplicate,
}

/// A swarm task — runtime definition for dynamic agent scaling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmTask {
    /// Agent template to clone
    pub template_agent: String,
    /// Number of instances to spawn
    pub count: usize,
    /// The task to accomplish
    pub task: String,
    /// How to split work
    pub split_strategy: SplitStrategy,
}

/// Result from a single agent in a team or swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub agent_id: String,
    pub agent_name: String,
    pub output: String,
    pub elapsed_ms: u64,
    pub success: bool,
}

/// Manages team definitions on disk.
pub struct TeamRegistry {
    dir: PathBuf,
    teams: Vec<TeamDefinition>,
}

impl TeamRegistry {
    pub fn new(data_dir: &Path) -> Result<Self> {
        let dir = data_dir.join("teams");
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("Failed to create teams dir: {}", dir.display()))?;

        let mut registry = Self {
            dir,
            teams: Vec::new(),
        };
        registry.load_all()?;
        tracing::info!(count = registry.teams.len(), "Team registry loaded");
        Ok(registry)
    }

    fn load_all(&mut self) -> Result<()> {
        for entry in std::fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().map_or(false, |e| e == "json") {
                match std::fs::read_to_string(&path) {
                    Ok(content) => match serde_json::from_str::<TeamDefinition>(&content) {
                        Ok(team) => {
                            tracing::debug!(id = %team.id, name = %team.name, "Loaded team");
                            self.teams.push(team);
                        }
                        Err(e) => tracing::warn!(
                            path = %path.display(), error = %e,
                            "Skipped corrupt team file"
                        ),
                    },
                    Err(e) => tracing::warn!(
                        path = %path.display(), error = %e,
                        "Failed to read team file"
                    ),
                }
            }
        }
        self.teams.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(())
    }

    fn persist(&self, team: &TeamDefinition) -> Result<()> {
        let path = self.dir.join(format!("{}.json", team.id));
        let content = serde_json::to_string_pretty(team)
            .context("Failed to serialize team")?;
        std::fs::write(&path, content)
            .with_context(|| format!("Failed to write team: {}", path.display()))?;
        Ok(())
    }

    pub fn create(&mut self, mut team: TeamDefinition) -> Result<TeamDefinition> {
        if self.teams.iter().any(|t| t.id == team.id) {
            let suffix = &team.created_at.timestamp().to_string()[..6];
            team.id = format!("{}-{}", team.id, suffix);
        }
        self.persist(&team)?;
        self.teams.push(team.clone());
        self.teams.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(team)
    }

    pub fn get(&self, id: &str) -> Option<&TeamDefinition> {
        self.teams.iter().find(|t| t.id == id)
    }

    pub fn update(&mut self, team: TeamDefinition) -> Result<()> {
        self.persist(&team)?;
        if let Some(existing) = self.teams.iter_mut().find(|t| t.id == team.id) {
            *existing = team;
        }
        Ok(())
    }

    pub fn delete(&mut self, id: &str) -> Result<()> {
        let path = self.dir.join(format!("{}.json", id));
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        self.teams.retain(|t| t.id != id);
        tracing::info!(id = %id, "Team deleted");
        Ok(())
    }

    pub fn list(&self) -> &[TeamDefinition] {
        &self.teams
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_create_team() {
        let tmp = TempDir::new().unwrap();
        let mut registry = TeamRegistry::new(tmp.path()).unwrap();
        let team = TeamDefinition::new(
            "Dev Team",
            "Full stack review pipeline",
            ExecutionMode::Sequential,
            vec!["code-reviewer".into(), "security-auditor".into()],
        );
        let created = registry.create(team).unwrap();
        assert_eq!(created.id, "dev-team");
        assert_eq!(created.agents.len(), 2);
        assert_eq!(created.mode, ExecutionMode::Sequential);
    }

    #[test]
    fn test_team_crud() {
        let tmp = TempDir::new().unwrap();
        let mut registry = TeamRegistry::new(tmp.path()).unwrap();

        let team = TeamDefinition::new("Test", "Test team", ExecutionMode::Parallel, vec![]);
        registry.create(team).unwrap();
        assert_eq!(registry.list().len(), 1);

        registry.delete("test").unwrap();
        assert_eq!(registry.list().len(), 0);
    }

    #[test]
    fn test_persist_and_reload() {
        let tmp = TempDir::new().unwrap();
        {
            let mut registry = TeamRegistry::new(tmp.path()).unwrap();
            let team = TeamDefinition::new("Persistent", "Survives reload", ExecutionMode::Parallel, vec!["a".into()]);
            registry.create(team).unwrap();
        }
        let registry = TeamRegistry::new(tmp.path()).unwrap();
        assert_eq!(registry.list().len(), 1);
        assert_eq!(registry.list()[0].name, "Persistent");
    }

    #[test]
    fn test_execution_modes() {
        assert_eq!(ExecutionMode::default(), ExecutionMode::Parallel);
    }
}
