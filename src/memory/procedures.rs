// Ern-OS — Tier 7: Procedures — reusable workflow templates (Self-Skills)

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcedureStep {
    pub tool: String,
    pub purpose: String,
    #[serde(default)]
    pub instruction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Procedure {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub steps: Vec<ProcedureStep>,
    pub success_count: usize,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
}

pub struct ProcedureStore {
    procedures: Vec<Procedure>,
    file_path: Option<PathBuf>,
}

impl ProcedureStore {
    pub fn new() -> Self { Self { procedures: Vec::new(), file_path: None } }

    pub fn open(path: &Path) -> Result<Self> {
        tracing::info!(module = "procedures", fn_name = "open", "procedures::open called");
        let mut store = Self { procedures: Vec::new(), file_path: Some(path.to_path_buf()) };
        if path.exists() {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read procedures: {}", path.display()))?;
            store.procedures = serde_json::from_str(&content)?;
        }
        Ok(store)
    }

    fn persist(&self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
            std::fs::write(path, serde_json::to_string_pretty(&self.procedures)?)?;
        }
        Ok(())
    }

    pub fn add(&mut self, name: &str, steps: Vec<ProcedureStep>) -> Result<()> {
        tracing::info!(module = "procedures", fn_name = "add", "procedures::add called");
        self.procedures.push(Procedure {
            id: uuid::Uuid::new_v4().to_string(), name: name.to_string(),
            description: String::new(), steps, success_count: 0, last_used: None,
        });
        self.persist()
    }

    /// Add a skill only if no procedure with the same name exists.
    pub fn add_if_new(&mut self, name: &str, description: &str, steps: Vec<ProcedureStep>) -> Result<bool> {
        tracing::info!(module = "procedures", fn_name = "add_if_new", "procedures::add_if_new called");
        if self.find_by_name(name).is_some() {
            tracing::debug!(name, "Skill deduplicated — similar procedure exists");
            return Ok(false);
        }
        self.procedures.push(Procedure {
            id: uuid::Uuid::new_v4().to_string(), name: name.to_string(),
            description: description.to_string(), steps, success_count: 0, last_used: None,
        });
        self.persist()?;
        Ok(true)
    }

    /// Refine an existing procedure's steps.
    pub fn refine(&mut self, id: &str, steps: Vec<ProcedureStep>) -> Result<()> {
        let proc = self.procedures.iter_mut().find(|p| p.id == id)
            .context(format!("Procedure '{}' not found for refinement", id))?;
        proc.steps = steps;
        proc.last_used = Some(chrono::Utc::now());
        self.persist()
    }

    pub fn record_success(&mut self, id: &str) -> Result<()> {
        if let Some(p) = self.procedures.iter_mut().find(|p| p.id == id) {
            p.success_count += 1;
            p.last_used = Some(chrono::Utc::now());
            self.persist()?;
        }
        Ok(())
    }

    /// Record success by name (used by delayed reinforcement).
    pub fn record_success_by_name(&mut self, name: &str) -> Result<()> {
        let lower = name.to_lowercase();
        if let Some(p) = self.procedures.iter_mut().find(|p| p.name.to_lowercase() == lower) {
            p.success_count += 1;
            p.last_used = Some(chrono::Utc::now());
            self.persist()?;
        }
        Ok(())
    }

    pub fn remove(&mut self, id: &str) -> Result<()> {
        let before = self.procedures.len();
        self.procedures.retain(|p| p.id != id);
        if self.procedures.len() == before { anyhow::bail!("Procedure '{}' not found", id); }
        self.persist()
    }

    pub fn find_by_name(&self, name: &str) -> Option<&Procedure> {
        let lower = name.to_lowercase();
        self.procedures.iter().find(|p| p.name.to_lowercase().contains(&lower))
    }

    pub fn all(&self) -> &[Procedure] { &self.procedures }
    pub fn count(&self) -> usize { self.procedures.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_find() {
        let mut store = ProcedureStore::new();
        store.add("Deploy", vec![
            ProcedureStep { tool: "shell".into(), purpose: "build".into(), instruction: "cargo build".into() },
        ]).unwrap();
        assert!(store.find_by_name("deploy").is_some());
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_add_if_new_deduplicates() {
        let mut store = ProcedureStore::new();
        assert!(store.add_if_new("Deploy", "deploy app", vec![]).unwrap());
        assert!(!store.add_if_new("Deploy", "deploy app v2", vec![]).unwrap());
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_add_if_new_allows_different() {
        let mut store = ProcedureStore::new();
        store.add_if_new("Deploy", "deploy app", vec![]).unwrap();
        assert!(store.add_if_new("Test Suite", "run tests", vec![]).unwrap());
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn test_refine_updates_steps() {
        let mut store = ProcedureStore::new();
        store.add_if_new("Deploy", "deploy app", vec![
            ProcedureStep { tool: "shell".into(), purpose: "build".into(), instruction: "cargo build".into() },
        ]).unwrap();
        let id = store.all()[0].id.clone();
        store.refine(&id, vec![
            ProcedureStep { tool: "shell".into(), purpose: "test".into(), instruction: "cargo test".into() },
            ProcedureStep { tool: "shell".into(), purpose: "build".into(), instruction: "cargo build --release".into() },
        ]).unwrap();
        assert_eq!(store.all()[0].steps.len(), 2);
    }
}
