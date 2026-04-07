//! Tier 7: Procedures — reusable multi-step workflow templates.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Procedure {
    pub id: String,
    pub name: String,
    pub steps: Vec<String>,
    pub success_count: usize,
}

pub struct ProcedureStore {
    procedures: Vec<Procedure>,
}

impl ProcedureStore {
    pub fn new() -> Self { Self { procedures: Vec::new() } }

    pub fn add(&mut self, name: &str, steps: Vec<String>) {
        self.procedures.push(Procedure {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(), steps, success_count: 0,
        });
    }

    pub fn record_success(&mut self, id: &str) {
        if let Some(p) = self.procedures.iter_mut().find(|p| p.id == id) {
            p.success_count += 1;
        }
    }

    pub fn find_by_name(&self, name: &str) -> Option<&Procedure> {
        self.procedures.iter().find(|p| p.name.contains(name))
    }

    pub fn all(&self) -> &[Procedure] { &self.procedures }
    pub fn count(&self) -> usize { self.procedures.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_procedure() {
        let mut store = ProcedureStore::new();
        store.add("deploy", vec!["build".to_string(), "test".to_string(), "push".to_string()]);
        assert_eq!(store.count(), 1);
        assert_eq!(store.procedures[0].steps.len(), 3);
    }
}
