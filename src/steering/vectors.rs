// Ern-OS — Steering vector store

use super::SteeringVector;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Manages control vector files on disk.
pub struct VectorStore {
    dir: PathBuf,
    vectors: Vec<SteeringVector>,
}

impl VectorStore {
    pub fn new(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let mut store = Self { dir: dir.to_path_buf(), vectors: Vec::new() };
        store.scan()?;
        Ok(store)
    }

    /// Scan the directory for .gguf vector files.
    fn scan(&mut self) -> Result<()> {
        if !self.dir.exists() { return Ok(()); }
        for entry in std::fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().map_or(false, |e| e == "gguf") {
                let name = path.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                self.vectors.push(SteeringVector {
                    name: name.clone(),
                    path: path.to_string_lossy().to_string(),
                    strength: 1.0,
                    active: false,
                    description: format!("Control vector: {}", name),
                });
            }
        }
        tracing::info!(count = self.vectors.len(), "Steering vectors scanned");
        Ok(())
    }

    pub fn list(&self) -> &[SteeringVector] { &self.vectors }

    pub fn activate(&mut self, name: &str, strength: f32) -> Result<()> {
        let v = self.vectors.iter_mut().find(|v| v.name == name)
            .with_context(|| format!("Vector '{}' not found", name))?;
        v.active = true;
        v.strength = strength.clamp(0.0, 2.0);
        Ok(())
    }

    pub fn deactivate(&mut self, name: &str) -> Result<()> {
        let v = self.vectors.iter_mut().find(|v| v.name == name)
            .with_context(|| format!("Vector '{}' not found", name))?;
        v.active = false;
        Ok(())
    }

    pub fn active_vectors(&self) -> Vec<&SteeringVector> {
        self.vectors.iter().filter(|v| v.active).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_empty_store() {
        let tmp = TempDir::new().unwrap();
        let store = VectorStore::new(tmp.path()).unwrap();
        assert!(store.list().is_empty());
    }

    #[test]
    fn test_scan_gguf() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("curiosity.gguf"), b"fake").unwrap();
        let store = VectorStore::new(tmp.path()).unwrap();
        assert_eq!(store.list().len(), 1);
        assert_eq!(store.list()[0].name, "curiosity");
    }

    #[test]
    fn test_activate_deactivate() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("focus.gguf"), b"fake").unwrap();
        let mut store = VectorStore::new(tmp.path()).unwrap();
        store.activate("focus", 0.8).unwrap();
        assert_eq!(store.active_vectors().len(), 1);
        store.deactivate("focus").unwrap();
        assert!(store.active_vectors().is_empty());
    }
}
