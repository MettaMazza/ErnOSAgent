//! Control vector management — load, scale, compose GGUF control vectors.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A loaded control vector with its scaling configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedVector {
    /// Path to the .gguf control vector file.
    pub path: PathBuf,
    /// Human-readable name (derived from filename if not set).
    pub name: String,
    /// Scale factor. Positive = toward trait, negative = away.
    pub scale: f64,
    /// Whether this vector is currently active.
    pub active: bool,
}

/// Full steering configuration for the llama-server instance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SteeringConfig {
    /// Loaded control vectors.
    pub vectors: Vec<LoadedVector>,
    /// Optional layer range for targeted steering. (start, end) inclusive.
    pub layer_range: Option<(u32, u32)>,
}

impl SteeringConfig {
    /// Build the CLI arguments for llama-server from the current steering config.
    pub fn to_server_args(&self) -> Vec<String> {
        let mut args = Vec::new();

        for vector in &self.vectors {
            if !vector.active {
                continue;
            }

            if (vector.scale - 1.0).abs() < f64::EPSILON {
                // Scale is exactly 1.0 — use simple --control-vector
                args.push("--control-vector".to_string());
                args.push(vector.path.display().to_string());
            } else {
                // Use scaled variant
                args.push("--control-vector-scaled".to_string());
                args.push(format!("{}:{}", vector.path.display(), vector.scale));
            }
        }

        if let Some((start, end)) = self.layer_range {
            args.push("--control-vector-layer-range".to_string());
            args.push(start.to_string());
            args.push(end.to_string());
        }

        args
    }

    /// Scan a directory for available .gguf vector files.
    pub fn scan_directory(vectors_dir: &Path) -> Result<Vec<LoadedVector>> {
        let mut vectors = Vec::new();

        if !vectors_dir.exists() {
            return Ok(vectors);
        }

        let entries = std::fs::read_dir(vectors_dir)
            .with_context(|| format!("Failed to read vectors directory: {}", vectors_dir.display()))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.extension().map_or(false, |ext| ext == "gguf") {
                let name = path
                    .file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unnamed".to_string());

                vectors.push(LoadedVector {
                    path,
                    name,
                    scale: 1.0,
                    active: false,
                });
            }
        }

        vectors.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(vectors)
    }

    /// Load a vector by path and activate it.
    pub fn load_vector(&mut self, path: PathBuf, scale: f64) -> Result<()> {
        if !path.exists() {
            anyhow::bail!("Vector file does not exist: {}", path.display());
        }
        if path.extension().map_or(true, |ext| ext != "gguf") {
            anyhow::bail!("Vector file must be a .gguf file: {}", path.display());
        }

        let name = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unnamed".to_string());

        // Check if already loaded
        if let Some(existing) = self.vectors.iter_mut().find(|v| v.path == path) {
            existing.scale = scale;
            existing.active = true;
            tracing::info!(name = %existing.name, scale = scale, "Updated existing vector");
        } else {
            self.vectors.push(LoadedVector {
                path,
                name: name.clone(),
                scale,
                active: true,
            });
            tracing::info!(name = %name, scale = scale, "Loaded new vector");
        }

        Ok(())
    }

    /// Set the scale for a loaded vector by name.
    pub fn set_scale(&mut self, name: &str, scale: f64) -> Result<()> {
        let vector = self
            .vectors
            .iter_mut()
            .find(|v| v.name == name)
            .with_context(|| format!("Vector '{}' not found in loaded vectors", name))?;

        vector.scale = scale;
        tracing::info!(name = %name, scale = scale, "Vector scale updated");
        Ok(())
    }

    /// Remove a vector by name.
    pub fn remove_vector(&mut self, name: &str) -> Result<()> {
        let initial_len = self.vectors.len();
        self.vectors.retain(|v| v.name != name);

        if self.vectors.len() == initial_len {
            anyhow::bail!("Vector '{}' not found", name);
        }

        tracing::info!(name = %name, "Vector removed");
        Ok(())
    }

    /// List all active vectors.
    pub fn active_vectors(&self) -> Vec<&LoadedVector> {
        self.vectors.iter().filter(|v| v.active).collect()
    }

    /// Check if any steering is active.
    pub fn has_active_vectors(&self) -> bool {
        self.vectors.iter().any(|v| v.active)
    }

    /// Human-readable summary for status bar.
    pub fn status_summary(&self) -> String {
        let active = self.active_vectors();
        if active.is_empty() {
            "No steering".to_string()
        } else {
            active
                .iter()
                .map(|v| format!("{}×{:.1}", v.name, v.scale))
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_to_server_args_empty() {
        let config = SteeringConfig::default();
        assert!(config.to_server_args().is_empty());
    }

    #[test]
    fn test_to_server_args_single_vector() {
        let config = SteeringConfig {
            vectors: vec![LoadedVector {
                path: PathBuf::from("/vectors/honesty.gguf"),
                name: "honesty".to_string(),
                scale: 1.0,
                active: true,
            }],
            layer_range: None,
        };
        let args = config.to_server_args();
        assert_eq!(args, vec!["--control-vector", "/vectors/honesty.gguf"]);
    }

    #[test]
    fn test_to_server_args_scaled_vector() {
        let config = SteeringConfig {
            vectors: vec![LoadedVector {
                path: PathBuf::from("/vectors/creativity.gguf"),
                name: "creativity".to_string(),
                scale: 1.5,
                active: true,
            }],
            layer_range: None,
        };
        let args = config.to_server_args();
        assert_eq!(
            args,
            vec!["--control-vector-scaled", "/vectors/creativity.gguf:1.5"]
        );
    }

    #[test]
    fn test_to_server_args_with_layer_range() {
        let config = SteeringConfig {
            vectors: vec![LoadedVector {
                path: PathBuf::from("/vectors/test.gguf"),
                name: "test".to_string(),
                scale: 1.0,
                active: true,
            }],
            layer_range: Some((10, 20)),
        };
        let args = config.to_server_args();
        assert!(args.contains(&"--control-vector-layer-range".to_string()));
        assert!(args.contains(&"10".to_string()));
        assert!(args.contains(&"20".to_string()));
    }

    #[test]
    fn test_to_server_args_inactive_vector_skipped() {
        let config = SteeringConfig {
            vectors: vec![LoadedVector {
                path: PathBuf::from("/vectors/test.gguf"),
                name: "test".to_string(),
                scale: 1.0,
                active: false,
            }],
            layer_range: None,
        };
        assert!(config.to_server_args().is_empty());
    }

    #[test]
    fn test_scan_directory() {
        let tmp = TempDir::new().unwrap();
        std::fs::write(tmp.path().join("honesty.gguf"), b"fake").unwrap();
        std::fs::write(tmp.path().join("creativity.gguf"), b"fake").unwrap();
        std::fs::write(tmp.path().join("not_a_vector.txt"), b"fake").unwrap();

        let vectors = SteeringConfig::scan_directory(tmp.path()).unwrap();
        assert_eq!(vectors.len(), 2);
        assert!(vectors.iter().any(|v| v.name == "honesty"));
        assert!(vectors.iter().any(|v| v.name == "creativity"));
    }

    #[test]
    fn test_scan_directory_nonexistent() {
        let vectors = SteeringConfig::scan_directory(Path::new("/nonexistent/path")).unwrap();
        assert!(vectors.is_empty());
    }

    #[test]
    fn test_load_remove_vector() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.gguf");
        std::fs::write(&path, b"fake").unwrap();

        let mut config = SteeringConfig::default();
        config.load_vector(path, 1.5).unwrap();

        assert_eq!(config.vectors.len(), 1);
        assert_eq!(config.vectors[0].name, "test");
        assert_eq!(config.vectors[0].scale, 1.5);
        assert!(config.vectors[0].active);

        config.remove_vector("test").unwrap();
        assert!(config.vectors.is_empty());
    }

    #[test]
    fn test_set_scale() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("v.gguf");
        std::fs::write(&path, b"fake").unwrap();

        let mut config = SteeringConfig::default();
        config.load_vector(path, 1.0).unwrap();
        config.set_scale("v", 2.5).unwrap();

        assert_eq!(config.vectors[0].scale, 2.5);
    }

    #[test]
    fn test_set_scale_not_found() {
        let mut config = SteeringConfig::default();
        assert!(config.set_scale("nonexistent", 1.0).is_err());
    }

    #[test]
    fn test_status_summary() {
        let config = SteeringConfig {
            vectors: vec![
                LoadedVector {
                    path: PathBuf::from("a.gguf"), name: "honesty".to_string(),
                    scale: 1.5, active: true,
                },
                LoadedVector {
                    path: PathBuf::from("b.gguf"), name: "creativity".to_string(),
                    scale: 0.8, active: true,
                },
                LoadedVector {
                    path: PathBuf::from("c.gguf"), name: "inactive".to_string(),
                    scale: 1.0, active: false,
                },
            ],
            layer_range: None,
        };
        let summary = config.status_summary();
        assert!(summary.contains("honesty×1.5"));
        assert!(summary.contains("creativity×0.8"));
        assert!(!summary.contains("inactive"));
    }

    #[test]
    fn test_has_active_vectors() {
        let mut config = SteeringConfig::default();
        assert!(!config.has_active_vectors());

        config.vectors.push(LoadedVector {
            path: PathBuf::from("a.gguf"), name: "a".to_string(),
            scale: 1.0, active: true,
        });
        assert!(config.has_active_vectors());
    }
}
