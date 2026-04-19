//! On-demand skill loading — markdown-based project knowledge.
//!
//! Skills are markdown files in `data/skills/` with YAML frontmatter.
//! Only metadata (name + description) is loaded into the prompt;
//! full content is loaded on-demand when the agent invokes a skill.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// A single skill definition.
#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub file_path: PathBuf,
    /// Content is lazily loaded — `None` until explicitly requested.
    content: Option<String>,
}

/// Registry of available skills.
pub struct SkillRegistry {
    skills: Vec<Skill>,
    dir: PathBuf,
}

impl SkillRegistry {
    /// Load skill metadata from all markdown files in the skills directory.
    pub fn new(data_dir: &Path) -> Self {
        let dir = data_dir.join("skills");
        let _ = std::fs::create_dir_all(&dir);
        let skills = load_skill_metadata(&dir);
        tracing::info!(count = skills.len(), "Skill registry loaded");
        Self { skills, dir }
    }

    /// Get skill summaries for injection into system prompt metadata.
    pub fn skill_summaries(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }
        let mut out = String::from("[Available Skills]\n");
        for skill in &self.skills {
            out.push_str(&format!("• {}: {}\n", skill.name, skill.description));
        }
        out
    }

    /// Load the full content of a specific skill by name.
    pub fn load_skill_content(&mut self, name: &str) -> Result<String> {
        if let Some(skill) = self.skills.iter_mut().find(|s| s.name == name) {
            if let Some(ref content) = skill.content {
                return Ok(content.clone());
            }
            let content = std::fs::read_to_string(&skill.file_path)
                .with_context(|| format!("Failed to read skill: {}", skill.file_path.display()))?;
            let body = strip_frontmatter(&content);
            skill.content = Some(body.clone());
            Ok(body)
        } else {
            anyhow::bail!("Skill '{}' not found. Available: {}", name, self.skill_names().join(", "))
        }
    }

    /// List all skill names.
    pub fn skill_names(&self) -> Vec<&str> {
        self.skills.iter().map(|s| s.name.as_str()).collect()
    }

    /// Count of registered skills.
    pub fn count(&self) -> usize {
        self.skills.len()
    }

    /// Reload skills from disk.
    pub fn reload(&mut self) {
        self.skills = load_skill_metadata(&self.dir);
        tracing::info!(count = self.skills.len(), "Skills reloaded");
    }
}

/// Load skill metadata from markdown frontmatter in the skills directory.
fn load_skill_metadata(dir: &Path) -> Vec<Skill> {
    let mut skills = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return skills,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().map_or(false, |e| e == "md") {
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Some((name, desc)) = parse_frontmatter(&content) {
                    skills.push(Skill {
                        name,
                        description: desc,
                        file_path: path,
                        content: None,
                    });
                }
            }
        }
    }
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

/// Parse YAML frontmatter for name and description.
fn parse_frontmatter(content: &str) -> Option<(String, String)> {
    let trimmed = content.trim();
    if !trimmed.starts_with("---") { return None; }
    let rest = &trimmed[3..];
    let end = rest.find("---")?;
    let frontmatter = &rest[..end];

    let mut name = None;
    let mut description = None;

    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("name:") {
            name = Some(val.trim().to_string());
        } else if let Some(val) = line.strip_prefix("description:") {
            description = Some(val.trim().to_string());
        }
    }

    Some((name?, description.unwrap_or_default()))
}

/// Strip YAML frontmatter from markdown content.
fn strip_frontmatter(content: &str) -> String {
    let trimmed = content.trim();
    if !trimmed.starts_with("---") { return trimmed.to_string(); }
    let rest = &trimmed[3..];
    match rest.find("---") {
        Some(end) => rest[end + 3..].trim().to_string(),
        None => trimmed.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_frontmatter() {
        let content = "---\nname: rust-errors\ndescription: Error handling patterns\n---\n# Content";
        let (name, desc) = parse_frontmatter(content).unwrap();
        assert_eq!(name, "rust-errors");
        assert_eq!(desc, "Error handling patterns");
    }

    #[test]
    fn test_parse_frontmatter_missing() {
        assert!(parse_frontmatter("No frontmatter here").is_none());
    }

    #[test]
    fn test_strip_frontmatter() {
        let content = "---\nname: test\n---\n# Body content";
        let body = strip_frontmatter(content);
        assert_eq!(body, "# Body content");
    }

    #[test]
    fn test_strip_frontmatter_none() {
        let body = strip_frontmatter("Just plain text");
        assert_eq!(body, "Just plain text");
    }

    #[test]
    fn test_skill_registry_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let registry = SkillRegistry::new(tmp.path());
        assert_eq!(registry.count(), 0);
        assert!(registry.skill_summaries().is_empty());
    }

    #[test]
    fn test_skill_registry_with_skills() {
        let tmp = tempfile::TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(
            skills_dir.join("test.md"),
            "---\nname: test-skill\ndescription: A test skill\n---\n# Test\nDo the thing."
        ).unwrap();

        let mut registry = SkillRegistry::new(tmp.path());
        assert_eq!(registry.count(), 1);
        assert!(registry.skill_summaries().contains("test-skill"));

        let content = registry.load_skill_content("test-skill").unwrap();
        assert!(content.contains("Do the thing"));
    }

    #[test]
    fn test_load_nonexistent_skill() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut registry = SkillRegistry::new(tmp.path());
        assert!(registry.load_skill_content("nope").is_err());
    }

    #[test]
    fn test_skill_names() {
        let tmp = tempfile::TempDir::new().unwrap();
        let skills_dir = tmp.path().join("skills");
        std::fs::create_dir_all(&skills_dir).unwrap();
        std::fs::write(skills_dir.join("a.md"), "---\nname: alpha\ndescription: A\n---\nContent").unwrap();
        std::fs::write(skills_dir.join("b.md"), "---\nname: beta\ndescription: B\n---\nContent").unwrap();

        let registry = SkillRegistry::new(tmp.path());
        let names = registry.skill_names();
        assert_eq!(names, vec!["alpha", "beta"]);
    }
}
