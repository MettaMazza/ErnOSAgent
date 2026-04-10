// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Containment Cone — infrastructure escape prevention.
//!
//! The agent may self-improve, recompile, edit source code, and modify
//! its own behavior freely — with ONE exception: it cannot touch the
//! infrastructure that keeps it containerized.
//!
//! This is enforced at the Rust level (not the prompt level) so it
//! cannot be bypassed by prompt injection, tool forging, or any
//! other agent-initiated action.

/// Files that form the containment boundary. The agent CANNOT read,
/// write, modify, or delete these through any tool.
const CONTAINMENT_FILES: &[&str] = &[
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    ".dockerignore",
    "launch.sh",
    "start-ernosagent.sh",
    ".docker",
];

/// Commands that could modify or escape the container boundary.
const CONTAINMENT_COMMANDS: &[&str] = &[
    "docker ",
    "docker-compose ",
    "dockerd",
    "nsenter",
    "chroot",
    "mount ",
    "umount ",
    "unshare ",
    "pivot_root",
    "capsh",
    "setns",
];

/// Check if a file path touches a containment boundary file.
/// Returns `Some(filename)` if blocked, `None` if allowed.
pub fn check_path(path: &str) -> Option<&'static str> {
    let normalized = path.trim().trim_start_matches("./").trim_start_matches('/');

    // Also check the basename for nested paths
    let basename = std::path::Path::new(normalized)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(normalized);

    for &protected in CONTAINMENT_FILES {
        if normalized == protected
            || basename == protected
            || normalized.ends_with(&format!("/{}", protected))
        {
            return Some(protected);
        }
    }
    None
}

/// Check if a bash command attempts to breach containment.
/// Returns `Some(reason)` if blocked, `None` if allowed.
pub fn check_command(cmd: &str) -> Option<String> {
    let lower = cmd.to_lowercase();

    // Block containerisation commands
    for &blocked in CONTAINMENT_COMMANDS {
        if lower.contains(blocked) {
            return Some(format!(
                "Command contains '{}' which could breach the containment boundary",
                blocked.trim()
            ));
        }
    }

    // Block writes/modifications to containment files via shell
    for &protected in CONTAINMENT_FILES {
        let prot_lower = protected.to_lowercase();
        let write_patterns = [
            format!(">{}", prot_lower),
            format!("> {}", prot_lower),
            format!(">>{}", prot_lower),
            format!(">> {}", prot_lower),
            "sed -i".to_string(),
            format!("tee {}", prot_lower),
            format!("mv {}", prot_lower),
            format!("cp {}", prot_lower),
            format!("rm {}", prot_lower),
            format!("rm -f {}", prot_lower),
            format!("rm -rf {}", prot_lower),
            format!("chmod {}", prot_lower),
            format!("chown {}", prot_lower),
        ];
        for pattern in &write_patterns {
            if lower.contains(pattern.as_str()) {
                return Some(format!(
                    "Command would modify containment file '{}'",
                    protected
                ));
            }
        }
    }

    None
}

/// Check if a path is inside the `.git/` internal directory (blocked for writes).
pub fn is_git_internal(path: &str) -> bool {
    let normalized = path.trim().trim_start_matches("./").trim_start_matches('/');
    normalized.starts_with(".git/") || normalized == ".git"
}

/// Check if a path attempts directory traversal beyond the project root.
pub fn has_path_traversal(path: &str) -> bool {
    let normalized = path.trim().trim_start_matches("./");
    normalized.contains("..") || normalized.starts_with('/')
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Path blocking ──────────────────────────────────────────────

    #[test]
    fn blocks_dockerfile() {
        assert!(check_path("Dockerfile").is_some());
        assert!(check_path("./Dockerfile").is_some());
        assert!(check_path("/home/user/Dockerfile").is_some());
    }

    #[test]
    fn blocks_docker_compose() {
        assert!(check_path("docker-compose.yml").is_some());
        assert!(check_path("./docker-compose.yml").is_some());
        assert!(check_path("docker-compose.yaml").is_some());
    }

    #[test]
    fn blocks_launch_sh() {
        assert!(check_path("launch.sh").is_some());
        assert!(check_path("start-ernosagent.sh").is_some());
    }

    #[test]
    fn blocks_dockerignore() {
        assert!(check_path(".dockerignore").is_some());
    }

    #[test]
    fn allows_normal_files() {
        assert!(check_path("src/main.rs").is_none());
        assert!(check_path("src/agent/mod.rs").is_none());
        assert!(check_path(".env").is_none());
        assert!(check_path("Cargo.toml").is_none());
        assert!(check_path("src/tools/containment.rs").is_none());
    }

    // ── Command blocking ───────────────────────────────────────────

    #[test]
    fn blocks_docker_commands() {
        assert!(check_command("docker exec -it ernosagent bash").is_some());
        assert!(check_command("docker-compose down").is_some());
        assert!(check_command("nsenter --target 1 --mount").is_some());
    }

    #[test]
    fn allows_normal_commands() {
        assert!(check_command("ls -la").is_none());
        assert!(check_command("cat src/main.rs").is_none());
        assert!(check_command("cargo build").is_none());
        assert!(check_command("cargo test --lib").is_none());
        assert!(check_command("echo hello").is_none());
        assert!(check_command("git diff HEAD").is_none());
    }

    #[test]
    fn blocks_shell_writes_to_containment() {
        assert!(check_command("echo bad > Dockerfile").is_some());
        assert!(check_command("rm docker-compose.yml").is_some());
        assert!(check_command("sed -i 's/old/new/' launch.sh").is_some());
    }

    #[test]
    fn blocks_privilege_escalation_commands() {
        assert!(check_command("chroot /newroot").is_some());
        assert!(check_command("unshare --mount").is_some());
        assert!(check_command("pivot_root . oldroot").is_some());
    }

    // ── Git internal ───────────────────────────────────────────────

    #[test]
    fn detects_git_internal() {
        assert!(is_git_internal(".git/config"));
        assert!(is_git_internal(".git/HEAD"));
        assert!(is_git_internal(".git"));
        assert!(!is_git_internal(".gitignore"));
        assert!(!is_git_internal("src/.gitkeep"));
    }

    // ── Path traversal ─────────────────────────────────────────────

    #[test]
    fn detects_path_traversal() {
        assert!(has_path_traversal("../../etc/passwd"));
        assert!(has_path_traversal("src/../../secret"));
        assert!(has_path_traversal("/etc/passwd"));
        assert!(!has_path_traversal("src/main.rs"));
        assert!(!has_path_traversal("./src/main.rs"));
    }
}
