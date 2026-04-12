// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.

//! Peer identity profile — display name, status, bio, preferences.
//!
//! Each peer has an identity that persists across sessions.
//! Distinct from `PeerId` (cryptographic address) — this is the
//! human-facing profile.

use crate::network::peer_id::PeerId;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Peer online status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PeerStatus {
    Online,
    Idle,
    DoNotDisturb,
    Invisible,
    Offline,
}

impl std::fmt::Display for PeerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Online => write!(f, "🟢 Online"),
            Self::Idle => write!(f, "🟡 Idle"),
            Self::DoNotDisturb => write!(f, "🔴 DND"),
            Self::Invisible => write!(f, "⚪ Invisible"),
            Self::Offline => write!(f, "⚫ Offline"),
        }
    }
}

/// Per-peer identity profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerIdentity {
    pub peer_id: PeerId,
    pub display_name: String,
    pub status: PeerStatus,
    pub bio: String,
    pub agent_version: String,
    pub os: String,
    pub arch: String,
    pub created_at: String,
    pub last_updated: String,
}

impl PeerIdentity {
    /// Create a new identity with system-detected defaults.
    pub fn new(peer_id: PeerId) -> Self {
        let hostname = hostname::get()
            .map(|h| h.to_string_lossy().to_string())
            .unwrap_or_else(|_| "unknown".to_string());

        Self {
            peer_id,
            display_name: hostname,
            status: PeerStatus::Online,
            bio: String::new(),
            agent_version: env!("CARGO_PKG_VERSION").to_string(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            last_updated: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Load identity from disk or create a new one.
    pub fn load_or_create(mesh_dir: &Path, peer_id: PeerId) -> Result<Self> {
        let path = mesh_dir.join("identity.json");
        if path.exists() {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read identity from {}", path.display()))?;
            let mut identity: Self = serde_json::from_str(&content)
                .with_context(|| "Failed to parse identity")?;
            // Update peer_id in case keys changed
            identity.peer_id = peer_id;
            identity.last_updated = chrono::Utc::now().to_rfc3339();
            Ok(identity)
        } else {
            let identity = Self::new(peer_id);
            identity.save(mesh_dir)?;
            Ok(identity)
        }
    }

    /// Persist identity to disk.
    pub fn save(&self, mesh_dir: &Path) -> Result<()> {
        let path = mesh_dir.join("identity.json");
        std::fs::create_dir_all(mesh_dir)
            .with_context(|| format!("Failed to create mesh dir {}", mesh_dir.display()))?;
        let json = serde_json::to_string_pretty(self)
            .context("Failed to serialise identity")?;
        std::fs::write(&path, json)
            .with_context(|| format!("Failed to write identity to {}", path.display()))?;
        Ok(())
    }

    /// Update display name.
    pub fn set_display_name(&mut self, name: String) {
        self.display_name = name;
        self.last_updated = chrono::Utc::now().to_rfc3339();
    }

    /// Update status.
    pub fn set_status(&mut self, status: PeerStatus) {
        self.status = status;
        self.last_updated = chrono::Utc::now().to_rfc3339();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_dir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir()
            .join(format!("ernos_identity_test_{}_{}", std::process::id(), n));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn test_new_identity() {
        let peer = PeerId("test".to_string());
        let identity = PeerIdentity::new(peer.clone());
        assert_eq!(identity.peer_id, peer);
        assert!(!identity.display_name.is_empty());
        assert_eq!(identity.status, PeerStatus::Online);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = temp_dir();
        let peer = PeerId("persist".to_string());

        {
            let mut identity = PeerIdentity::new(peer.clone());
            identity.set_display_name("TestNode".to_string());
            identity.save(&dir).unwrap();
        }

        {
            let identity = PeerIdentity::load_or_create(&dir, peer.clone()).unwrap();
            assert_eq!(identity.display_name, "TestNode");
        }
    }

    #[test]
    fn test_update_status() {
        let mut identity = PeerIdentity::new(PeerId("test".into()));
        identity.set_status(PeerStatus::DoNotDisturb);
        assert_eq!(identity.status, PeerStatus::DoNotDisturb);
    }

    #[test]
    fn test_status_display() {
        assert!(format!("{}", PeerStatus::Online).contains("Online"));
        assert!(format!("{}", PeerStatus::Offline).contains("Offline"));
    }
}
