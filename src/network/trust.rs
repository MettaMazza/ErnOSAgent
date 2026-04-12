// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.

//! Trust gate — binary attestation for mesh peers.
//!
//! Every peer starts as `Unattested`. After successful challenge-response
//! verification (SHA-256 binary hash + source hash), they transition to `Attested`.
//! Violations can downgrade trust. Higher trust unlocks more sensitive operations
//! (weight sharing requires `FullTrust`, code sharing requires `FullTrust`).

use crate::network::peer_id::PeerId;
use crate::network::wire::Attestation;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Trust level for a peer — determines what resources they can access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
pub enum TrustLevel {
    /// No attestation received — limited to basic messaging.
    Unattested = 0,
    /// Attestation received and verified — full mesh participation.
    Attested = 1,
    /// Multiple successful attestation cycles + clean history.
    FullTrust = 2,
}

impl std::fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unattested => write!(f, "Unattested"),
            Self::Attested => write!(f, "Attested"),
            Self::FullTrust => write!(f, "FullTrust"),
        }
    }
}

/// Per-peer trust record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustRecord {
    pub peer_id: PeerId,
    pub level: TrustLevel,
    pub attestation: Option<Attestation>,
    pub attestation_count: u32,
    pub violation_count: u32,
    pub last_attested: Option<String>,
    pub last_violation: Option<String>,
}

/// Trust gate — manages attestation and trust levels for all known peers.
pub struct TrustGate {
    records: HashMap<String, TrustRecord>,
    store_path: PathBuf,
    /// Our own attestation for responding to challenges.
    local_attestation: Attestation,
}

impl TrustGate {
    /// Load trust records from disk or start fresh.
    pub fn load(mesh_dir: &Path, local_attestation: Attestation) -> Result<Self> {
        let store_path = mesh_dir.join("trust.json");
        let records = if store_path.exists() {
            let content = std::fs::read_to_string(&store_path)
                .with_context(|| format!("Failed to read trust store from {}", store_path.display()))?;
            serde_json::from_str(&content)
                .with_context(|| "Failed to parse trust store")?
        } else {
            HashMap::new()
        };

        Ok(Self { records, store_path, local_attestation })
    }

    /// Get our local attestation.
    pub fn local_attestation(&self) -> &Attestation {
        &self.local_attestation
    }

    /// Record a successful attestation for a peer.
    pub fn attest(&mut self, peer_id: &PeerId, attestation: Attestation) {
        let record = self.records
            .entry(peer_id.0.clone())
            .or_insert_with(|| TrustRecord {
                peer_id: peer_id.clone(),
                level: TrustLevel::Unattested,
                attestation: None,
                attestation_count: 0,
                violation_count: 0,
                last_attested: None,
                last_violation: None,
            });

        record.attestation = Some(attestation);
        record.attestation_count += 1;
        record.last_attested = Some(chrono::Utc::now().to_rfc3339());

        // Upgrade trust level based on attestation history
        record.level = if record.attestation_count >= 5 && record.violation_count == 0 {
            TrustLevel::FullTrust
        } else {
            TrustLevel::Attested
        };

        tracing::debug!(
            peer = %peer_id,
            level = %record.level,
            count = record.attestation_count,
            "Attestation recorded"
        );
    }

    /// Record a violation, potentially downgrading trust.
    pub fn record_violation(&mut self, peer_id: &PeerId, reason: &str) {
        let record = self.records
            .entry(peer_id.0.clone())
            .or_insert_with(|| TrustRecord {
                peer_id: peer_id.clone(),
                level: TrustLevel::Unattested,
                attestation: None,
                attestation_count: 0,
                violation_count: 0,
                last_attested: None,
                last_violation: None,
            });

        record.violation_count += 1;
        record.last_violation = Some(chrono::Utc::now().to_rfc3339());

        // Downgrade trust on violation
        if record.violation_count >= 3 {
            record.level = TrustLevel::Unattested;
        } else if record.level == TrustLevel::FullTrust {
            record.level = TrustLevel::Attested;
        }

        tracing::warn!(
            peer = %peer_id,
            reason = reason,
            violations = record.violation_count,
            level = %record.level,
            "Trust violation recorded"
        );
    }

    /// Get trust level for a peer.
    pub fn trust_level(&self, peer_id: &PeerId) -> TrustLevel {
        self.records.get(&peer_id.0)
            .map(|r| r.level)
            .unwrap_or(TrustLevel::Unattested)
    }

    /// Check if a peer can receive weight (LoRA) data.
    pub fn can_share_weights(&self, peer_id: &PeerId) -> bool {
        self.trust_level(peer_id) >= TrustLevel::FullTrust
    }

    /// Check if a peer can receive code patches.
    pub fn can_share_code(&self, peer_id: &PeerId) -> bool {
        self.trust_level(peer_id) >= TrustLevel::FullTrust
    }

    /// Check if a peer is at least attested.
    pub fn is_attested(&self, peer_id: &PeerId) -> bool {
        self.trust_level(peer_id) >= TrustLevel::Attested
    }

    /// Get the full trust record for a peer.
    pub fn get_record(&self, peer_id: &PeerId) -> Option<&TrustRecord> {
        self.records.get(&peer_id.0)
    }

    /// Get counts by trust level.
    pub fn trust_summary(&self) -> (usize, usize, usize) {
        let mut unattested = 0;
        let mut attested = 0;
        let mut full_trust = 0;
        for record in self.records.values() {
            match record.level {
                TrustLevel::Unattested => unattested += 1,
                TrustLevel::Attested => attested += 1,
                TrustLevel::FullTrust => full_trust += 1,
            }
        }
        (unattested, attested, full_trust)
    }

    /// Get all known peer IDs.
    pub fn known_peers(&self) -> Vec<PeerId> {
        self.records.values().map(|r| r.peer_id.clone()).collect()
    }

    /// Persist trust records to disk.
    pub fn save(&self) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.records)
            .context("Failed to serialise trust records")?;
        std::fs::write(&self.store_path, json)
            .with_context(|| format!("Failed to write trust store to {}", self.store_path.display()))?;
        Ok(())
    }

    /// Destroy trust records (called during self-destruct).
    pub fn destroy(&self) -> Result<()> {
        if self.store_path.exists() {
            let zeros = vec![0u8; 512];
            std::fs::write(&self.store_path, &zeros).ok();
            std::fs::remove_file(&self.store_path).ok();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_attestation() -> Attestation {
        Attestation {
            binary_hash: "abc123".to_string(),
            commit: "def456".to_string(),
            built_at: "2026-01-01T00:00:00Z".to_string(),
            source_hash: "789abc".to_string(),
        }
    }

    fn temp_dir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir()
            .join(format!("ernos_trust_test_{}_{}", std::process::id(), n));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn test_unknown_peer_is_unattested() {
        let dir = temp_dir();
        let gate = TrustGate::load(&dir, test_attestation()).unwrap();
        let peer = PeerId("unknown".to_string());
        assert_eq!(gate.trust_level(&peer), TrustLevel::Unattested);
        assert!(!gate.is_attested(&peer));
        assert!(!gate.can_share_weights(&peer));
        assert!(!gate.can_share_code(&peer));
    }

    #[test]
    fn test_attestation_upgrades_trust() {
        let dir = temp_dir();
        let mut gate = TrustGate::load(&dir, test_attestation()).unwrap();
        let peer = PeerId("peer_a".to_string());

        gate.attest(&peer, test_attestation());
        assert_eq!(gate.trust_level(&peer), TrustLevel::Attested);
        assert!(gate.is_attested(&peer));
        assert!(!gate.can_share_weights(&peer));
    }

    #[test]
    fn test_full_trust_after_five_attestations() {
        let dir = temp_dir();
        let mut gate = TrustGate::load(&dir, test_attestation()).unwrap();
        let peer = PeerId("trusted".to_string());

        for _ in 0..5 {
            gate.attest(&peer, test_attestation());
        }
        assert_eq!(gate.trust_level(&peer), TrustLevel::FullTrust);
        assert!(gate.can_share_weights(&peer));
        assert!(gate.can_share_code(&peer));
    }

    #[test]
    fn test_violation_downgrades_trust() {
        let dir = temp_dir();
        let mut gate = TrustGate::load(&dir, test_attestation()).unwrap();
        let peer = PeerId("violator".to_string());

        for _ in 0..5 {
            gate.attest(&peer, test_attestation());
        }
        assert_eq!(gate.trust_level(&peer), TrustLevel::FullTrust);

        gate.record_violation(&peer, "test violation");
        assert_eq!(gate.trust_level(&peer), TrustLevel::Attested);
    }

    #[test]
    fn test_three_violations_drop_to_unattested() {
        let dir = temp_dir();
        let mut gate = TrustGate::load(&dir, test_attestation()).unwrap();
        let peer = PeerId("bad_actor".to_string());

        gate.attest(&peer, test_attestation());
        for _ in 0..3 {
            gate.record_violation(&peer, "repeated violation");
        }
        assert_eq!(gate.trust_level(&peer), TrustLevel::Unattested);
    }

    #[test]
    fn test_trust_summary() {
        let dir = temp_dir();
        let mut gate = TrustGate::load(&dir, test_attestation()).unwrap();

        gate.attest(&PeerId("a".into()), test_attestation());
        gate.attest(&PeerId("b".into()), test_attestation());
        // c is unknown = unattested
        gate.record_violation(&PeerId("c".into()), "test");

        let (unattested, attested, full_trust) = gate.trust_summary();
        assert_eq!(attested, 2);
        assert_eq!(unattested, 1);
        assert_eq!(full_trust, 0);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let dir = temp_dir();
        let peer = PeerId("persist_peer".to_string());

        {
            let mut gate = TrustGate::load(&dir, test_attestation()).unwrap();
            gate.attest(&peer, test_attestation());
            gate.save().unwrap();
        }

        {
            let gate = TrustGate::load(&dir, test_attestation()).unwrap();
            assert_eq!(gate.trust_level(&peer), TrustLevel::Attested);
        }
    }

    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::Unattested < TrustLevel::Attested);
        assert!(TrustLevel::Attested < TrustLevel::FullTrust);
    }
}
