// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.

//! Mesh network API routes — exposed at `/api/mesh/*`.
//!
//! Provides JSON endpoints for the web UI dashboard to display:
//! - Mesh status, peer list, trust levels
//! - Compute job status, contribution leaderboard
//! - DHT stats, content filter stats
//! - Governance status, alerts

use crate::web::state::SharedState;
use axum::extract::State;
use axum::Json;
use serde::Serialize;

/// Full mesh status for the dashboard.
#[derive(Serialize)]
pub struct MeshStatusResponse {
    pub enabled: bool,
    pub peer_id: String,
    pub display_name: String,
    pub connected_peers: usize,
    pub known_peers: usize,
    pub governance_phase: String,
    pub trust_unattested: usize,
    pub trust_attested: usize,
    pub trust_full: usize,
    pub quarantined: usize,
    pub jobs_queued: usize,
    pub jobs_in_progress: usize,
    pub jobs_completed: usize,
    pub jobs_failed: usize,
    pub relay_count: usize,
    pub content_scanned: u64,
    pub content_blocked: u64,
    pub content_peers: usize,
    pub dht_entries: usize,
    pub integrity_valid: bool,
}

/// GET /api/mesh/status — Full mesh status snapshot.
#[cfg(feature = "mesh")]
pub async fn mesh_status(State(state): State<SharedState>) -> Json<MeshStatusResponse> {
    let state = state.read().await;
    if let Some(ref coord_lock) = state.mesh_coordinator {
        let coord = coord_lock.read().await;
        let status = coord.status().await;
        Json(MeshStatusResponse {
            enabled: status.enabled,
            peer_id: status.peer_id,
            display_name: status.display_name,
            connected_peers: status.connected_peers,
            known_peers: status.known_peers,
            governance_phase: status.governance_phase,
            trust_unattested: status.trust_summary.0,
            trust_attested: status.trust_summary.1,
            trust_full: status.trust_summary.2,
            quarantined: status.quarantined_count,
            jobs_queued: status.compute_jobs.0,
            jobs_in_progress: status.compute_jobs.1,
            jobs_completed: status.compute_jobs.2,
            jobs_failed: status.compute_jobs.3,
            relay_count: status.relay_count,
            content_scanned: status.content_stats.0,
            content_blocked: status.content_stats.1,
            content_peers: status.content_stats.2,
            dht_entries: status.dht_entries,
            integrity_valid: status.integrity_valid,
        })
    } else {
        Json(MeshStatusResponse {
            enabled: false,
            peer_id: String::new(),
            display_name: String::new(),
            connected_peers: 0,
            known_peers: 0,
            governance_phase: "N/A".into(),
            trust_unattested: 0,
            trust_attested: 0,
            trust_full: 0,
            quarantined: 0,
            jobs_queued: 0,
            jobs_in_progress: 0,
            jobs_completed: 0,
            jobs_failed: 0,
            relay_count: 0,
            content_scanned: 0,
            content_blocked: 0,
            content_peers: 0,
            dht_entries: 0,
            integrity_valid: false,
        })
    }
}

/// Fallback when mesh feature is not compiled in.
#[cfg(not(feature = "mesh"))]
pub async fn mesh_status(State(_state): State<SharedState>) -> (StatusCode, Json<MeshStatusResponse>) {
    (StatusCode::NOT_FOUND, Json(MeshStatusResponse {
        enabled: false,
        peer_id: String::new(),
        display_name: String::new(),
        connected_peers: 0,
        known_peers: 0,
        governance_phase: "not_compiled".into(),
        trust_unattested: 0,
        trust_attested: 0,
        trust_full: 0,
        quarantined: 0,
        jobs_queued: 0,
        jobs_in_progress: 0,
        jobs_completed: 0,
        jobs_failed: 0,
        relay_count: 0,
        content_scanned: 0,
        content_blocked: 0,
        content_peers: 0,
        dht_entries: 0,
        integrity_valid: false,
    }))
}

/// Individual peer detail for the mesh peer list.
#[derive(Serialize, Clone)]
pub struct MeshPeerDto {
    pub peer_id: String,
    pub display_name: String,
    pub trust_level: String,
    pub latency_ms: Option<u64>,
    pub last_seen: String,
    pub connected: bool,
}

/// GET /api/mesh/peers — individual peer details.
#[cfg(feature = "mesh")]
pub async fn mesh_peers(State(state): State<SharedState>) -> Json<Vec<MeshPeerDto>> {
    let state = state.read().await;
    if let Some(ref coord_lock) = state.mesh_coordinator {
        let coord = coord_lock.read().await;
        let peers = coord.peer_list().await;
        Json(peers.into_iter().map(|p| MeshPeerDto {
            peer_id: p.peer_id,
            display_name: p.display_name,
            trust_level: p.trust_level,
            latency_ms: p.latency_ms,
            last_seen: p.last_seen,
            connected: p.connected,
        }).collect())
    } else {
        Json(Vec::new())
    }
}

/// Fallback when mesh feature is not compiled in.
#[cfg(not(feature = "mesh"))]
pub async fn mesh_peers(State(_state): State<SharedState>) -> Json<Vec<MeshPeerDto>> {
    Json(Vec::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mesh_status_response_serialization() {
        let resp = MeshStatusResponse {
            enabled: true,
            peer_id: "abc123".to_string(),
            display_name: "ErnOS-1".to_string(),
            connected_peers: 3,
            known_peers: 7,
            governance_phase: "active".to_string(),
            trust_unattested: 2,
            trust_attested: 3,
            trust_full: 2,
            quarantined: 0,
            jobs_queued: 1,
            jobs_in_progress: 2,
            jobs_completed: 10,
            jobs_failed: 0,
            relay_count: 1,
            content_scanned: 100,
            content_blocked: 3,
            content_peers: 5,
            dht_entries: 42,
            integrity_valid: true,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"enabled\":true"));
        assert!(json.contains("\"connected_peers\":3"));
    }

    #[test]
    fn test_mesh_peer_dto_serialization() {
        let peer = MeshPeerDto {
            peer_id: "peer-001".to_string(),
            display_name: "Node-Alpha".to_string(),
            trust_level: "attested".to_string(),
            latency_ms: Some(42),
            last_seen: "2026-04-12T12:00:00Z".to_string(),
            connected: true,
        };
        let json = serde_json::to_string(&peer).unwrap();
        assert!(json.contains("\"latency_ms\":42"));
        assert!(json.contains("\"trust_level\":\"attested\""));
    }
}
