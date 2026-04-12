// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.

//! Multi-instance end-to-end mesh tests.
//!
//! Simulates multiple ErnOS mesh nodes in a single process using separate
//! MeshCoordinator instances on distinct ports and tmp directories.
//! Tests: peer discovery, trust attestation, message routing, compute offload,
//! weight exchange, governance, and content filter enforcement.

#[cfg(feature = "mesh")]
mod mesh_e2e {
    use ernosagent::network::compute::{ComputeJob, JobStatus, RelaySlot};
    use ernosagent::network::mesh_loop::{MeshConfig, MeshCoordinator};
    use ernosagent::network::peer_id::PeerId;
    use ernosagent::network::web_proxy::ProxyResponse;
    use ernosagent::network::wire::{Attestation, LessonPayload};
    use std::path::PathBuf;

    fn temp_dir(name: &str) -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir()
            .join(format!("ernos_mesh_e2e_{}_{}_{}", name, std::process::id(), n));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn sim_config() -> MeshConfig {
        MeshConfig {
            enabled: false,
            simulation_mode: true,
            port: 0,
            ..Default::default()
        }
    }

    // ── Test 1: Two nodes initialise with distinct identities ──────

    #[tokio::test]
    async fn test_two_nodes_distinct_identity() {
        let dir_a = temp_dir("e2e_id_a");
        let dir_b = temp_dir("e2e_id_b");

        let node_a = MeshCoordinator::init(&dir_a, sim_config()).await.unwrap();
        let node_b = MeshCoordinator::init(&dir_b, sim_config()).await.unwrap();

        // PeerId is derived from binary hash so it's the same for both — 
        // verify that the coordinators have independent state
        assert_eq!(node_a.peer_id(), node_b.peer_id(),
            "Same binary should produce same peer ID");
        // But their data dirs must be independent  
        assert_ne!(
            format!("{:?}", node_a.identity),
            format!("{:?}", node_b.identity),
            "Two nodes should have distinct identity objects (different timestamps)"
        );
    }

    // ── Test 2: Cross-node trust attestation ──────────────────────

    #[tokio::test]
    async fn test_cross_node_attestation() {
        let dir_a = temp_dir("e2e_trust_a");
        let dir_b = temp_dir("e2e_trust_b");

        let mut node_a = MeshCoordinator::init(&dir_a, sim_config()).await.unwrap();
        let mut node_b = MeshCoordinator::init(&dir_b, sim_config()).await.unwrap();

        let attestation_a = Attestation {
            binary_hash: node_a.watchdog.binary_hash().to_string(),
            commit: "v1.0.0-test".to_string(),
            built_at: chrono::Utc::now().to_rfc3339(),
            source_hash: "test_hash_a".to_string(),
        };
        let attestation_b = Attestation {
            binary_hash: node_b.watchdog.binary_hash().to_string(),
            commit: "v1.0.0-test".to_string(),
            built_at: chrono::Utc::now().to_rfc3339(),
            source_hash: "test_hash_b".to_string(),
        };

        // Node A attests Node B
        node_a.trust_gate.attest(&node_b.peer_id(), attestation_b);
        assert_eq!(
            node_a.trust_gate.trust_level(&node_b.peer_id()),
            ernosagent::network::trust::TrustLevel::Attested
        );

        // Node B attests Node A
        node_b.trust_gate.attest(&node_a.peer_id(), attestation_a);
        assert_eq!(
            node_b.trust_gate.trust_level(&node_a.peer_id()),
            ernosagent::network::trust::TrustLevel::Attested
        );
    }

    // ── Test 3: Compute job submission with equality ──────────────

    #[tokio::test]
    async fn test_compute_job_cross_node() {
        let dir_a = temp_dir("e2e_compute_a");
        let dir_b = temp_dir("e2e_compute_b");

        let node_a = MeshCoordinator::init(&dir_a, sim_config()).await.unwrap();
        let node_b = MeshCoordinator::init(&dir_b, sim_config()).await.unwrap();

        // Node B registers as relay (contributes)
        node_b.compute_pool.register_relay(RelaySlot {
            peer_id: node_b.peer_id(),
            available: true,
            bandwidth_kbps: 5000,
            requests_served: 0,
        }).await;

        // Node A tries to submit — below threshold but lenient default (-5)
        let job = ComputeJob {
            job_id: "e2e-job-1".into(),
            model: "test-model".into(),
            prompt: "Hello from node A".into(),
            max_tokens: 100,
            requester: node_a.peer_id(),
            assigned_to: None,
            status: JobStatus::Queued,
            result: None,
            submitted_at: chrono::Utc::now().to_rfc3339(),
            completed_at: None,
        };

        let result = node_a.compute_pool.submit_job(job).await;
        assert!(result.is_ok(), "Should accept with default -5 equality threshold");

        let stats = node_a.compute_pool.job_stats().await;
        assert_eq!(stats.0, 1, "Should have 1 queued job");
    }

    // ── Test 4: Knowledge export with PII stripping ──────────────

    #[tokio::test]
    async fn test_knowledge_export_e2e() {
        let dir_a = temp_dir("e2e_knowledge_a");
        let dir_b = temp_dir("e2e_knowledge_b");

        let node_a = MeshCoordinator::init(&dir_a, sim_config()).await.unwrap();
        let mut node_b = MeshCoordinator::init(&dir_b, sim_config()).await.unwrap();

        // Node A creates a lesson
        let lesson = LessonPayload {
            id: "lesson-1".into(),
            text: "Always validate user input before processing".into(),
            keywords: vec!["validation".into(), "security".into()],
            confidence: 0.9,
            origin: "node_a_local".into(),
            learned_at: chrono::Utc::now().to_rfc3339(),
        };

        // Export from A (PII stripped, anonymised)
        let exported = node_a.knowledge_sync.prepare_for_export(lesson).unwrap();
        assert_eq!(exported.origin, "mesh", "Origin should be anonymised");
        assert!(exported.confidence <= 0.8, "Confidence should be capped");

        // Import into B via process_inbound
        let imported = node_b.knowledge_sync.process_inbound(&exported);
        assert!(imported.is_ok(), "Import should succeed");
    }

    // ── Test 5: Content filter blocks malicious content ───────────

    #[tokio::test]
    async fn test_content_filter_cross_node() {
        let dir_a = temp_dir("e2e_filter_a");

        let mut node_a = MeshCoordinator::init(&dir_a, sim_config()).await.unwrap();
        let attacker = PeerId("malicious_peer".into());

        // Simulate receiving malicious content from another node
        let scan = node_a.content_filter.scan(&attacker, "<script>alert('xss')</script>");
        assert!(scan.is_blocked(), "XSS should be blocked");

        // Should be tracked
        let (scanned, blocked, _) = node_a.content_filter.stats();
        assert!(scanned > 0);
        assert!(blocked > 0);
    }

    // ── Test 6: DHT store and lookup ──────────────────────────────

    #[tokio::test]
    async fn test_dht_store_lookup() {
        let dir_a = temp_dir("e2e_dht_a");
        let dir_b = temp_dir("e2e_dht_b");

        let mut node_a = MeshCoordinator::init(&dir_a, sim_config()).await.unwrap();
        let mut node_b = MeshCoordinator::init(&dir_b, sim_config()).await.unwrap();

        // Node A stores data
        let data = b"shared mesh data";
        let key = node_a.dht.store(data, "test", 3600, node_a.peer_id());

        // Simulate network transfer: look up from A and store in B
        let entry = node_a.dht.lookup(&key).unwrap();
        node_b.dht.store(
            &entry.value,
            &entry.entry_type,
            3600,
            entry.origin.clone(),
        );

        // Node B can now look it up (same content key)
        let recovered = node_b.dht.lookup(&key);
        assert!(recovered.is_some());
        assert_eq!(recovered.unwrap().value, data);
    }

    // ── Test 7: Governance phase progression ──────────────────────

    #[tokio::test]
    async fn test_governance_progression() {
        let dir = temp_dir("e2e_governance");
        let mut node = MeshCoordinator::init(&dir, sim_config()).await.unwrap();

        // Start in Seed phase
        use ernosagent::network::governance::GovernancePhase;
        assert_eq!(*node.governance.phase(), GovernancePhase::Seed);

        // Simulate network growth
        node.governance.update_peer_count(12);
        assert_eq!(*node.governance.phase(), GovernancePhase::Growing);

        node.governance.update_peer_count(55);
        assert_eq!(*node.governance.phase(), GovernancePhase::Mature);
    }

    // ── Test 8: Full status snapshot ──────────────────────────────

    #[tokio::test]
    async fn test_full_status_snapshot() {
        let dir = temp_dir("e2e_status");
        let node = MeshCoordinator::init(&dir, sim_config()).await.unwrap();
        let status = node.status().await;

        assert!(!status.peer_id.is_empty());
        assert!(!status.display_name.is_empty());
        assert_eq!(status.connected_peers, 0);
        assert!(!status.governance_phase.is_empty());
    }

    // ── Test 9: Web proxy cache round-trip ───────────────────────

    #[tokio::test]
    async fn test_web_proxy_caching() {
        let dir = temp_dir("e2e_proxy");
        let mut node = MeshCoordinator::init(&dir, sim_config()).await.unwrap();

        // First request — cache miss
        let result1 = node.web_proxy.check_cache("https://example.com");
        assert!(result1.is_none());

        // Cache a response
        node.web_proxy.cache_response(
            "https://example.com",
            ProxyResponse {
                request_id: "r1".into(),
                status_code: 200,
                content_type: "text/html".into(),
                body: b"<html>Hello</html>".to_vec(),
                provider: PeerId("relay".into()),
                cached: false,
            },
        );

        // Second request — cache hit
        let result2 = node.web_proxy.check_cache("https://example.com");
        assert!(result2.is_some());
        assert_eq!(result2.unwrap().status_code, 200);
    }

    // ── Test 10: Self-destruct prevents reboot ───────────────────

    #[tokio::test]
    async fn test_self_destruct_blocks_reboot() {
        let dir = temp_dir("e2e_destruct");
        std::fs::create_dir_all(&dir).unwrap();

        // Init normally
        let node = MeshCoordinator::init(&dir, sim_config()).await.unwrap();
        let _peer_id = node.peer_id();
        drop(node);

        // Simulate self-destruct
        std::fs::write(dir.join("destruct.log"), "self-destructed").unwrap();

        // Attempt reboot — should fail
        let result = MeshCoordinator::init(&dir, sim_config()).await;
        match result {
            Err(e) => assert!(e.to_string().contains("self-destruct")),
            Ok(_) => panic!("Should have failed with self-destruct error"),
        }
    }

    // ── Test 11: MeshFS chunk and reassemble ─────────────────────

    #[tokio::test]
    async fn test_meshfs_e2e() {
        let dir = temp_dir("e2e_meshfs");
        let mut node = MeshCoordinator::init(&dir, sim_config()).await.unwrap();

        // Generate > 256KB to get multiple chunks (default chunk size is 256KB)
        let chunk = b"The quick brown fox jumps over the lazy dog. ";
        let data: Vec<u8> = chunk.repeat(6000); // ~270KB
        let manifest = node.mesh_fs.chunk_file(&data, Some("test.txt")).unwrap();

        assert!(manifest.chunk_hashes.len() > 1, 
            "Should chunk into multiple pieces, got {} chunks for {} bytes", 
            manifest.chunk_hashes.len(), data.len());

        let reassembled = node.mesh_fs.reassemble(&manifest).unwrap().unwrap();
        assert_eq!(reassembled, data);
    }

    // ── Test 12: Weight exchange versioning ──────────────────────

    #[tokio::test]
    async fn test_weight_exchange_e2e() {
        let dir = temp_dir("e2e_weights");
        let mut node = MeshCoordinator::init(&dir, sim_config()).await.unwrap();

        // Record an adapter version announcement from a peer
        node.weight_exchange.record_announcement(
            "v1_test".to_string(),
            "{}".to_string(),
            PeerId("origin_peer".into()),
        );

        let versions = node.weight_exchange.known_versions();
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0].version, "v1_test");
    }
}
