// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.

//! Mesh network module — distributed P2P infrastructure for ErnOS agents.
//!
//! Provides peer-to-peer connectivity, distributed compute pooling,
//! knowledge exchange, code propagation, internet relay, content-addressed
//! storage, decentralised governance, and human mesh communication.
//!
//! Gated behind the `mesh` Cargo feature flag and the `config.mesh.enabled`
//! runtime toggle. Default: off at both levels.

// Phase 1 — Foundation
pub mod peer_id;
pub mod wire;
pub mod crypto;
pub mod transport;

// Phase 2 — Security Pipeline (Neutralise)
pub mod trust;
pub mod sanctions;
pub mod neutralise;
pub mod content_filter;

// Phase 3 — Discovery, Identity, Capabilities
pub mod discovery;
pub mod identity;
pub mod capabilities;

// Phase 4 — Resource Sharing
pub mod compute;
pub mod knowledge_sync;
pub mod weight_exchange;
pub mod code_propagation;

// Phase 5 — Distributed Storage & Sandbox
pub mod dht;
pub mod mesh_fs;
pub mod sandbox;

// Phase 6 — Governance, Proxy, Human, Event Loop
pub mod governance;
pub mod web_proxy;
pub mod human_mesh;
pub mod mesh_loop;
