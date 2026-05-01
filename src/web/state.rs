// Ern-OS — Shared application state — thread-safe, scalable via Arc+RwLock.

use crate::agents::AgentRegistry;
use crate::agents::teams::TeamRegistry;
use crate::config::AppConfig;
use crate::learning::buffers::GoldenBuffer;
use crate::learning::buffers_rejection::RejectionBuffer;
use crate::learning::curriculum::CurriculumStore;
use crate::learning::verification::QuarantineBuffer;
use crate::learning::review::ReviewDeck;
use crate::memory::MemoryManager;
use crate::model::ModelSpec;
use crate::provider::Provider;
use crate::scheduler::store::JobStore;
use crate::session::SessionManager;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::tools::browser_tool::BrowserState;
use crate::platform::registry::PlatformRegistry;
use crate::interpretability::sae::SparseAutoencoder;
use crate::interpretability::live::LiveMonitor;
use crate::interpretability::snapshot::SnapshotStore;

/// Shared application state passed to all Axum handlers.
/// Designed for horizontal scaling — all state behind Arc+RwLock.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub model_spec: Arc<ModelSpec>,
    pub memory: Arc<RwLock<MemoryManager>>,
    pub sessions: Arc<RwLock<SessionManager>>,
    pub provider: Arc<dyn Provider>,
    pub golden_buffer: Arc<RwLock<GoldenBuffer>>,
    pub rejection_buffer: Arc<RwLock<RejectionBuffer>>,
    pub scheduler: Arc<RwLock<JobStore>>,
    pub agents: Arc<RwLock<AgentRegistry>>,
    pub teams: Arc<RwLock<TeamRegistry>>,
    pub browser: Arc<RwLock<BrowserState>>,
    pub platforms: Arc<RwLock<PlatformRegistry>>,
    /// Mutable config — for runtime updates from the Settings UI.
    /// Platform tokens, admin IDs, etc. are updated here and persisted to ern-os.toml.
    pub mutable_config: Arc<RwLock<AppConfig>>,
    /// Post-recompile resume message — (message, session_id, platform).
    /// Consumed by the appropriate delivery path based on platform.
    pub resume_message: Arc<RwLock<Option<(String, String, String)>>>,
    /// SAE for interpretability — loaded lazily from models/sae/
    pub sae: Arc<RwLock<Option<SparseAutoencoder>>>,
    /// Live SAE feature activation monitor — rolling window of recent activations.
    pub live_monitor: Arc<RwLock<LiveMonitor>>,
    /// Neural state snapshots — persisted to data/snapshots/
    pub snapshot_store: Arc<RwLock<SnapshotStore>>,
    /// Inference cancellation flag — shared across all platforms (WebUI, Discord).
    /// Set to true to abort a running inference stream. Reset to false before
    /// each new inference call. Uses AtomicBool instead of CancellationToken
    /// because the flag must be resettable (CancellationToken is single-use).
    pub cancel_flag: Arc<std::sync::atomic::AtomicBool>,
    /// Curriculum store — courses, lessons, and progress for the AI schooling pipeline.
    pub curriculum: Arc<RwLock<CurriculumStore>>,
    /// Quarantine buffer — unverified student answers awaiting external verification.
    pub quarantine: Arc<RwLock<QuarantineBuffer>>,
    /// Review deck — spaced repetition cards for curriculum retention.
    pub review_deck: Arc<RwLock<ReviewDeck>>,
}
