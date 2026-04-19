// Ern-OS — Shared application state — thread-safe, scalable via Arc+RwLock.

use crate::agents::AgentRegistry;
use crate::agents::teams::TeamRegistry;
use crate::config::AppConfig;
use crate::learning::buffers::GoldenBuffer;
use crate::learning::buffers_rejection::RejectionBuffer;
use crate::memory::MemoryManager;
use crate::model::ModelSpec;
use crate::provider::Provider;
use crate::scheduler::store::JobStore;
use crate::session::SessionManager;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::tools::browser_tool::BrowserState;
use crate::platform::registry::PlatformRegistry;

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
    /// Post-recompile resume message — consumed by the first WebSocket client that connects.
    pub resume_message: Arc<RwLock<Option<String>>>,
}
