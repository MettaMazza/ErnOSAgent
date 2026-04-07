//! Model specification types — auto-derived from provider APIs, never hardcoded.

pub mod spec;
pub mod registry;
pub mod router;

pub use spec::{ModelCapabilities, ModelSpec, Modality};
pub use registry::ModelRegistry;
pub use router::ModalityRouter;
