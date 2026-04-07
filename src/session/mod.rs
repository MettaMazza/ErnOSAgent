//! Session management — multi-session CRUD and persistence.

pub mod manager;
pub mod store;

pub use manager::SessionManager;
pub use store::Session;
