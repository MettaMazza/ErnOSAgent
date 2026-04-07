//! Observer / Skeptic audit system — quality gate before response delivery.

pub mod audit;
pub mod parser;
pub mod rules;

pub use audit::{AuditResult, Verdict};
