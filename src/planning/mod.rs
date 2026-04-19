//! Planning engine — task decomposition and DAG-based execution.
//!
//! Breaks high-level objectives into structured sub-task graphs,
//! tracks dependencies, and orchestrates execution via sub-agents.

pub mod dag;
pub mod planner;
pub mod executor;
