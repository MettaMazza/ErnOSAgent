//! Verification pipeline — self-testing loop for validating code changes.
//!
//! After the agent makes code changes, this module:
//! 1. Runs compile checks (build + tests)
//! 2. Optionally validates via headless browser
//! 3. Returns structured results for auto-fix loops

pub mod compiler_check;
#[cfg(feature = "desktop")]
pub mod browser;
pub mod pipeline;
