// Ern-OS — Platform adapters module
// Ported from ErnOSAgent — adapted for WebUI-as-hub architecture.
pub mod adapter;
pub mod registry;
pub mod router;

#[cfg(feature = "discord")]
pub mod discord;
#[cfg(feature = "discord")]
pub mod discord_handler;

#[cfg(feature = "telegram")]
pub mod telegram;
