//! Platform adapter system — Discord, Telegram, WhatsApp.

pub mod adapter;
pub mod registry;
pub mod discord;
pub mod telegram;
pub mod whatsapp;

pub use adapter::PlatformAdapter;
