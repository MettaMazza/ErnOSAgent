// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Interactive button components for Discord responses.
//!
//! Buttons are attached to the LAST chunk of a response:
//! - 🔊 TTS — plays ALL chunks of the response via Discord TTS
//! - 📋 Copy — sends an ephemeral message with the full stitched text
//! - 👍 Good — positive feedback signal for training (golden example)
//! - 👎 Bad — negative feedback signal for training (preference pair)

use serenity::builder::{CreateActionRow, CreateButton};
use serenity::model::prelude::ButtonStyle;

/// Build the interactive button row for a response message.
///
/// `response_id` is used as a prefix for button custom IDs to route
/// interaction events back to the correct response.
pub fn response_buttons(response_id: &str) -> CreateActionRow {
    CreateActionRow::Buttons(vec![
        CreateButton::new(format!("tts:{response_id}"))
            .emoji('🔊')
            .style(ButtonStyle::Secondary)
            .label("TTS"),
        CreateButton::new(format!("copy:{response_id}"))
            .emoji('📋')
            .style(ButtonStyle::Secondary)
            .label("Copy"),
        CreateButton::new(format!("good:{response_id}"))
            .emoji('👍')
            .style(ButtonStyle::Secondary),
        CreateButton::new(format!("bad:{response_id}"))
            .emoji('👎')
            .style(ButtonStyle::Secondary),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_response_buttons_creates_4_buttons() {
        let row = response_buttons("test_123");
        match row {
            CreateActionRow::Buttons(buttons) => {
                assert_eq!(buttons.len(), 4, "Should create exactly 4 buttons");
            }
            _ => panic!("Expected Buttons action row"),
        }
    }
}
