//! TUI theme — colour palette, styles, and typography tokens.

use ratatui::style::{Color, Modifier, Style};

pub struct Theme;

impl Theme {
    // ── Base colours ─────────────────────────────────────────────
    pub const BG: Color = Color::Rgb(15, 15, 20);
    pub const BG_SURFACE: Color = Color::Rgb(22, 22, 30);
    pub const BG_ELEVATED: Color = Color::Rgb(30, 30, 42);
    pub const BG_SELECTED: Color = Color::Rgb(38, 38, 55);

    pub const FG: Color = Color::Rgb(220, 220, 230);
    pub const FG_DIM: Color = Color::Rgb(140, 140, 160);
    pub const FG_MUTED: Color = Color::Rgb(80, 80, 100);

    // ── Accent colours ───────────────────────────────────────────
    pub const ACCENT: Color = Color::Rgb(120, 160, 255);   // Blue
    pub const ACCENT_HOT: Color = Color::Rgb(255, 120, 80);  // Orange
    pub const SUCCESS: Color = Color::Rgb(80, 200, 120);
    pub const WARNING: Color = Color::Rgb(255, 200, 60);
    pub const ERROR: Color = Color::Rgb(255, 80, 80);
    pub const INFO: Color = Color::Rgb(100, 180, 255);

    // ── Role colours ─────────────────────────────────────────────
    pub const USER: Color = Color::Rgb(120, 200, 255);
    pub const ASSISTANT: Color = Color::Rgb(180, 140, 255);
    pub const SYSTEM: Color = Color::Rgb(255, 200, 60);
    pub const TOOL: Color = Color::Rgb(80, 200, 120);
    pub const THINKING: Color = Color::Rgb(140, 140, 180);

    // ── Styles ───────────────────────────────────────────────────

    pub fn title() -> Style {
        Style::default().fg(Self::ACCENT).add_modifier(Modifier::BOLD)
    }

    pub fn status_ok() -> Style {
        Style::default().fg(Self::SUCCESS)
    }

    pub fn status_error() -> Style {
        Style::default().fg(Self::ERROR)
    }

    pub fn status_warn() -> Style {
        Style::default().fg(Self::WARNING)
    }

    pub fn selected() -> Style {
        Style::default().bg(Self::BG_SELECTED).fg(Self::FG)
    }

    pub fn dim() -> Style {
        Style::default().fg(Self::FG_DIM)
    }

    pub fn muted() -> Style {
        Style::default().fg(Self::FG_MUTED)
    }

    pub fn role_style(role: &str) -> Style {
        match role {
            "user" => Style::default().fg(Self::USER).add_modifier(Modifier::BOLD),
            "assistant" => Style::default().fg(Self::ASSISTANT).add_modifier(Modifier::BOLD),
            "system" => Style::default().fg(Self::SYSTEM),
            "tool" => Style::default().fg(Self::TOOL),
            _ => Style::default().fg(Self::FG_DIM),
        }
    }

    pub fn border() -> Style {
        Style::default().fg(Self::FG_MUTED)
    }

    pub fn border_focused() -> Style {
        Style::default().fg(Self::ACCENT)
    }

    pub fn input() -> Style {
        Style::default().fg(Self::FG).bg(Self::BG_SURFACE)
    }

    pub fn cursor() -> Style {
        Style::default().fg(Self::BG).bg(Self::ACCENT)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_styles() {
        let _ = Theme::role_style("user");
        let _ = Theme::role_style("assistant");
        let _ = Theme::role_style("system");
        let _ = Theme::role_style("tool");
        let _ = Theme::role_style("unknown");
    }

    #[test]
    fn test_theme_colours_are_distinct() {
        assert_ne!(Theme::BG, Theme::FG);
        assert_ne!(Theme::SUCCESS, Theme::ERROR);
        assert_ne!(Theme::USER, Theme::ASSISTANT);
    }
}
