//! Status bar — bottom bar showing model, provider, context, steering.

use crate::ui::theme::Theme;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

pub fn render_status_bar(
    frame: &mut Frame,
    area: Rect,
    model_line: &str,
    context_pct: f32,
    steering_summary: &str,
    is_generating: bool,
    react_turn: Option<usize>,
) {
    let mut spans = Vec::new();

    // Model
    spans.push(Span::styled(format!(" {} ", model_line), ratatui::style::Style::default().fg(Theme::ACCENT)));
    spans.push(Span::styled(" │ ", Theme::muted()));

    // Context usage
    let ctx_style = if context_pct > 0.8 {
        Theme::status_error()
    } else if context_pct > 0.6 {
        Theme::status_warn()
    } else {
        Theme::status_ok()
    };
    spans.push(Span::styled(format!("CTX {:.0}%", context_pct * 100.0), ctx_style));
    spans.push(Span::styled(" │ ", Theme::muted()));

    // Steering
    spans.push(Span::styled(steering_summary.to_string(), Theme::dim()));

    // Generation indicator
    if is_generating {
        spans.push(Span::styled(" │ ", Theme::muted()));
        let turn_text = react_turn
            .map(|t| format!(" ⚡ Generating (turn {}) ", t))
            .unwrap_or_else(|| " ⚡ Generating ".to_string());
        spans.push(Span::styled(turn_text, ratatui::style::Style::default().fg(Theme::ACCENT_HOT)));
    }

    let paragraph = Paragraph::new(Line::from(spans))
        .style(ratatui::style::Style::default().bg(Theme::BG_ELEVATED));

    frame.render_widget(paragraph, area);
}
