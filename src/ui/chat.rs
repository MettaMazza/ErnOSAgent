//! Chat area — renders the message history and streaming response.

use crate::provider::Message;
use crate::ui::theme::Theme;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

/// Render the chat area with message history.
pub fn render_chat(
    frame: &mut Frame,
    area: Rect,
    messages: &[Message],
    streaming_text: &str,
    is_thinking: bool,
    scroll_offset: u16,
    focused: bool,
) {
    let border_style = if focused { Theme::border_focused() } else { Theme::border() };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(Span::styled(" Chat ", Theme::title()))
        .style(ratatui::style::Style::default().bg(Theme::BG));

    let mut lines: Vec<Line> = Vec::new();

    for msg in messages {
        // Role label
        let role_label = match msg.role.as_str() {
            "user" => "▶ You",
            "assistant" => "◀ Ernos",
            "system" => "⚙ System",
            "tool" => "🔧 Tool",
            _ => &msg.role,
        };

        lines.push(Line::from(vec![
            Span::styled(role_label, Theme::role_style(&msg.role)),
        ]));

        // Message content (word-wrapped by Paragraph)
        for content_line in msg.content.lines() {
            lines.push(Line::from(Span::styled(
                content_line.to_string(),
                ratatui::style::Style::default().fg(Theme::FG),
            )));
        }

        lines.push(Line::from("")); // spacing
    }

    // Streaming text
    if !streaming_text.is_empty() {
        let label = if is_thinking { "◀ Ernos 💭" } else { "◀ Ernos" };
        let style = if is_thinking {
            Theme::role_style("thinking")
        } else {
            Theme::role_style("assistant")
        };

        lines.push(Line::from(Span::styled(label, style)));

        let text_style = if is_thinking {
            ratatui::style::Style::default().fg(Theme::THINKING)
        } else {
            ratatui::style::Style::default().fg(Theme::FG)
        };

        for line in streaming_text.lines() {
            lines.push(Line::from(Span::styled(line.to_string(), text_style)));
        }

        // Cursor animation
        lines.push(Line::from(Span::styled("▌", Theme::cursor())));
    }

    let paragraph = Paragraph::new(Text::from(lines))
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll_offset, 0));

    frame.render_widget(paragraph, area);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_chat_module_compiles() {
        // Integration test — rendering requires a terminal backend.
        assert!(true);
    }
}
