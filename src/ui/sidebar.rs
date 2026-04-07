//! Session sidebar — list and switch sessions.

use crate::session::manager::SessionSummary;
use crate::ui::theme::Theme;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};
use ratatui::Frame;

pub fn render_sidebar(
    frame: &mut Frame,
    area: Rect,
    sessions: &[SessionSummary],
    active_id: &str,
    selected_index: usize,
    focused: bool,
) {
    let border_style = if focused { Theme::border_focused() } else { Theme::border() };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(Span::styled(" Sessions ", Theme::title()))
        .style(ratatui::style::Style::default().bg(Theme::BG_SURFACE));

    let items: Vec<ListItem> = sessions
        .iter()
        .map(|s| {
            let is_active = s.id == active_id;
            let prefix = if is_active { "● " } else { "  " };
            let title = if s.title.len() > 22 {
                format!("{}...", &s.title[..22])
            } else {
                s.title.clone()
            };

            let style = if is_active {
                ratatui::style::Style::default().fg(Theme::ACCENT)
            } else {
                Theme::dim()
            };

            ListItem::new(Line::from(Span::styled(format!("{}{}", prefix, title), style)))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(selected_index));

    let list = List::new(items)
        .block(block)
        .highlight_style(Theme::selected());

    frame.render_stateful_widget(list, area, &mut state);
}
