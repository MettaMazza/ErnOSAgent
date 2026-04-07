//! Platform manager panel — shows connected platforms and controls.

use crate::platform::adapter::PlatformStatus;
use crate::ui::theme::Theme;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem};
use ratatui::Frame;

pub struct PlatformManagerState {
    pub visible: bool,
    pub selected: usize,
}

impl PlatformManagerState {
    pub fn new() -> Self {
        Self { visible: false, selected: 0 }
    }
}

pub fn render_platform_manager(
    frame: &mut Frame,
    area: Rect,
    statuses: &[PlatformStatus],
    state: &PlatformManagerState,
) {
    if !state.visible { return; }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::border_focused())
        .title(Span::styled(" Platforms ", Theme::title()))
        .style(ratatui::style::Style::default().bg(Theme::BG_ELEVATED));

    let items: Vec<ListItem> = statuses.iter()
        .map(|s| {
            let icon = if s.connected { "🟢" } else { "🔴" };
            let error_text = s.error.as_deref().unwrap_or("");
            let line = if error_text.is_empty() {
                format!("{} {}", icon, s.name)
            } else {
                format!("{} {} — {}", icon, s.name, error_text)
            };
            let style = if s.connected { Theme::status_ok() } else { Theme::status_error() };
            ListItem::new(Line::from(Span::styled(line, style)))
        })
        .collect();

    let list = List::new(items).block(block);
    frame.render_widget(list, area);
}
