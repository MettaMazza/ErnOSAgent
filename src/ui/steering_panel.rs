//! Steering panel — visual vector management.

use crate::steering::vectors::SteeringConfig;
use crate::ui::theme::Theme;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState};
use ratatui::Frame;

pub struct SteeringPanelState {
    pub visible: bool,
    pub selected: usize,
}

impl SteeringPanelState {
    pub fn new() -> Self {
        Self { visible: false, selected: 0 }
    }
}

pub fn render_steering_panel(
    frame: &mut Frame,
    area: Rect,
    config: &SteeringConfig,
    state: &SteeringPanelState,
) {
    if !state.visible { return; }

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::border_focused())
        .title(Span::styled(" Steering Vectors ", Theme::title()))
        .style(ratatui::style::Style::default().bg(Theme::BG_ELEVATED));

    let items: Vec<ListItem> = config.vectors.iter()
        .map(|v| {
            let icon = if v.active { "✅" } else { "⬜" };
            let line = format!("{} {} (×{:.1})", icon, v.name, v.scale);
            let style = if v.active {
                ratatui::style::Style::default().fg(Theme::SUCCESS)
            } else {
                Theme::dim()
            };
            ListItem::new(Line::from(Span::styled(line, style)))
        })
        .collect();

    if items.is_empty() {
        let empty = List::new(vec![ListItem::new(Line::from(Span::styled(
            "No vectors found in vectors directory", Theme::muted(),
        )))])
        .block(block);
        frame.render_widget(empty, area);
        return;
    }

    let mut list_state = ListState::default();
    list_state.select(Some(state.selected));

    let list = List::new(items)
        .block(block)
        .highlight_style(Theme::selected());

    frame.render_stateful_widget(list, area, &mut list_state);
}
