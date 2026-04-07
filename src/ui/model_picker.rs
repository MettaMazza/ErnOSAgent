//! Model picker — fuzzy-searchable model list popup.

use crate::model::spec::ModelSummary;
use crate::ui::theme::Theme;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState};
use ratatui::Frame;

pub struct ModelPickerState {
    pub visible: bool,
    pub search: String,
    pub selected: usize,
    pub models: Vec<ModelSummary>,
    pub filtered: Vec<usize>,
}

impl ModelPickerState {
    pub fn new() -> Self {
        Self { visible: false, search: String::new(), selected: 0, models: Vec::new(), filtered: Vec::new() }
    }

    pub fn open(&mut self, models: Vec<ModelSummary>) {
        self.models = models;
        self.search.clear();
        self.selected = 0;
        self.filter();
        self.visible = true;
    }

    pub fn close(&mut self) {
        self.visible = false;
    }

    pub fn filter(&mut self) {
        let query = self.search.to_lowercase();
        self.filtered = self.models.iter()
            .enumerate()
            .filter(|(_, m)| query.is_empty() || m.name.to_lowercase().contains(&query))
            .map(|(i, _)| i)
            .collect();
        if self.selected >= self.filtered.len() {
            self.selected = 0;
        }
    }

    pub fn select_up(&mut self) {
        if self.selected > 0 { self.selected -= 1; }
    }

    pub fn select_down(&mut self) {
        if self.selected + 1 < self.filtered.len() { self.selected += 1; }
    }

    pub fn confirm(&mut self) -> Option<&ModelSummary> {
        self.filtered.get(self.selected).map(|&idx| &self.models[idx])
    }
}

pub fn render_model_picker(frame: &mut Frame, area: Rect, state: &ModelPickerState) {
    if !state.visible { return; }

    let popup_w = area.width.min(60);
    let popup_h = area.height.min(20);
    let x = area.x + (area.width - popup_w) / 2;
    let y = area.y + (area.height - popup_h) / 2;
    let popup_area = Rect::new(x, y, popup_w, popup_h);

    frame.render_widget(Clear, popup_area);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Theme::border_focused())
        .title(Span::styled(
            format!(" Model Picker [{}] ", if state.search.is_empty() { "type to search" } else { &state.search }),
            Theme::title(),
        ))
        .style(ratatui::style::Style::default().bg(Theme::BG_ELEVATED));

    let items: Vec<ListItem> = state.filtered.iter()
        .map(|&idx| {
            let m = &state.models[idx];
            ListItem::new(Line::from(Span::raw(m.display_line())))
        })
        .collect();

    let mut list_state = ListState::default();
    list_state.select(Some(state.selected));

    let list = List::new(items)
        .block(block)
        .highlight_style(Theme::selected());

    frame.render_stateful_widget(list, popup_area, &mut list_state);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::spec::ModelCapabilities;

    #[test]
    fn test_model_picker_open_close() {
        let mut state = ModelPickerState::new();
        assert!(!state.visible);
        state.open(vec![]);
        assert!(state.visible);
        state.close();
        assert!(!state.visible);
    }

    #[test]
    fn test_model_picker_filter() {
        let mut state = ModelPickerState::new();
        state.open(vec![
            ModelSummary { name: "gemma4:26b".to_string(), provider: "llamacpp".to_string(),
                parameter_size: "26B".to_string(), quantization_level: "Q4_K_M".to_string(),
                capabilities: ModelCapabilities::default(), context_length: 0 },
            ModelSummary { name: "llama3:8b".to_string(), provider: "ollama".to_string(),
                parameter_size: "8B".to_string(), quantization_level: "Q4_0".to_string(),
                capabilities: ModelCapabilities::default(), context_length: 0 },
        ]);

        assert_eq!(state.filtered.len(), 2);

        state.search = "gemma".to_string();
        state.filter();
        assert_eq!(state.filtered.len(), 1);
    }
}
