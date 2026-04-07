//! Input handler — text input widget with cursor and history.

use crate::ui::theme::Theme;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

/// Text input state.
pub struct InputState {
    pub text: String,
    pub cursor: usize,
    pub history: Vec<String>,
    pub history_index: Option<usize>,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            history: Vec::new(),
            history_index: None,
        }
    }

    pub fn insert_char(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            let prev = self.text[..self.cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
            self.text.remove(prev);
            self.cursor = prev;
        }
    }

    pub fn delete(&mut self) {
        if self.cursor < self.text.len() {
            self.text.remove(self.cursor);
        }
    }

    pub fn move_left(&mut self) {
        if self.cursor > 0 {
            self.cursor = self.text[..self.cursor]
                .char_indices()
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
        }
    }

    pub fn move_right(&mut self) {
        if self.cursor < self.text.len() {
            self.cursor += self.text[self.cursor..].chars().next().map(|c| c.len_utf8()).unwrap_or(0);
        }
    }

    pub fn home(&mut self) {
        self.cursor = 0;
    }

    pub fn end(&mut self) {
        self.cursor = self.text.len();
    }

    pub fn submit(&mut self) -> Option<String> {
        if self.text.trim().is_empty() {
            return None;
        }

        let text = self.text.clone();
        self.history.push(text.clone());
        self.text.clear();
        self.cursor = 0;
        self.history_index = None;
        Some(text)
    }

    pub fn history_up(&mut self) {
        if self.history.is_empty() { return; }
        let new_idx = match self.history_index {
            Some(0) => return,
            Some(idx) => idx - 1,
            None => self.history.len() - 1,
        };
        self.history_index = Some(new_idx);
        self.text = self.history[new_idx].clone();
        self.cursor = self.text.len();
    }

    pub fn history_down(&mut self) {
        match self.history_index {
            Some(idx) if idx < self.history.len() - 1 => {
                self.history_index = Some(idx + 1);
                self.text = self.history[idx + 1].clone();
                self.cursor = self.text.len();
            }
            Some(_) => {
                self.history_index = None;
                self.text.clear();
                self.cursor = 0;
            }
            None => {}
        }
    }
}

/// Render the input widget.
pub fn render_input(frame: &mut Frame, area: Rect, state: &InputState, focused: bool) {
    let border_style = if focused { Theme::border_focused() } else { Theme::border() };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(border_style)
        .title(Span::styled(" Message ", Theme::title()))
        .style(Theme::input());

    let display_text = if state.text.is_empty() && !focused {
        "Type a message... (Enter to send)".to_string()
    } else {
        state.text.clone()
    };

    let style = if state.text.is_empty() && !focused {
        Theme::muted()
    } else {
        ratatui::style::Style::default().fg(Theme::FG)
    };

    let paragraph = Paragraph::new(Line::from(Span::styled(display_text, style)))
        .block(block);

    frame.render_widget(paragraph, area);

    // Show cursor
    if focused {
        let cursor_x = area.x + 1 + state.cursor as u16;
        let cursor_y = area.y + 1;
        if cursor_x < area.x + area.width - 1 {
            frame.set_cursor_position((cursor_x, cursor_y));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_insert_and_backspace() {
        let mut state = InputState::new();
        state.insert_char('H');
        state.insert_char('i');
        assert_eq!(state.text, "Hi");
        assert_eq!(state.cursor, 2);

        state.backspace();
        assert_eq!(state.text, "H");
        assert_eq!(state.cursor, 1);
    }

    #[test]
    fn test_input_cursor_movement() {
        let mut state = InputState::new();
        state.text = "Hello".to_string();
        state.cursor = 3;

        state.move_left();
        assert_eq!(state.cursor, 2);

        state.move_right();
        assert_eq!(state.cursor, 3);

        state.home();
        assert_eq!(state.cursor, 0);

        state.end();
        assert_eq!(state.cursor, 5);
    }

    #[test]
    fn test_input_submit() {
        let mut state = InputState::new();
        state.text = "Hello".to_string();
        state.cursor = 5;

        let result = state.submit();
        assert_eq!(result, Some("Hello".to_string()));
        assert!(state.text.is_empty());
        assert_eq!(state.cursor, 0);
    }

    #[test]
    fn test_input_submit_empty() {
        let mut state = InputState::new();
        assert!(state.submit().is_none());
    }

    #[test]
    fn test_input_history() {
        let mut state = InputState::new();
        state.text = "first".to_string();
        state.submit();
        state.text = "second".to_string();
        state.submit();

        state.history_up();
        assert_eq!(state.text, "second");

        state.history_up();
        assert_eq!(state.text, "first");

        state.history_down();
        assert_eq!(state.text, "second");
    }

    #[test]
    fn test_input_delete() {
        let mut state = InputState::new();
        state.text = "Hello".to_string();
        state.cursor = 2;
        state.delete();
        assert_eq!(state.text, "Helo");
    }

    #[test]
    fn test_input_backspace_at_start() {
        let mut state = InputState::new();
        state.text = "Hi".to_string();
        state.cursor = 0;
        state.backspace(); // should be a no-op
        assert_eq!(state.text, "Hi");
    }
}
