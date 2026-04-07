//! Application state machine and main TUI event loop.
//!
//! Wires together: config → provider → session → prompt → ReAct → observer → TUI.

use crate::config::AppConfig;
use crate::inference::context;
use crate::model::spec::ModelSpec;
use crate::prompt;
use crate::provider::{self, Message, Provider};
use crate::react::r#loop::{self as react_loop, ReactConfig, ReactEvent};
use crate::react::reply;
use crate::session::manager::SessionManager;
use crate::steering::vectors::SteeringConfig;
use crate::tools::executor::ToolExecutor;
use crate::ui::{
    chat, input, model_picker, platform_manager, sidebar, status_bar, steering_panel,
};
use crate::ui::input::InputState;
use crate::ui::model_picker::ModelPickerState;
use crate::ui::platform_manager::PlatformManagerState;
use crate::ui::steering_panel::SteeringPanelState;
use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::Terminal;
use std::io::stdout;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Which panel has focus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Focus {
    Chat,
    Input,
    Sidebar,
}

/// Application state.
pub struct AppState {
    pub focus: Focus,
    pub input: InputState,
    pub model_picker: ModelPickerState,
    pub steering_panel: SteeringPanelState,
    pub platform_manager: PlatformManagerState,
    pub sidebar_selected: usize,
    pub scroll_offset: u16,
    pub is_generating: bool,
    pub streaming_text: String,
    pub thinking_text: String,
    pub is_thinking: bool,
    pub react_turn: Option<usize>,
    pub model_spec: ModelSpec,
    pub steering_config: SteeringConfig,
    pub context_usage: f32,
    pub status_message: Option<String>,
    pub should_quit: bool,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            focus: Focus::Input,
            input: InputState::new(),
            model_picker: ModelPickerState::new(),
            steering_panel: SteeringPanelState::new(),
            platform_manager: PlatformManagerState::new(),
            sidebar_selected: 0,
            scroll_offset: 0,
            is_generating: false,
            streaming_text: String::new(),
            thinking_text: String::new(),
            is_thinking: false,
            react_turn: None,
            model_spec: ModelSpec::default(),
            steering_config: SteeringConfig::default(),
            context_usage: 0.0,
            status_message: None,
            should_quit: false,
        }
    }
}

/// Main application entry point. Sets up terminal and runs the event loop.
pub async fn run(config: AppConfig) -> Result<()> {
    // ── 1. Initialise provider ───────────────────────────────────────
    let provider: Arc<dyn Provider> = match config.general.active_provider.as_str() {
        "llamacpp" => Arc::new(provider::llamacpp::LlamaCppProvider::new(
            &config.llamacpp,
        )),
        "ollama" => Arc::new(provider::ollama::OllamaProvider::new(&config.ollama)),
        "lmstudio" => Arc::new(provider::lmstudio::LMStudioProvider::new(
            &config.lmstudio,
        )),
        "huggingface" => Arc::new(provider::huggingface::HuggingFaceProvider::new(
            &config.huggingface,
        )),
        other => {
            anyhow::bail!(
                "Unknown provider '{}'. Valid: llamacpp, ollama, lmstudio, huggingface",
                other
            );
        }
    };

    tracing::info!(
        provider = %config.general.active_provider,
        model = %config.general.active_model,
        "Provider initialised"
    );

    // ── 2. Auto-derive model spec ────────────────────────────────────
    let model_spec = match provider.get_model_spec(&config.general.active_model).await {
        Ok(spec) => {
            tracing::info!(
                model = %spec.name,
                context = spec.context_length,
                caps = %spec.capabilities.modality_badges(),
                "Model spec auto-derived"
            );
            spec
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                model = %config.general.active_model,
                "Failed to auto-derive model spec (provider may be offline)"
            );
            ModelSpec {
                name: config.general.active_model.clone(),
                provider: config.general.active_provider.clone(),
                ..Default::default()
            }
        }
    };

    // ── 3. Load prompts ──────────────────────────────────────────────
    let core_prompt = prompt::core::build_core_prompt();
    let identity_prompt = prompt::identity::load_identity(&config.persona_path())
        .unwrap_or_else(|e| {
            tracing::warn!(error = %e, "Failed to load identity prompt");
            String::new()
        });

    // ── 4. Initialise session manager ────────────────────────────────
    let mut session_mgr = SessionManager::new(
        &config.sessions_dir(),
        &config.general.active_model,
        &config.general.active_provider,
    )?;

    // ── 5. Load steering vectors ─────────────────────────────────────
    let mut steering_config = SteeringConfig::default();
    if let Ok(vectors) = SteeringConfig::scan_directory(&config.vectors_dir()) {
        steering_config.vectors = vectors;
    }

    // ── 6. Setup tool executor ───────────────────────────────────────
    let executor = ToolExecutor::new();
    // Tools will be registered as they're implemented (web_search, file_read, etc.)

    // ── 7. Setup terminal ────────────────────────────────────────────
    enable_raw_mode().context("Failed to enable raw mode")?;
    stdout()
        .execute(EnterAlternateScreen)
        .context("Failed to enter alternate screen")?;

    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend).context("Failed to create terminal")?;
    terminal.clear().context("Failed to clear terminal")?;

    let mut state = AppState::new();
    state.model_spec = model_spec.clone();
    state.steering_config = steering_config;

    // Update context usage for initial session
    state.context_usage = context::context_usage(
        &session_mgr.active().messages,
        model_spec.context_length,
    );

    tracing::info!("TUI event loop starting");

    // ── 8. Event channels ────────────────────────────────────────────
    let (react_event_tx, mut react_event_rx) = mpsc::channel::<ReactEvent>(256);
    let (_submit_tx, _submit_rx) = mpsc::channel::<String>(8);

    // ── 9. Main loop ─────────────────────────────────────────────────
    let result = loop {
        // Render
        terminal.draw(|frame| {
            let size = frame.area();

            let h_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(28), Constraint::Min(40)])
                .split(size);

            let sidebar_area = h_chunks[0];
            let main_area = h_chunks[1];

            let v_chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(6),
                    Constraint::Length(3),
                    Constraint::Length(1),
                ])
                .split(main_area);

            let chat_area = v_chunks[0];
            let input_area = v_chunks[1];
            let status_area = v_chunks[2];

            // Sidebar
            sidebar::render_sidebar(
                frame,
                sidebar_area,
                session_mgr.list(),
                session_mgr.active_id(),
                state.sidebar_selected,
                state.focus == Focus::Sidebar,
            );

            // Chat
            chat::render_chat(
                frame,
                chat_area,
                &session_mgr.active().messages,
                &state.streaming_text,
                state.is_thinking,
                state.scroll_offset,
                state.focus == Focus::Chat,
            );

            // Input
            input::render_input(
                frame,
                input_area,
                &state.input,
                state.focus == Focus::Input,
            );

            // Status bar
            let model_line = if state.model_spec.is_derived() {
                state.model_spec.status_line()
            } else {
                format!("{} (connecting...)", config.general.active_model)
            };
            status_bar::render_status_bar(
                frame,
                status_area,
                &model_line,
                state.context_usage,
                &state.steering_config.status_summary(),
                state.is_generating,
                state.react_turn,
            );

            // Overlays
            model_picker::render_model_picker(frame, size, &state.model_picker);
            steering_panel::render_steering_panel(
                frame,
                size,
                &state.steering_config,
                &state.steering_panel,
            );
            platform_manager::render_platform_manager(
                frame,
                size,
                &[],
                &state.platform_manager,
            );
        })?;

        // Drain react events (non-blocking)
        while let Ok(event) = react_event_rx.try_recv() {
            match event {
                ReactEvent::TurnStarted { turn } => {
                    state.react_turn = Some(turn);
                }
                ReactEvent::Token(token) => {
                    state.is_thinking = false;
                    state.streaming_text.push_str(&token);
                }
                ReactEvent::Thinking(token) => {
                    state.is_thinking = true;
                    state.thinking_text.push_str(&token);
                }
                ReactEvent::ToolExecuting { name, .. } => {
                    state.status_message = Some(format!("🔧 Running {}...", name));
                }
                ReactEvent::ToolCompleted { name, result: tr } => {
                    let icon = if tr.success { "✅" } else { "❌" };
                    state.status_message = Some(format!("{} {} done", icon, name));
                }
                ReactEvent::AuditRunning => {
                    state.status_message = Some("🔍 Observer auditing...".to_string());
                }
                ReactEvent::AuditCompleted { verdict, reason } => {
                    state.status_message =
                        Some(format!("Observer: {} ({})", verdict, reason));
                }
                ReactEvent::ResponseReady { text } => {
                    // Add assistant message to session
                    session_mgr.active_mut().add_message(Message {
                        role: "assistant".to_string(),
                        content: text,
                        images: Vec::new(),
                    });
                    session_mgr.active_mut().auto_title();
                    let _ = session_mgr.save_active();
                    let _ = session_mgr.refresh_list();

                    state.is_generating = false;
                    state.streaming_text.clear();
                    state.thinking_text.clear();
                    state.is_thinking = false;
                    state.react_turn = None;
                    state.scroll_offset = 0;
                    state.status_message = None;

                    // Update context usage
                    state.context_usage = context::context_usage(
                        &session_mgr.active().messages,
                        model_spec.context_length,
                    );
                }
                ReactEvent::Error(e) => {
                    state.status_message = Some(format!("Error: {}", e));
                    tracing::error!(error = %e, "ReAct error");
                }
            }
        }

        // Poll keyboard events (16ms → ~60fps)
        if event::poll(std::time::Duration::from_millis(16))? {
            if let Event::Key(key) = event::read()? {
                let submitted = handle_key_event(&mut state, key);

                // Handle message submission
                if let Some(text) = submitted {
                    if !state.is_generating {
                        // Add user message to session
                        session_mgr.active_mut().add_message(Message {
                            role: "user".to_string(),
                            content: text.clone(),
                            images: Vec::new(),
                        });

                        state.is_generating = true;
                        state.streaming_text.clear();
                        state.thinking_text.clear();
                        state.scroll_offset = 0;

                        // Build context
                        let context_prompt = prompt::context::build_context_prompt(
                            &model_spec,
                            &session_mgr.active().title,
                            session_mgr.active().messages.len(),
                            state.context_usage,
                            &executor.available_tools(),
                            &state.steering_config,
                            "", // memory summary (will be filled when memory is wired)
                            "", // platform status
                        );

                        let system_prompt = prompt::assemble_system_prompt(
                            &core_prompt,
                            &context_prompt,
                            &identity_prompt,
                        );

                        let messages = context::build_context(
                            &system_prompt,
                            &[], // memory messages
                            &session_mgr.active().messages,
                            model_spec.context_length.max(131072), // min 128K if not derived
                        );

                        // Build tools list (always include reply_request)
                        let tools = vec![reply::reply_request_tool()];

                        let react_config = ReactConfig {
                            observer_enabled: config.observer.enabled,
                            observer_model: if config.observer.model.is_empty() {
                                None
                            } else {
                                Some(config.observer.model.clone())
                            },
                            max_audit_rejections: config.observer.max_rejections,
                        };

                        // Spawn ReAct loop
                        let provider_clone = Arc::clone(&provider);
                        let model_name = config.general.active_model.clone();
                        let tx = react_event_tx.clone();
                        let system_prompt_clone = system_prompt.clone();
                        let identity_clone = identity_prompt.clone();

                        tokio::spawn(async move {
                            let _result = react_loop::execute_react_loop(
                                &provider_clone,
                                &model_name,
                                messages,
                                &tools,
                                &ToolExecutor::new(), // Fresh executor for this turn
                                &react_config,
                                &system_prompt_clone,
                                &identity_clone,
                                tx,
                            )
                            .await;
                        });
                    }
                }
            }
        }

        if state.should_quit {
            // Save before exit
            let _ = session_mgr.save_active();
            break Ok(());
        }
    };

    // Cleanup terminal
    disable_raw_mode().context("Failed to disable raw mode")?;
    stdout()
        .execute(LeaveAlternateScreen)
        .context("Failed to leave alternate screen")?;

    tracing::info!("TUI event loop exited");
    result
}

/// Process keyboard events. Returns Some(text) if a message was submitted.
fn handle_key_event(state: &mut AppState, key: KeyEvent) -> Option<String> {
    // Global shortcuts
    match (key.modifiers, key.code) {
        (KeyModifiers::CONTROL, KeyCode::Char('c'))
        | (KeyModifiers::CONTROL, KeyCode::Char('q')) => {
            state.should_quit = true;
            return None;
        }
        (KeyModifiers::CONTROL, KeyCode::Char('m')) => {
            state.model_picker.visible = !state.model_picker.visible;
            return None;
        }
        (KeyModifiers::CONTROL, KeyCode::Char('v')) => {
            state.steering_panel.visible = !state.steering_panel.visible;
            return None;
        }
        (KeyModifiers::CONTROL, KeyCode::Char('p')) => {
            state.platform_manager.visible = !state.platform_manager.visible;
            return None;
        }
        (_, KeyCode::Tab) => {
            state.focus = match state.focus {
                Focus::Input => Focus::Chat,
                Focus::Chat => Focus::Sidebar,
                Focus::Sidebar => Focus::Input,
            };
            return None;
        }
        _ => {}
    }

    // Model picker active
    if state.model_picker.visible {
        match key.code {
            KeyCode::Esc => state.model_picker.close(),
            KeyCode::Up => state.model_picker.select_up(),
            KeyCode::Down => state.model_picker.select_down(),
            KeyCode::Enter => {
                if let Some(_model) = state.model_picker.confirm() {
                    // Model switch will be handled by session manager
                }
                state.model_picker.close();
            }
            KeyCode::Char(c) => {
                state.model_picker.search.push(c);
                state.model_picker.filter();
            }
            KeyCode::Backspace => {
                state.model_picker.search.pop();
                state.model_picker.filter();
            }
            _ => {}
        }
        return None;
    }

    // Focus-specific input handling
    match state.focus {
        Focus::Input => match key.code {
            KeyCode::Enter => {
                return state.input.submit();
            }
            KeyCode::Char(c) => state.input.insert_char(c),
            KeyCode::Backspace => state.input.backspace(),
            KeyCode::Delete => state.input.delete(),
            KeyCode::Left => state.input.move_left(),
            KeyCode::Right => state.input.move_right(),
            KeyCode::Home => state.input.home(),
            KeyCode::End => state.input.end(),
            KeyCode::Up => state.input.history_up(),
            KeyCode::Down => state.input.history_down(),
            _ => {}
        },
        Focus::Chat => match key.code {
            KeyCode::Up => state.scroll_offset = state.scroll_offset.saturating_add(1),
            KeyCode::Down => state.scroll_offset = state.scroll_offset.saturating_sub(1),
            KeyCode::Home => state.scroll_offset = 0,
            _ => {}
        },
        Focus::Sidebar => match key.code {
            KeyCode::Up => state.sidebar_selected = state.sidebar_selected.saturating_sub(1),
            KeyCode::Down => state.sidebar_selected += 1,
            KeyCode::Enter => {
                // Session switch handled externally
            }
            _ => {}
        },
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_state_defaults() {
        let state = AppState::new();
        assert_eq!(state.focus, Focus::Input);
        assert!(!state.should_quit);
        assert!(!state.is_generating);
        assert!(state.streaming_text.is_empty());
    }

    #[test]
    fn test_focus_cycle() {
        let mut state = AppState::new();
        assert_eq!(state.focus, Focus::Input);

        handle_key_event(&mut state, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(state.focus, Focus::Chat);

        handle_key_event(&mut state, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(state.focus, Focus::Sidebar);

        handle_key_event(&mut state, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
        assert_eq!(state.focus, Focus::Input);
    }

    #[test]
    fn test_ctrl_c_quits() {
        let mut state = AppState::new();
        handle_key_event(
            &mut state,
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
        );
        assert!(state.should_quit);
    }

    #[test]
    fn test_ctrl_m_toggles_model_picker() {
        let mut state = AppState::new();
        assert!(!state.model_picker.visible);

        handle_key_event(
            &mut state,
            KeyEvent::new(KeyCode::Char('m'), KeyModifiers::CONTROL),
        );
        assert!(state.model_picker.visible);

        handle_key_event(
            &mut state,
            KeyEvent::new(KeyCode::Char('m'), KeyModifiers::CONTROL),
        );
        assert!(!state.model_picker.visible);
    }

    #[test]
    fn test_input_typing() {
        let mut state = AppState::new();
        state.focus = Focus::Input;

        handle_key_event(
            &mut state,
            KeyEvent::new(KeyCode::Char('H'), KeyModifiers::NONE),
        );
        handle_key_event(
            &mut state,
            KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE),
        );

        assert_eq!(state.input.text, "Hi");
    }

    #[test]
    fn test_message_submit() {
        let mut state = AppState::new();
        state.focus = Focus::Input;
        state.input.text = "Hello Ernos".to_string();
        state.input.cursor = 11;

        let result = handle_key_event(
            &mut state,
            KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE),
        );
        assert_eq!(result, Some("Hello Ernos".to_string()));
        assert!(state.input.text.is_empty());
    }

    #[test]
    fn test_chat_scrolling() {
        let mut state = AppState::new();
        state.focus = Focus::Chat;

        handle_key_event(
            &mut state,
            KeyEvent::new(KeyCode::Up, KeyModifiers::NONE),
        );
        assert_eq!(state.scroll_offset, 1);

        handle_key_event(
            &mut state,
            KeyEvent::new(KeyCode::Down, KeyModifiers::NONE),
        );
        assert_eq!(state.scroll_offset, 0);
    }
}
