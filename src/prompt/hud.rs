// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Dynamic HUD — regenerated before every inference call.

/// Runtime context for the HUD.
pub struct HudContext {
    pub model_name: String,
    pub provider: String,
    pub context_length: usize,
    pub session_id: String,
    pub turn_count: usize,
    pub platform: String,
    pub timeline_count: usize,
    pub lesson_count: usize,
    pub procedure_count: usize,
    pub scratchpad_count: usize,
    pub document_count: usize,
    pub golden_count: usize,
    pub rejection_count: usize,
    pub observer_enabled: bool,
    pub conversation_stack: Option<crate::prompt::conversation_stack::ConversationStack>,
}

/// Build the dynamic HUD string from live system state.
pub fn build_hud(ctx: &HudContext) -> String {
    let utc_now = chrono::Utc::now();
    let local_now = chrono::Local::now();

    let mut hud = format_hud_template(ctx, &utc_now, &local_now);

    if let Some(ref stack) = ctx.conversation_stack {
        let section = stack.to_hud_section();
        if !section.is_empty() {
            hud.push_str(&section);
        }
    }

    hud
}

/// Format the HUD template with all system state values.
fn format_hud_template(
    ctx: &HudContext,
    utc_now: &chrono::DateTime<chrono::Utc>,
    local_now: &chrono::DateTime<chrono::Local>,
) -> String {
    format!(
        "# System State (Live)\n\n\
         ## Ground Truth — Current Date & Time\n\
         This is live data from the host system clock. It is the authoritative source for all temporal reasoning.\n\
         - UTC:   {}\n\
         - Local: {}\n\n\
         ## Active Model\n\
         - Name: {}\n\
         - Provider: {}\n\
         - Context: {} tokens\n\n\
         ## Session\n\
         - ID: {}\n\
         - Turn: {}\n\
         - Platform: {}\n\n\
         ## Memory\n\
         - Timeline: {} entries\n\
         - Lessons: {} rules\n\
         - Procedures: {} skills\n\
         - Scratchpad: {} notes\n\
         - Documents: {} chunks\n\n\
         ## Learning Buffers\n\
         - Golden (SFT): {} samples\n\
         - Rejection (DPO): {} pairs\n\n\
         ## Observer: {}",
        utc_now.format("%A, %B %d, %Y at %H:%M:%S UTC"),
        local_now.format("%A, %B %d, %Y at %H:%M:%S %Z"),
        ctx.model_name, ctx.provider, ctx.context_length,
        ctx.session_id, ctx.turn_count, ctx.platform,
        ctx.timeline_count, ctx.lesson_count, ctx.procedure_count, ctx.scratchpad_count, ctx.document_count,
        ctx.golden_count, ctx.rejection_count,
        if ctx.observer_enabled { "Enabled" } else { "Disabled" },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_hud() {
        let ctx = HudContext {
            model_name: "gemma-4-31B".to_string(),
            provider: "llamacpp".to_string(),
            context_length: 131072,
            session_id: "abc123".to_string(),
            turn_count: 5,
            platform: "web".to_string(),
            timeline_count: 42,
            lesson_count: 8,
            procedure_count: 3,
            scratchpad_count: 12,
            document_count: 5,
            golden_count: 2,
            rejection_count: 0,
            observer_enabled: true,
            conversation_stack: None,
        };

        let hud = build_hud(&ctx);
        assert!(hud.contains("gemma-4-31B"));
        assert!(hud.contains("131072"));
        assert!(hud.contains("abc123"));
        assert!(hud.contains("Turn: 5"));
        assert!(hud.contains("Platform: web"));
        assert!(hud.contains("Timeline: 42"));
        assert!(hud.contains("Enabled"));
    }

    #[test]
    fn test_hud_contains_time() {
        let ctx = HudContext {
            model_name: "test".to_string(),
            provider: "test".to_string(),
            context_length: 0,
            session_id: "test".to_string(),
            turn_count: 0,
            platform: "web".to_string(),
            timeline_count: 0,
            lesson_count: 0,
            procedure_count: 0,
            scratchpad_count: 0,
            document_count: 0,
            golden_count: 0,
            rejection_count: 0,
            observer_enabled: false,
            conversation_stack: None,
        };

        let hud = build_hud(&ctx);
        assert!(hud.contains("UTC"));
        assert!(hud.contains("Ground Truth"));
    }
}
