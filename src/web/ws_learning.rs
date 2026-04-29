//! WebSocket learning utilities — delayed reinforcement, insight extraction, feedback classification.

use crate::web::state::AppState;

/// Tool chain pending delayed reinforcement evaluation.
#[derive(Clone)]
pub struct PendingToolChain {
    pub user_query: String,
    pub tools: Vec<(String, String)>,
    pub reply: String,
    pub _session_id: String,
}

/// Ingest an assistant turn into memory and session persistence.
pub async fn ingest_assistant_turn(state: &AppState, text: &str, session_id: &str) {
    let mut memory = state.memory.write().await;
    memory.ingest_turn("assistant", text, session_id, None);
    drop(memory);

    let mut sessions = state.sessions.write().await;
    if let Some(session) = sessions.get_mut(session_id) {
        session.messages.push(crate::provider::Message::text("assistant", text));
        session.updated_at = chrono::Utc::now();
        let updated = session.clone();
        let _ = sessions.update(&updated);
    }
}

/// Spawn background insight extraction. Fire-and-forget.
/// Rule 2.4: If extraction fails → logged as warn, disabled for this turn.
/// Rule 2.7: On by default, no feature flag.
pub fn spawn_insight_extraction(state: &AppState, query: &str, reply: &str) {
    if !crate::observer::insights::is_worth_extracting(query) { return; }
    let provider = state.provider.clone();
    let memory = state.memory.clone();
    let query = query.to_string();
    let reply = reply.to_string();

    tokio::spawn(async move {
        match crate::observer::insights::extract_insights(&*provider, &query, &reply).await {
            Ok(insights) => {
                let mut mem = memory.write().await;
                let mut added = 0usize;
                for insight in insights {
                    if insight.confidence >= 0.7 {
                        if let Ok(true) = mem.lessons.add_if_new(
                            &insight.rule, "auto-extracted", insight.confidence
                        ) { added += 1; }
                    }
                }
                if added > 0 {
                    tracing::info!(count = added, "Insights extracted and stored");
                    let _ = mem.lessons.enforce_cap(200);
                }
            }
            Err(e) => tracing::warn!(error = %e, "Insight extraction failed — skipping"),
        }
    });
}

/// Delayed reinforcement — evaluate the previous tool chain based on the user's next message.
pub fn spawn_delayed_reinforcement(state: &AppState, chain: &PendingToolChain, next_user_msg: &str) {
    if chain.tools.is_empty() { return; }

    let memory = state.memory.clone();
    let golden = state.golden_buffer.clone();
    let rejection = state.rejection_buffer.clone();
    let chain = chain.clone();
    let next_msg = next_user_msg.to_string();

    tokio::spawn(async move {
        let signal = classify_user_feedback(&next_msg);
        match signal {
            FeedbackSignal::Approved => handle_approved(&chain, &memory, &golden).await,
            FeedbackSignal::Rejected => handle_rejected(&chain, &next_msg, &rejection).await,
            FeedbackSignal::Neutral => tracing::debug!("Delayed reinforcement: NEUTRAL — no action"),
        }
    });
}

async fn handle_approved(
    chain: &PendingToolChain,
    memory: &tokio::sync::RwLock<crate::memory::MemoryManager>,
    golden: &tokio::sync::RwLock<crate::learning::buffers::GoldenBuffer>,
) {
    tracing::info!(
        tools = chain.tools.len(),
        query = %chain.user_query.chars().take(80).collect::<String>(),
        "Delayed reinforcement: APPROVED — auto-creating procedure"
    );

    let steps: Vec<crate::memory::procedures::ProcedureStep> = chain.tools.iter().map(|(tool, args)| {
        crate::memory::procedures::ProcedureStep {
            tool: tool.clone(),
            purpose: format!("Part of chain for: {}", chain.user_query.chars().take(100).collect::<String>()),
            instruction: args.clone(),
        }
    }).collect();

    let proc_name = derive_procedure_name(&chain.tools);
    {
        let mut mem = memory.write().await;
        match mem.procedures.add_if_new(&proc_name, &chain.user_query, steps) {
            Ok(true) => {
                tracing::info!(name = %proc_name, "Auto-procedure created from approved chain");
                let _ = mem.procedures.record_success_by_name(&proc_name);
            }
            Ok(false) => {
                let _ = mem.procedures.record_success_by_name(&proc_name);
                tracing::debug!(name = %proc_name, "Procedure already exists — incrementing success");
            }
            Err(e) => tracing::warn!(error = %e, "Failed to auto-create procedure"),
        }
    }

    {
        let mut gb = golden.write().await;
        let sample = crate::learning::TrainingSample {
            id: uuid::Uuid::new_v4().to_string(),
            input: chain.user_query.clone(),
            output: chain.reply.clone(),
            method: crate::learning::TrainingMethod::Sft,
            quality_score: 0.85,
            timestamp: chrono::Utc::now(),
        };
        let _ = gb.add(sample);
    }
}

async fn handle_rejected(
    chain: &PendingToolChain,
    next_msg: &str,
    rejection: &tokio::sync::RwLock<crate::learning::buffers_rejection::RejectionBuffer>,
) {
    tracing::info!(
        tools = chain.tools.len(),
        query = %chain.user_query.chars().take(80).collect::<String>(),
        "Delayed reinforcement: REJECTED — adding to rejection buffer"
    );

    let mut rb = rejection.write().await;
    let _ = rb.add_pair(&chain.user_query, next_msg, &chain.reply, "implicit_rejection");
}

/// User feedback classification.
pub enum FeedbackSignal {
    Approved,
    Rejected,
    Neutral,
}

/// Classify the user's next message as implicit approval, rejection, or neutral.
pub fn classify_user_feedback(msg: &str) -> FeedbackSignal {
    let lower = msg.trim().to_lowercase();

    let rejection_patterns = [
        "no ", "not what", "wrong", "that's not", "thats not", "didn't ask",
        "didnt ask", "not right", "incorrect", "try again", "redo", "undo",
        "revert", "broken", "doesn't work", "doesnt work", "failed",
        "not working", "stop", "bad", "terrible", "awful", "useless",
        "why did you", "i said", "that was wrong",
    ];
    for pat in &rejection_patterns {
        if lower.contains(pat) { return FeedbackSignal::Rejected; }
    }

    let approval_patterns = [
        "now ", "next", "great", "thanks", "perfect", "good", "nice",
        "awesome", "excellent", "ok now", "alright", "cool", "yes",
        "continue", "also ", "and ", "can you also",
    ];
    for pat in &approval_patterns {
        if lower.starts_with(pat) || lower.contains(pat) { return FeedbackSignal::Approved; }
    }

    FeedbackSignal::Neutral
}

/// Derive a procedure name from the tool chain.
pub fn derive_procedure_name(tools: &[(String, String)]) -> String {
    let tool_names: Vec<&str> = tools.iter().map(|(t, _)| t.as_str()).collect();
    format!("chain_{}", tool_names.join("_"))
}
