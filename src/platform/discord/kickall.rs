// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! /kickall and /cancelkickall — server shutdown commands (admin-only).
//!
//! Starts a 24-hour countdown. Sends a farewell notice to the channel.
//! After 24h, kicks all non-bot, non-owner members from every guild.
//! Cancellable at any time via /cancelkickall.

use serenity::all::{CommandInteraction, Context, GuildId, Member};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Global state for the kickall timer.
static KICKALL_ACTIVE: std::sync::OnceLock<Arc<AtomicBool>> = std::sync::OnceLock::new();
static KICKALL_HANDLE: std::sync::OnceLock<Mutex<Option<tokio::task::JoinHandle<()>>>> =
    std::sync::OnceLock::new();

fn active_flag() -> &'static Arc<AtomicBool> {
    KICKALL_ACTIVE.get_or_init(|| Arc::new(AtomicBool::new(false)))
}

fn handle_slot() -> &'static Mutex<Option<tokio::task::JoinHandle<()>>> {
    KICKALL_HANDLE.get_or_init(|| Mutex::new(None))
}

const FAREWELL_MESSAGE: &str = r#"**⚠️ SERVER SHUTDOWN NOTICE — 24 HOURS ⚠️**

Thank you for taking part in ErnOS. It's clear this is not a valuable project to many people who decide to check it out — it's more bait for trolls than anything valuable. I'm done and the work is over. I'm so sick of people trolling me and my work, and when I say something or pop off with some attitude I'm the bad guy. Well I'll be the bad guy. You can all get fucked you soft fucking pussy bitches, and if you really did care about this project, that shows... because like 3 people ever fucking did anything of value to justify being here. You're parasites.

**All members will be kicked in 24 hours. This server is shutting down.**

_The owner can cancel this with `/cancelkickall` at any time._"#;

/// Handle the /kickall command.
pub async fn handle_kickall(
    ctx: &Context,
    command: &CommandInteraction,
    admin_user_ids: &[String],
) -> String {
    let author_id = command.user.id.to_string();

    // Admin-only gate
    if !admin_user_ids.iter().any(|id| id == &author_id) {
        return "❌ This command is admin-only.".to_string();
    }

    // Check if already active
    if active_flag().load(Ordering::SeqCst) {
        return "⚠️ A /kickall countdown is already active. Use `/cancelkickall` to cancel it first.".to_string();
    }

    let guild_id = match command.guild_id {
        Some(id) => id,
        None => return "❌ This command can only be used in a server.".to_string(),
    };

    let channel_id = command.channel_id;
    let http = ctx.http.clone();
    let owner_id = command.user.id;

    // Send the farewell notice to the channel (visible to everyone)
    if let Err(e) = channel_id
        .say(&http, FAREWELL_MESSAGE)
        .await
    {
        tracing::error!(error = %e, "Failed to send kickall farewell message");
    }

    // Mark as active
    active_flag().store(true, Ordering::SeqCst);

    // Spawn the 24-hour countdown
    let active = active_flag().clone();
    let handle = tokio::spawn(async move {
        tracing::warn!(
            guild = %guild_id,
            hours = 24,
            "KICKALL COUNTDOWN STARTED — all members will be kicked in 24 hours"
        );

        // Sleep for 24 hours (in 1-minute increments for cancellation responsiveness)
        let total_minutes = 24 * 60;
        for minute in 0..total_minutes {
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;

            if !active.load(Ordering::SeqCst) {
                tracing::info!("KICKALL CANCELLED during countdown (minute {})", minute + 1);
                return;
            }

            // Hourly status log
            if (minute + 1) % 60 == 0 {
                let hours_remaining = 24 - ((minute + 1) / 60);
                tracing::warn!(
                    hours_remaining = hours_remaining,
                    "KICKALL countdown: {} hours remaining",
                    hours_remaining
                );
            }
        }

        // Final check before executing
        if !active.load(Ordering::SeqCst) {
            tracing::info!("KICKALL CANCELLED at the last moment");
            return;
        }

        tracing::warn!(guild = %guild_id, "KICKALL EXECUTING — kicking all members");

        // Send a final warning
        let _ = channel_id
            .say(&http, "⏰ **24 hours have passed. Executing server shutdown. Goodbye.**")
            .await;

        // Execute the kicks
        execute_kickall(&http, guild_id, owner_id).await;

        active.store(false, Ordering::SeqCst);
        tracing::warn!("KICKALL COMPLETE");
    });

    // Store the handle for cancellation
    {
        let mut slot = handle_slot().lock().await;
        *slot = Some(handle);
    }

    "⚠️ **KICKALL INITIATED** — 24-hour countdown started. All members will be kicked when the timer expires.\n\nUse `/cancelkickall` to abort.".to_string()
}

/// Handle the /cancelkickall command.
pub async fn handle_cancelkickall(
    _ctx: &Context,
    command: &CommandInteraction,
    admin_user_ids: &[String],
) -> String {
    let author_id = command.user.id.to_string();

    if !admin_user_ids.iter().any(|id| id == &author_id) {
        return "❌ This command is admin-only.".to_string();
    }

    if !active_flag().load(Ordering::SeqCst) {
        return "ℹ️ No kickall countdown is currently active.".to_string();
    }

    // Cancel the countdown
    active_flag().store(false, Ordering::SeqCst);

    // Abort the task
    {
        let mut slot = handle_slot().lock().await;
        if let Some(handle) = slot.take() {
            handle.abort();
        }
    }

    tracing::info!(
        cancelled_by = %command.user.name,
        "KICKALL CANCELLED by admin"
    );

    "✅ **KICKALL CANCELLED** — countdown aborted. No members will be kicked.".to_string()
}

/// Execute the actual member kicks across the guild.
async fn execute_kickall(
    http: &serenity::http::Http,
    guild_id: GuildId,
    owner_id: serenity::model::id::UserId,
) {
    // Fetch members in batches
    let mut after: Option<serenity::model::id::UserId> = None;
    let mut total_kicked = 0_usize;
    let mut total_failed = 0_usize;

    loop {
        let members: Vec<Member> = match guild_id
            .members(http, Some(1000), after)
            .await
        {
            Ok(m) => m,
            Err(e) => {
                tracing::error!(error = %e, "Failed to fetch guild members for kickall");
                break;
            }
        };

        if members.is_empty() {
            break;
        }

        let batch_size = members.len();
        after = members.last().map(|m| m.user.id);

        for member in &members {
            // Skip bots, the server owner, and anyone with a role
            // (roles list always contains @everyone as guild_id — skip that)
            if member.user.bot || member.user.id == owner_id {
                continue;
            }
            let everyone_role = serenity::model::id::RoleId::new(guild_id.get());
            let has_role = member.roles.iter().any(|r| *r != everyone_role);
            if has_role {
                tracing::debug!(
                    user = %member.user.name,
                    roles = member.roles.len(),
                    "Skipping member with role(s)"
                );
                continue;
            }

            let reason = "Server shutdown — /kickall executed by owner";
            match guild_id.kick_with_reason(http, member.user.id, reason).await {
                Ok(()) => {
                    total_kicked += 1;
                    tracing::info!(
                        user = %member.user.name,
                        user_id = %member.user.id,
                        "Kicked member"
                    );
                }
                Err(e) => {
                    total_failed += 1;
                    tracing::warn!(
                        user = %member.user.name,
                        error = %e,
                        "Failed to kick member"
                    );
                }
            }

            // Rate limit: 50ms between kicks to avoid Discord API throttling
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // If we got fewer than 1000, we've fetched all members
        if batch_size < 1000 {
            break;
        }
    }

    tracing::warn!(
        kicked = total_kicked,
        failed = total_failed,
        "KICKALL execution complete"
    );
}
