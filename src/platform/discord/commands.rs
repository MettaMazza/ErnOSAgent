// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Discord slash commands — /status, /clean, /session, /kickall, /cancelkickall, /interview.

use serenity::all::{
    CommandInteraction, Context, CreateCommand, CreateInteractionResponse,
    CreateInteractionResponseMessage,
};

/// Onboarding config passed to command handlers.
#[derive(Clone, Default)]
pub struct OnboardingConfig {
    pub channel_id: String,
    pub role_id: String,
    pub guild_id: String,
}

/// Build the command list.
fn build_commands() -> Vec<CreateCommand> {
    vec![
        CreateCommand::new("status")
            .description("Show ErnOS system status — model, memory, and connection info"),
        CreateCommand::new("clean")
            .description("Factory reset your memory and session (admin-only)"),
        CreateCommand::new("session")
            .description("Show your current session info"),
        CreateCommand::new("interview")
            .description("Start an onboarding interview for the next unverified member (admin-only)"),
        CreateCommand::new("kickall")
            .description("⚠️ NUCLEAR: Start 24h countdown to kick all members (admin-only)"),
        CreateCommand::new("cancelkickall")
            .description("Cancel an active /kickall countdown (admin-only)"),
    ]
}

/// Register slash commands for a specific guild (instant propagation).
pub async fn register_guild_commands(http: &serenity::http::Http, guild_id: serenity::model::id::GuildId) {
    let commands = build_commands();
    match guild_id.set_commands(http, commands).await {
        Ok(cmds) => {
            tracing::info!(guild = %guild_id, count = cmds.len(), "Discord slash commands registered (guild)");
        }
        Err(e) => {
            tracing::error!(guild = %guild_id, error = %e, "Failed to register Discord slash commands for guild");
        }
    }
}

/// Dispatch a slash command interaction.
pub async fn handle_command(
    ctx: &Context,
    command: &CommandInteraction,
    admin_user_ids: &[String],
    onboarding_config: &OnboardingConfig,
) {
    // Slow commands: defer immediately (Discord 3s timeout), then follow up
    let is_slow = matches!(command.data.name.as_str(), "interview" | "kickall" | "cancelkickall");

    if is_slow {
        // Acknowledge immediately — buys us 15 minutes
        let defer = CreateInteractionResponse::Defer(
            CreateInteractionResponseMessage::new().ephemeral(true),
        );
        if let Err(e) = command.create_response(&ctx.http, defer).await {
            tracing::error!(command = %command.data.name, error = %e, "Failed to defer slash command");
            return;
        }

        let response_text = match command.data.name.as_str() {
            "interview" => handle_interview(ctx, command, admin_user_ids, onboarding_config).await,
            "kickall" => super::kickall::handle_kickall(ctx, command, admin_user_ids).await,
            "cancelkickall" => super::kickall::handle_cancelkickall(ctx, command, admin_user_ids).await,
            _ => "Unknown command.".to_string(),
        };

        let followup = serenity::builder::CreateInteractionResponseFollowup::new()
            .content(response_text)
            .ephemeral(true);
        if let Err(e) = command.create_followup(&ctx.http, followup).await {
            tracing::error!(command = %command.data.name, error = %e, "Failed to send slash command followup");
        }
    } else {
        // Fast commands: respond immediately
        let response_text = match command.data.name.as_str() {
            "status" => handle_status().await,
            "clean" => handle_clean(command).await,
            "session" => handle_session(command).await,
            _ => "Unknown command.".to_string(),
        };

        let response = CreateInteractionResponse::Message(
            CreateInteractionResponseMessage::new()
                .content(response_text)
                .ephemeral(true),
        );

        if let Err(e) = command.create_response(&ctx.http, response).await {
            tracing::error!(
                command = %command.data.name,
                error = %e,
                "Failed to respond to slash command"
            );
        }
    }
}

async fn handle_status() -> String {
    "🟢 **ErnOS Agent Online**\n\
    Use the chat to interact with the full ReAct pipeline.\n\
    For detailed status, check the web dashboard.".to_string()
}

async fn handle_clean(command: &CommandInteraction) -> String {
    format!(
        "🔄 Session reset requested by **{}**.\n\
        Note: Full factory reset requires admin access via the web dashboard.",
        command.user.name
    )
}

async fn handle_session(command: &CommandInteraction) -> String {
    format!(
        "📋 **Session Info**\n\
        User: {}\n\
        User ID: {}\n\
        Platform: Discord\n\
        Session key: discord:{}",
        command.user.name,
        command.user.id,
        command.user.id,
    )
}

/// `/interview` — Find the next member without a role and start their onboarding interview.
/// Admin-only. Processes one member at a time to avoid overwhelming the agent.
async fn handle_interview(
    ctx: &Context,
    command: &CommandInteraction,
    admin_user_ids: &[String],
    config: &OnboardingConfig,
) -> String {
    let caller_id = command.user.id.to_string();

    // Admin check
    if !admin_user_ids.iter().any(|id| id == &caller_id) {
        return "❌ Admin-only command.".to_string();
    }

    // Check onboarding is configured
    if config.channel_id.is_empty() || config.role_id.is_empty() || config.guild_id.is_empty() {
        return "❌ Onboarding not configured — set `onboarding_channel_id`, `new_member_role_id`, and `guild_id` in config.".to_string();
    }

    let guild_id: u64 = match config.guild_id.parse() {
        Ok(id) => id,
        Err(_) => return "❌ Invalid guild_id in config.".to_string(),
    };

    let _role_id: u64 = match config.role_id.parse() {
        Ok(id) => id,
        Err(_) => return "❌ Invalid role_id in config.".to_string(),
    };

    let channel_id: u64 = match config.channel_id.parse() {
        Ok(id) => id,
        Err(_) => return "❌ Invalid onboarding_channel_id in config.".to_string(),
    };

    // Fetch guild members
    let guild = serenity::model::id::GuildId::new(guild_id);

    // Fetch members in batches (Discord limits to 1000 per request)
    let members = match guild.members(&ctx.http, Some(1000), None::<serenity::model::id::UserId>).await {
        Ok(m) => m,
        Err(e) => return format!("❌ Failed to fetch guild members: {}", e),
    };

    // Find the next member without ANY role (besides @everyone)
    // Skip bots and admins
    // Also skip anyone with an active interview
    let mut unverified: Vec<&serenity::model::guild::Member> = members.iter()
        .filter(|m| {
            // Skip bots
            if m.user.bot { return false; }
            // Skip admins
            let uid = m.user.id.to_string();
            if admin_user_ids.iter().any(|id| id == &uid) { return false; }
            // Skip members who already have roles (besides @everyone)
            // @everyone role ID == guild ID
            let has_role = m.roles.iter().any(|r| r.get() != guild_id);
            if has_role { return false; }
            // Skip members with active interviews
            if super::onboarding::is_user_being_interviewed(&uid) { return false; }
            true
        })
        .collect();

    // Sort by join date (oldest first — they've been waiting longest)
    unverified.sort_by(|a, b| {
        a.joined_at.cmp(&b.joined_at)
    });

    let total_unverified = unverified.len();

    let member = match unverified.first() {
        Some(m) => *m,
        None => return "✅ No unverified members remaining — everyone has been interviewed or has a role.".to_string(),
    };

    let user_id = member.user.id.get();
    let user_name = &member.user.name;

    // Create the interview thread
    match super::onboarding::create_interview_thread(
        &ctx.http, channel_id, user_id, user_name
    ).await {
        Ok(thread_id) => {
            tracing::info!(
                user_id = user_id,
                user_name = %user_name,
                thread_id = thread_id,
                remaining = total_unverified - 1,
                "Backfill interview started via /interview"
            );
            format!(
                "📋 **Interview started** for **{}** (ID: {})\n\
                Thread created in <#{}>\n\
                Remaining unverified: **{}**\n\
                Run `/interview` again for the next one.",
                user_name, user_id, config.channel_id, total_unverified - 1
            )
        }
        Err(e) => {
            tracing::error!(
                error = %e,
                user_id = user_id,
                "Failed to create backfill interview thread"
            );
            format!("❌ Failed to create interview thread for {}: {}", user_name, e)
        }
    }
}
