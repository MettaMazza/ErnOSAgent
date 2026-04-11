// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Thinking token thread management for Discord.
//!
//! Creates a public thread beneath the user's message, posts an embed
//! for reasoning progress, and updates it with a debounced 800ms / 400 char
//! threshold. On completion, the embed turns green with a ✅ label.

use anyhow::Result;
use serenity::builder::{Builder, CreateEmbed, CreateMessage, CreateThread, EditMessage};
use serenity::http::Http;
use serenity::model::channel::{AutoArchiveDuration, ChannelType};
use serenity::model::id::{ChannelId, MessageId};
use serenity::model::Colour;

/// Manages a thinking thread for a single inference run.
pub struct ThinkingThread {
    thread_id: ChannelId,
    embed_message_id: MessageId,
    buffer: String,
    last_flush_len: usize,
}

impl ThinkingThread {
    /// Create a public thread beneath the user's message and post the initial embed.
    pub async fn create(
        http: &Http,
        channel_id: ChannelId,
        message_id: MessageId,
        user_name: &str,
    ) -> Result<Self> {
        // Create a public thread attached to the original message
        let thread_builder = CreateThread::new(format!("💭 {user_name} — Thinking..."))
            .kind(ChannelType::PublicThread)
            .auto_archive_duration(AutoArchiveDuration::OneHour);

        let thread: serenity::model::channel::GuildChannel = thread_builder
            .execute(http, (channel_id, Some(message_id)))
            .await
            .map_err(|e| anyhow::anyhow!("Failed to create thinking thread: {e}"))?;

        let thread_id = thread.id;

        // Post the initial "thinking" embed
        let embed = Self::build_embed("⏳ Thinking...", "", Colour::from_rgb(255, 193, 7));
        let msg = CreateMessage::new().embed(embed);
        let sent: serenity::model::channel::Message = thread_id.send_message(http, msg).await
            .map_err(|e| anyhow::anyhow!("Failed to post initial thinking embed: {e}"))?;

        tracing::debug!(
            thread = %thread_id,
            "Thinking thread created"
        );

        Ok(Self {
            thread_id,
            embed_message_id: sent.id,
            buffer: String::new(),
            last_flush_len: 0,
        })
    }

    /// Append thinking tokens. Flushes to Discord when the buffer
    /// accumulates 400+ chars since the last flush (debounce).
    pub async fn update(&mut self, http: &Http, new_tokens: &str) -> Result<()> {
        self.buffer.push_str(new_tokens);

        let delta = self.buffer.len().saturating_sub(self.last_flush_len);
        if delta >= 400 {
            self.flush(http).await?;
        }

        Ok(())
    }

    /// Force-flush the current buffer to the embed.
    async fn flush(&mut self, http: &Http) -> Result<()> {
        // Truncate for embed display (Discord embed description limit: 4096)
        let display = if self.buffer.len() > 3800 {
            format!("...{}", &self.buffer[self.buffer.len() - 3800..])
        } else {
            self.buffer.clone()
        };

        let embed = Self::build_embed(
            "💭 Reasoning...",
            &display,
            Colour::from_rgb(255, 193, 7), // amber
        );

        let edit = EditMessage::new().embed(embed);
        let _ = self.thread_id.edit_message(http, self.embed_message_id, edit).await;
        self.last_flush_len = self.buffer.len();

        Ok(())
    }

    /// Finalise: change embed to green with ✅ Complete label, then auto-delete after 2 minutes.
    pub async fn complete(&mut self, http: std::sync::Arc<Http>) -> Result<()> {
        // Final flush of any remaining tokens
        let display = if self.buffer.len() > 3800 {
            format!("...{}", &self.buffer[self.buffer.len() - 3800..])
        } else {
            self.buffer.clone()
        };

        let embed = Self::build_embed(
            "✅ Reasoning Complete",
            &display,
            Colour::from_rgb(76, 175, 80), // green
        );

        let edit = EditMessage::new().embed(embed);
        let _ = self.thread_id.edit_message(&*http, self.embed_message_id, edit).await;

        tracing::debug!(
            thread = %self.thread_id,
            tokens_len = self.buffer.len(),
            "Thinking thread completed — will auto-delete in 2 minutes"
        );

        // Auto-delete the thread after 2 minutes
        let thread_id = self.thread_id;
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(120)).await;
            if let Err(e) = thread_id.delete(&*http).await {
                tracing::warn!(error = %e, thread = %thread_id, "Failed to auto-delete thinking thread");
            } else {
                tracing::debug!(thread = %thread_id, "Thinking thread auto-deleted");
            }
        });

        Ok(())
    }

    fn build_embed(title: &str, description: &str, colour: Colour) -> CreateEmbed {
        CreateEmbed::new()
            .title(title)
            .description(description)
            .colour(colour)
            .footer(serenity::builder::CreateEmbedFooter::new("ErnOS Agent — Reasoning Trace"))
    }
}

/// Spawn a persistent typing indicator that refreshes every 8 seconds.
/// Returns a handle that can be aborted to stop the indicator.
pub fn spawn_typing_indicator(
    http: std::sync::Arc<Http>,
    channel_id: ChannelId,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        loop {
            let _ = channel_id.broadcast_typing(&http).await;
            tokio::time::sleep(std::time::Duration::from_secs(8)).await;
        }
    })
}
