// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
// Module: Tool schema definitions

// ─── Original work by @mettamazza — do not remove this attribution ───
//! Tool schema definitions for LLM function-calling.
//!
//! Every tool in the executor must have a matching schema here so the LLM
//! knows its name, description, and parameter structure. The `reply_request`
//! tool is defined separately in `react::reply` and appended automatically.

use crate::provider::{ToolDefinition, ToolFunction};

/// Build a ToolDefinition helper (reduces boilerplate).
fn def(name: &str, description: &str, params: serde_json::Value) -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: ToolFunction {
            name: name.to_string(),
            description: description.to_string(),
            parameters: params,
        },
    }
}

/// Return all tool definitions for injection into the LLM inference call.
///
/// This does NOT include `reply_request` — that is appended separately
/// by the chat handler to guarantee it is always last.
pub fn all_tool_definitions() -> Vec<ToolDefinition> {
    vec![
        // ── Browser Automation ───────────────────────────────────
        def("browser_navigate",
            "Open the built-in Chrome browser, navigate to a URL, and return a preview of the rendered page content.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The fully qualified URL to visit (e.g. 'https://github.com/')."
                    }
                },
                "required": ["url"]
            })),
        def("browser_click",
            "Click on an element in the active browser tab.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "A valid CSS selector for the element to click."
                    }
                },
                "required": ["selector"]
            })),
        def("browser_type",
            "Type text into an input field on the active browser tab.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "selector": {
                        "type": "string",
                        "description": "A valid CSS selector for the input element."
                    },
                    "text": {
                        "type": "string",
                        "description": "The exact text string to type into the element."
                    }
                },
                "required": ["selector", "text"]
            })),

        // ── Memory ───────────────────────────────────────────────
        def("memory_tool",
            "Access and manage the 5-tier memory system. Use this to store information, \
             check memory status, recall from past interactions, or trigger consolidation.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["store", "status", "recall", "consolidate"],
                        "description": "The memory operation: 'store' saves a key-value pair to persistent memory, \
                                        'status' shows all tier counts, \
                                        'recall' searches memory for a query, \
                                        'consolidate' compacts the context."
                    },
                    "key": {
                        "type": "string",
                        "description": "Storage key/label for the 'store' action."
                    },
                    "value": {
                        "type": "string",
                        "description": "Content to store (for 'store' action)."
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for the 'recall' action."
                    },
                    "budget": {
                        "type": "integer",
                        "description": "Maximum characters to return for 'recall'."
                    }
                },
                "required": ["action"]
            }),
        ),

        def("scratchpad_tool",
            "Read, write, or delete entries in the persistent scratchpad (key-value notes).",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "read", "write", "delete"],
                        "description": "The scratchpad operation."
                    },
                    "key": {
                        "type": "string",
                        "description": "The entry key (required for write/delete)."
                    },
                    "value": {
                        "type": "string",
                        "description": "The entry value (required for write)."
                    }
                },
                "required": ["action"]
            }),
        ),

        def("lessons_tool",
            "Manage learned rules and insights. Store new lessons, search existing ones, \
             or reinforce/weaken lesson confidence.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "store", "search", "reinforce", "weaken"],
                        "description": "The lessons operation."
                    },
                    "rule": {
                        "type": "string",
                        "description": "The lesson rule text (required for 'store')."
                    },
                    "keywords": {
                        "type": "string",
                        "description": "Comma-separated keywords for the lesson."
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Initial confidence (0.0-1.0) for 'store'."
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for 'search' action."
                    }
                },
                "required": ["action"]
            }),
        ),

        def("timeline_tool",
            "Browse the sequential event timeline. View recent events, search by query, \
             or get timeline statistics.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["recent", "search", "stats"],
                        "description": "The timeline operation."
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for 'search' action."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum events to return (default: 10)."
                    }
                },
                "required": ["action"]
            }),
        ),

        // ── Reasoning ────────────────────────────────────────────
        def("reasoning_tool",
            "Store, search, and review persistent reasoning traces. Use 'store' to save \
             a chain of thought, 'search' to find past reasoning, 'review' for self-audit, \
             'stats' for reasoning metrics.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["store", "search", "review", "stats"],
                        "description": "The reasoning operation."
                    },
                    "thinking": {
                        "type": "string",
                        "description": "The reasoning chain / thinking text (for 'store')."
                    },
                    "outcome": {
                        "type": "string",
                        "description": "The conclusion or outcome reached (for 'store')."
                    },
                    "decisions": {
                        "type": "string",
                        "description": "Comma-separated tool decisions made (for 'store')."
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for 'search')."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results (default: 5)."
                    }
                },
                "required": ["action"]
            }),
        ),

        // ── Web ──────────────────────────────────────────────────
        def("web_tool",
            "Search the web or visit a specific URL. Use 'search' for general queries \
             (DuckDuckGo) or 'visit' to fetch and read page content.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["search", "visit"],
                        "description": "The web operation."
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (for 'search')."
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to visit (for 'visit')."
                    }
                },
                "required": ["action"]
            }),
        ),

        // ── Codebase ─────────────────────────────────────────────
        def("codebase_read",
            "Read the contents of a file from the local filesystem.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file."
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional start line (1-indexed)."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional end line (1-indexed, inclusive)."
                    }
                },
                "required": ["path"]
            }),
        ),

        def("codebase_write",
            "Write content to a file, creating it if it doesn't exist.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file."
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete file content to write."
                    }
                },
                "required": ["path", "content"]
            }),
        ),

        def("codebase_patch",
            "Replace a specific string in a file with new content.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file."
                    },
                    "find": {
                        "type": "string",
                        "description": "The exact string to find and replace."
                    },
                    "replace": {
                        "type": "string",
                        "description": "The replacement string."
                    }
                },
                "required": ["path", "find", "replace"]
            }),
        ),

        def("codebase_list",
            "List the contents of a directory.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the directory (default: current dir)."
                    }
                },
                "required": []
            }),
        ),

        def("codebase_search",
            "Search for a pattern across files in a directory.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to search in."
                    },
                    "query": {
                        "type": "string",
                        "description": "The search pattern."
                    }
                },
                "required": ["path", "query"]
            }),
        ),

        def("codebase_delete",
            "Delete a file or empty directory.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to delete."
                    }
                },
                "required": ["path"]
            }),
        ),

        def("codebase_insert",
            "Insert content at a specific line in a file.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file."
                    },
                    "line": {
                        "type": "integer",
                        "description": "Line number to insert at (1-indexed)."
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to insert."
                    }
                },
                "required": ["path", "line", "content"]
            }),
        ),

        def("codebase_multi_patch",
            "Apply multiple find-replace patches to a file atomically.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file."
                    },
                    "patches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "find": { "type": "string" },
                                "replace": { "type": "string" }
                            }
                        },
                        "description": "Array of {find, replace} patches."
                    }
                },
                "required": ["path", "patches"]
            }),
        ),

        // ── Shell ────────────────────────────────────────────────
        def("run_command",
            "Execute a shell command and return its output. Commands run in a \
             sandboxed environment with safety checks.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute."
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds (default: 30)."
                    }
                },
                "required": ["command"]
            }),
        ),

        // ── Compiler ─────────────────────────────────────────────
        def("system_recompile",
            "Trigger a recompilation of the ErnOS agent itself. If this fails due to \
             warnings or test failures, you MUST fix the issues using codebase_edit and \
             retry automatically. Do NOT ask the user — self-heal recursively until it passes.",
            serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        ),

        // ── Git ──────────────────────────────────────────────────
        def("git_tool",
            "Perform git operations: status, diff, log, blame, branches, commit, stash.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["status", "diff", "log", "blame", "branches", "commit",
                                 "stash", "stash_pop"],
                        "description": "The git operation."
                    },
                    "staged": {
                        "type": "boolean",
                        "description": "Show staged diff (for 'diff')."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of log entries (for 'log')."
                    },
                    "path": {
                        "type": "string",
                        "description": "File path (for 'blame')."
                    },
                    "line": {
                        "type": "integer",
                        "description": "Starting line number for 'blame' (default: 1)."
                    },
                    "message": {
                        "type": "string",
                        "description": "Commit message (for 'commit')."
                    }
                },
                "required": ["action"]
            }),
        ),

        // ── Tool Forge ───────────────────────────────────────────
        def("tool_forge",
            "Create, edit, test, list, or delete runtime-forged tools.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "edit", "test", "dry_run", "enable", "disable", "delete", "list"],
                        "description": "The forge operation."
                    },
                    "name": {
                        "type": "string",
                        "description": "Tool name (for create/edit/test/delete)."
                    },
                    "code": {
                        "type": "string",
                        "description": "Tool implementation code (for create/edit)."
                    },
                    "description": {
                        "type": "string",
                        "description": "Tool description (for create)."
                    },
                    "test_input": {
                        "type": "string",
                        "description": "Test arguments JSON (for test)."
                    }
                },
                "required": ["action"]
            }),
        ),

        // ── Steering ─────────────────────────────────────────────
        def("steering_tool",
            "Control SAE feature steering and vector steering. Amplify or suppress \
             cognitive features (reasoning, creativity, safety) in real-time.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["feature_list", "feature_steer", "feature_clear",
                                 "feature_status", "vector_scan", "vector_activate",
                                 "vector_deactivate", "vector_status"],
                        "description": "The steering operation."
                    },
                    "feature_id": {
                        "type": "integer",
                        "description": "Feature ID (for feature_steer)."
                    },
                    "scale": {
                        "type": "number",
                        "description": "Steering scale (for feature_steer / vector_scale)."
                    },
                    "name": {
                        "type": "string",
                        "description": "Vector name (for vector_toggle / vector_scale)."
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (for feature_list)."
                    }
                },
                "required": ["action"]
            }),
        ),

        // ── Interpretability ─────────────────────────────────────
        def("interpretability_tool",
            "Inspect the neural interpretability layer. Get SAE snapshots, list features, \
             check safety alerts, extract directions, or browse the feature catalog.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["snapshot", "features", "safety_alerts",
                                 "cognitive_profile", "emotional_state",
                                 "extract_direction", "catalog"],
                        "description": "The interpretability operation."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max features to return (default: 15)."
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Minimum activation threshold."
                    },
                    "text": {
                        "type": "string",
                        "description": "Text for direction extraction."
                    },
                    "feature_id": {
                        "type": "integer",
                        "description": "Feature ID (for specific feature inspection)."
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by category (for 'features', 'catalog')."
                    }
                },
                "required": ["action"]
            }),
        ),

        // ── Download ─────────────────────────────────────────────
        def("download_tool",
            "Download files from URLs with progress tracking.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["download", "status", "list"],
                        "description": "The download operation. 'status' without filename shows all downloads. 'list' shows all files."
                    },
                    "url": {
                        "type": "string",
                        "description": "URL to download (required for 'download')."
                    },
                    "filename": {
                        "type": "string",
                        "description": "Filename (optional for 'download' to override, optional for 'status' to check specific file)."
                    }
                },
                "required": ["action"]
            }),
        ),

        // ── Synaptic Graph ───────────────────────────────────────
        def("operate_synaptic_graph",
            "Interact with the synaptic knowledge graph. Store concepts, search by meaning, \
             query beliefs, create relations between concepts, or run graph analytics.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["store", "search", "beliefs", "relate",
                                 "stats", "layers", "link_memory"],
                        "description": "The synaptic graph operation."
                    },
                    "concept": {
                        "type": "string",
                        "description": "Concept text (for 'store', 'search', 'link_memory')."
                    },
                    "data": {
                        "type": "string",
                        "description": "Data/context for the concept (for 'store')."
                    },
                    "from": {
                        "type": "string",
                        "description": "Source concept (for 'relate')."
                    },
                    "relation": {
                        "type": "string",
                        "description": "Relation type (for 'relate')."
                    },
                    "to": {
                        "type": "string",
                        "description": "Target concept (for 'relate')."
                    },
                    "target": {
                        "type": "string",
                        "description": "Target memory entry ID (for 'link_memory')."
                    },
                    "memory_type": {
                        "type": "string",
                        "description": "Memory tier to link to: 'timeline', 'lesson', 'scratchpad' (for 'link_memory')."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (for 'beliefs'). Default: 10."
                    }
                },
                "required": ["action"]
            }),
        ),

        // ── 3D Turing Grid ───────────────────────────────────────
        def("operate_turing_grid",
            "Operate the 3D Turing computation grid. Move the head, read/write cells, \
             scan regions, execute programs, manage pipelines and daemons.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["move", "read", "write", "scan", "read_range",
                                 "index", "execute", "pipeline", "deploy_daemon",
                                 "label", "goto", "link", "history", "undo"],
                        "description": "The grid operation."
                    },
                    "dx": { "type": "integer", "description": "Relative X delta for 'move'. Positive = right." },
                    "dy": { "type": "integer", "description": "Relative Y delta for 'move'. Positive = up." },
                    "dz": { "type": "integer", "description": "Relative Z delta for 'move'. Positive = forward." },
                    "content": { "type": "string", "description": "Content to write to cell (for 'write')." },
                    "format": { "type": "string", "description": "Format for write: 'text', 'python', 'javascript', 'rust'. Default: 'text'." },
                    "name": { "type": "string", "description": "Label name (for 'label' and 'goto')." },
                    "target_x": { "type": "integer", "description": "Target X coordinate for 'link'." },
                    "target_y": { "type": "integer", "description": "Target Y coordinate for 'link'." },
                    "target_z": { "type": "integer", "description": "Target Z coordinate for 'link'." },
                    "radius": { "type": "integer", "description": "Scan radius (for 'scan'). Default: 5." },
                    "cells": { "type": "string", "description": "Coordinate tuples for 'pipeline', e.g. '(0,0,0),(1,0,0)'." },
                    "interval": { "type": "integer", "description": "Daemon interval in seconds (for 'deploy_daemon'). Min: 10." },
                    "x_bounds": { "type": "string", "description": "X range for 'read_range', e.g. '0,5'." },
                    "y_bounds": { "type": "string", "description": "Y range for 'read_range', e.g. '0,5'." },
                    "z_bounds": { "type": "string", "description": "Z range for 'read_range', e.g. '0,0'." }
                },
                "required": ["action"]
            }),
        ),

        // ── Self-Supervised Learning ─────────────────────────────
        def("distill_knowledge",
            "Generate synthetic training data by having an expert model create \
             domain-specific Q&A pairs. These are added to the golden buffer \
             for future LoRA training. Requires ERNOS_TRAINING_ENABLED=1.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": "The domain or topic to generate training data for \
                                        (e.g. 'Rust async programming', 'distributed systems')."
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of Q&A pairs to generate (default: 5)."
                    }
                },
                "required": ["domain"]
            }),
        ),

        def("performance_review",
            "Self-introspection tool — reviews training data, failure patterns, \
             success patterns, and learned lessons. Use to understand what needs \
             improvement and what is working well.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": ["full", "failures", "successes"],
                        "description": "Review scope: 'full' for everything, \
                                        'failures' for failure patterns only, \
                                        'successes' for golden examples only."
                    }
                },
                "required": []
            }),
        ),

        // ── Scheduler ────────────────────────────────────────────
        def("scheduler_tool",
            "Create, list, delete, toggle, or force-run scheduled tasks. \
             Use this to set up reminders, recurring jobs, or idle-triggered \
             autonomy tasks. Jobs execute through the full ReAct + Observer pipeline.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "delete", "toggle", "run_now"],
                        "description": "The scheduler operation."
                    },
                    "name": {
                        "type": "string",
                        "description": "Job name (required for 'create')."
                    },
                    "instruction": {
                        "type": "string",
                        "description": "Natural language instruction for the job (required for 'create')."
                    },
                    "schedule_type": {
                        "type": "string",
                        "enum": ["cron", "once", "interval", "idle"],
                        "description": "Schedule type (required for 'create'). \
                                        'cron' = standard cron expression, \
                                        'once' = ISO datetime, \
                                        'interval' = seconds between runs, \
                                        'idle' = fire when user idle for N seconds."
                    },
                    "schedule_value": {
                        "type": "string",
                        "description": "Schedule value (required for 'create'). \
                                        Cron: '0 9 * * *', Once: '2026-04-13T09:00:00Z', \
                                        Interval: '3600', Idle: '300'."
                    },
                    "job_id": {
                        "type": "string",
                        "description": "Job ID (required for delete/toggle/run_now)."
                    }
                },
                "required": ["action"]
            }),
        ),

        // ── Autonomy History ─────────────────────────────────────
        def("autonomy_history",
            "Introspect past autonomy sessions — review what the system did during \
             idle periods. Use to understand patterns, avoid repeating work, and \
             track autonomous productivity.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "detail", "search", "stats"],
                        "description": "The introspection action: \
                                        'list' shows recent sessions, \
                                        'detail' shows a specific cycle, \
                                        'search' finds sessions by keyword, \
                                        'stats' shows aggregate statistics."
                    },
                    "cycle": {
                        "type": "integer",
                        "description": "Cycle number for 'detail' action."
                    },
                    "query": {
                        "type": "string",
                        "description": "Search keyword for 'search' action."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default: 20 for list, 10 for search)."
                    }
                },
                "required": ["action"]
            }),
        ),

        // ── Moderation (Discord-only) ────────────────────────────
        def("moderation_tool",
            "Self-moderation tool — mute abusive users, set topic boundaries, \
             escalate to admin, make onboarding decisions, ban users. Discord-only.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["mute_user", "unmute_user", "list_muted",
                                 "set_boundary", "remove_boundary", "escalate",
                                 "onboarding_decision", "ban_user"],
                        "description": "The moderation action."
                    },
                    "user_id": {
                        "type": "string",
                        "description": "Discord user ID."
                    },
                    "user_name": {
                        "type": "string",
                        "description": "Discord username (for audit logs)."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for the action."
                    },
                    "duration_minutes": {
                        "type": "integer",
                        "description": "Mute duration in minutes. Omit for permanent mute."
                    },
                    "topic": {
                        "type": "string",
                        "description": "Topic to set/remove boundary on."
                    },
                    "concern": {
                        "type": "string",
                        "description": "Description of the ethical concern (for escalate)."
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Severity level (for escalate)."
                    },
                    "context": {
                        "type": "string",
                        "description": "Additional context."
                    },
                    "decision": {
                        "type": "string",
                        "enum": ["pass", "fail"],
                        "description": "Interview decision (for onboarding_decision)."
                    },
                    "scores": {
                        "type": "string",
                        "description": "Scoring breakdown (for onboarding_decision, e.g. 'Technical:20 Philosophy:15 Attitude:20 Engagement:18 = 73/100')."
                    }
                },
                "required": ["action"]
            }),
        ),

        // ── Image Generation ─────────────────────────────────────
        def("image_tool",
            "Generate an image using local Flux Dev. Limited to 1 image per turn. \
             Requires FLUX_SERVER_URL to be configured. You will SEE the generated \
             image in the next message — describe what you see to the user.",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image to generate. \
                                        Be specific: include subject, style, lighting, \
                                        composition, mood."
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels (default: 1024, max: 2048)."
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels (default: 1024, max: 2048)."
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Inference steps — more = higher quality but slower (default: 50)."
                    },
                    "guidance": {
                        "type": "number",
                        "description": "Guidance scale — higher = more prompt adherence (default: 3.5)."
                    }
                },
                "required": ["prompt"]
            }),
        ),
    ]
}

/// Get just the tool names (for the system prompt "Available Tools" section).
pub fn all_tool_names() -> Vec<String> {
    all_tool_definitions()
        .iter()
        .map(|t| t.function.name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_tool_definitions_count() {
        let defs = all_tool_definitions();
        // 28 tools (excluding reply_request and refuse_request which are added separately)
        assert!(defs.len() >= 28, "Expected at least 28 tools, got {}", defs.len());
    }

    #[test]
    fn test_all_definitions_have_valid_structure() {
        for def in all_tool_definitions() {
            assert_eq!(def.tool_type, "function");
            assert!(!def.function.name.is_empty());
            assert!(!def.function.description.is_empty());
            assert!(def.function.parameters.is_object());
        }
    }

    #[test]
    fn test_tool_names_match_definitions() {
        let names = all_tool_names();
        let defs = all_tool_definitions();
        assert_eq!(names.len(), defs.len());
        for (name, def) in names.iter().zip(defs.iter()) {
            assert_eq!(name, &def.function.name);
        }
    }

    #[test]
    fn test_critical_tools_present() {
        let names = all_tool_names();
        assert!(names.contains(&"memory_tool".to_string()));
        assert!(names.contains(&"web_tool".to_string()));
        assert!(names.contains(&"run_command".to_string()));
        assert!(names.contains(&"codebase_read".to_string()));
        assert!(names.contains(&"reasoning_tool".to_string()));
        assert!(names.contains(&"operate_turing_grid".to_string()));
        assert!(names.contains(&"operate_synaptic_graph".to_string()));
    }
}
