// Ern-OS — Tool schema definitions (extracted for governance compliance).
// Each function returns a JSON schema for one tool.

pub fn shell_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "run_bash_command",
            "description": "Execute a shell command and return stdout/stderr",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "The command to execute" },
                    "working_dir": { "type": "string", "description": "Working directory (optional)" }
                },
                "required": ["command"]
            }
        }
    })
}

pub fn web_search_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web (8-tier waterfall: Brave → Serper → Tavily → SerpAPI → DuckDuckGo → Google → Wikipedia → Google News RSS) or visit a URL directly to extract its text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["search", "visit"], "description": "Action: 'search' (default) or 'visit' to fetch a URL" },
                    "query": { "type": "string", "description": "Search query (for action=search)" },
                    "url": { "type": "string", "description": "URL to visit (for action=visit)" }
                },
                "required": ["query"]
            }
        }
    })
}

pub fn memory_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "memory",
            "description": "Interact with the 7-tier cognitive memory system. Actions: recall, status, consolidate, search, reset",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["recall", "status", "consolidate", "search", "reset"] },
                    "query": { "type": "string", "description": "Search query (for recall/search)" }
                },
                "required": ["action"]
            }
        }
    })
}

pub fn scratchpad_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "scratchpad",
            "description": "Pin/unpin persistent notes. Actions: pin, unpin, list, get",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["pin", "unpin", "list", "get"] },
                    "key": { "type": "string" },
                    "value": { "type": "string" }
                },
                "required": ["action"]
            }
        }
    })
}

pub fn synaptic_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "synaptic",
            "description": "Interact with the synaptic knowledge graph. Actions: store, store_relationship, search, beliefs, recent, stats, layers, co_activate",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["store", "store_relationship", "search", "beliefs", "recent", "stats", "layers", "co_activate"] },
                    "concept": { "type": "string" },
                    "data": { "type": "object" },
                    "target": { "type": "string" },
                    "edge_type": { "type": "string" },
                    "layer": { "type": "string" },
                    "limit": { "type": "integer" }
                },
                "required": ["action"]
            }
        }
    })
}

pub fn timeline_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "timeline",
            "description": "Query the conversation timeline. Actions: recent, search, session",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["recent", "search", "session"] },
                    "query": { "type": "string" },
                    "limit": { "type": "integer" },
                    "session_id": { "type": "string" }
                },
                "required": ["action"]
            }
        }
    })
}

pub fn lessons_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "lessons",
            "description": "Manage learned behavioral rules. Actions: add, remove, list, search",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["add", "remove", "list", "search"] },
                    "rule": { "type": "string" },
                    "confidence": { "type": "number" },
                    "id": { "type": "string" },
                    "query": { "type": "string" }
                },
                "required": ["action"]
            }
        }
    })
}

pub fn self_skills_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "self_skills",
            "description": "Manage reusable procedural skills (workflows you've learned). Actions: list, view, create, refine, delete",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["list", "view", "create", "refine", "delete"] },
                    "name": { "type": "string", "description": "Skill name (used for create/view, also accepted for refine/delete as lookup)" },
                    "description": { "type": "string", "description": "Skill description (for create)" },
                    "steps": { "type": "array", "description": "Array of step objects with tool+instruction" },
                    "id": { "type": "string", "description": "Skill ID (for refine/delete — alternative to name)" }
                },
                "required": ["action"]
            }
        }
    })
}

pub fn learning_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "learning",
            "description": "Manage the self-learning pipeline. Actions: status, buffer_stats, trigger_training, list_adapters, sleep",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["status", "buffer_stats", "trigger_training", "list_adapters", "sleep"] },
                    "method": { "type": "string", "description": "Training method (sft/dpo/kto)" }
                },
                "required": ["action"]
            }
        }
    })
}

pub fn steering_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "steering",
            "description": "Manage cognitive steering vectors. Actions: list, activate, deactivate, status",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["list", "activate", "deactivate", "status"] },
                    "name": { "type": "string", "description": "Steering vector name" },
                    "strength": { "type": "number", "description": "Vector strength 0.0-2.0" }
                },
                "required": ["action"]
            }
        }
    })
}

pub fn interpretability_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "interpretability",
            "description": "SAE feature analysis and activation inspection. Actions: top_features, encode, snapshot",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["top_features", "encode", "snapshot"] },
                    "input": { "type": "string", "description": "Text to analyse" }
                },
                "required": ["action"]
            }
        }
    })
}

pub fn codebase_search_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "codebase_search",
            "description": "Recursively search files in a directory for content matching a query",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search pattern (text or regex)" },
                    "path": { "type": "string", "description": "Directory to search (default: '.')" },
                    "max_results": { "type": "integer", "description": "Max results (default: 20)" }
                },
                "required": ["query"]
            }
        }
    })
}

pub fn file_read_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to the file to read" }
                },
                "required": ["path"]
            }
        }
    })
}

pub fn file_write_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write content to a file (creates parent directories if needed)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Path to write to" },
                    "content": { "type": "string", "description": "Content to write" }
                },
                "required": ["path", "content"]
            }
        }
    })
}

pub fn browser_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "browser",
            "description": "Interactive browser (visible window when headed mode enabled). Every action returns DOM context showing available links, buttons, inputs, and headings so you can choose valid selectors. Actions: open, click, type, navigate, wait, extract, screenshot, evaluate, close, list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["open", "click", "type", "navigate", "wait", "extract", "screenshot", "evaluate", "close", "list"], "description": "Browser action to perform" },
                    "page_id": { "type": "string", "description": "Page identifier (returned by 'open', omit for 'open')" },
                    "url": { "type": "string", "description": "URL for open/navigate" },
                    "selector": { "type": "string", "description": "CSS selector for click/type/wait/extract" },
                    "text": { "type": "string", "description": "Text for type action" },
                    "script": { "type": "string", "description": "JavaScript for evaluate" },
                    "attribute": { "type": "string", "description": "HTML attribute for extract (omit for innerText)" },
                    "timeout_ms": { "type": "integer", "description": "Timeout for wait (default: 5000)" }
                },
                "required": ["action"]
            }
        }
    })
}

pub fn create_artifact_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "create_artifact",
            "description": "Create a rich markdown document (report, analysis, plan, code reference) that persists and is rendered as an interactive card in the UI. Use this for substantial output — anything longer than a few paragraphs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": { "type": "string", "description": "Document title" },
                    "content": { "type": "string", "description": "Full markdown content" },
                    "artifact_type": { "type": "string", "enum": ["report", "plan", "analysis", "code"], "description": "Type of artifact" }
                },
                "required": ["title", "content"]
            }
        }
    })
}

pub fn generate_image_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a text prompt using local Flux model. Returns a markdown image tag.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": { "type": "string", "description": "Detailed image description" },
                    "width": { "type": "integer", "description": "Width in pixels (default: 1024)" },
                    "height": { "type": "integer", "description": "Height in pixels (default: 1024)" },
                    "steps": { "type": "integer", "description": "Inference steps (default: 30)" },
                    "guidance": { "type": "number", "description": "Guidance scale (default: 3.5)" }
                },
                "required": ["prompt"]
            }
        }
    })
}

pub fn spawn_sub_agent_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "spawn_sub_agent",
            "description": "Spawn an isolated sub-agent with restricted tools to handle a focused task. Returns a summary — parent context stays clean.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": { "type": "string", "description": "What the sub-agent should accomplish" },
                    "tools": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "List of tool names the sub-agent may use"
                    },
                    "max_turns": { "type": "integer", "description": "Maximum turns (default: 5)" }
                },
                "required": ["task", "tools"]
            }
        }
    })
}

pub fn codebase_edit_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "codebase_edit",
            "description": "Edit your own source code. Actions: 'patch' (find-replace), 'insert' (before/after anchor), 'multi_patch' (multiple patches), 'delete' (remove file). All edits are auto-checkpointed for rollback. Protected files (governance, secrets) are blocked.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["patch", "insert", "multi_patch", "delete"], "description": "The edit operation" },
                    "path": { "type": "string", "description": "File path to edit" },
                    "find": { "type": "string", "description": "Text to find (for patch)" },
                    "replace": { "type": "string", "description": "Replacement text (for patch)" },
                    "anchor": { "type": "string", "description": "Anchor text (for insert)" },
                    "content": { "type": "string", "description": "Content to insert (for insert)" },
                    "position": { "type": "string", "enum": ["before", "after"], "description": "Insert before or after anchor (default: after)" },
                    "patches": { "type": "array", "items": { "type": "object", "properties": { "find": { "type": "string" }, "replace": { "type": "string" } } }, "description": "Array of patches (for multi_patch)" }
                },
                "required": ["action", "path"]
            }
        }
    })
}

pub fn system_recompile_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "system_recompile",
            "description": "Recompile the Ern-OS engine. Runs an 8-stage pipeline: test gate, warning gate, build, changelog, resume state, binary stage, activity log, hot-swap. If tests or warnings fail, fix with codebase_edit and retry autonomously.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    })
}

pub fn checkpoint_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "checkpoint",
            "description": "Manage file checkpoints. Actions: 'list' (show all), 'rollback' (restore a file), 'prune' (remove old snapshots).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["list", "rollback", "prune"], "description": "Checkpoint operation" },
                    "id": { "type": "string", "description": "Checkpoint ID (for rollback)" },
                    "max_age_hours": { "type": "integer", "description": "Max age in hours (for prune, default 48)" }
                },
                "required": ["action"]
            }
        }
    })
}

pub fn system_logs_tool_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "system_logs",
            "description": "Read-only access to your own error logs and self-edit audit trail. Actions: 'tail' (last N lines), 'errors' (grep ERROR/WARN), 'search' (pattern match), 'self_edits' (recent codebase edit log).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["tail", "errors", "search", "self_edits"], "description": "Log operation" },
                    "lines": { "type": "integer", "description": "Number of lines (for tail, default 50)" },
                    "max": { "type": "integer", "description": "Max results (for errors/search, default 20-30)" },
                    "pattern": { "type": "string", "description": "Search pattern (for search)" }
                },
                "required": ["action"]
            }
        }
    })
}
