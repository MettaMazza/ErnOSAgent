# Tools

Ern-OS provides 27 tools across two layers. Tool schemas are defined in `src/tools/schema.rs` and `src/tools/schema_definitions.rs`. Execution is handled by `src/web/tool_dispatch.rs` which routes all tool calls through `AppState`.

## Layer 1 Tools (18 tools)

Available during fast reply (Layer 1). Defined by `layer1_tools()`.

| Tool | Description |
|------|-------------|
| `start_react_system` | Escalate to the ReAct agentic loop with an objective and optional plan |
| `propose_plan` | Create a detailed plan for user approval before executing |
| `run_bash_command` | Execute a shell command |
| `web_search` | Search the web via 8-engine waterfall or visit a URL |
| `file_read` | Read file contents |
| `file_write` | Write content to a file |
| `codebase_search` | Recursively search files in a directory |
| `browser` | Interactive headless browser (10 actions) |
| `memory` | 7-tier memory: recall, status, consolidate, search, reset |
| `scratchpad` | Pin/unpin persistent notes |
| `timeline` | Query conversation timeline |
| `lessons` | Manage learned behavioral rules |
| `create_artifact` | Create rich markdown documents |
| `generate_image` | Generate images locally via Flux model |
| `steering` | Manage cognitive steering vectors |
| `interpretability` | SAE feature analysis and activation inspection |
| `learning` | Manage the self-learning pipeline |
| `system_logs` | Read-only access to error logs and self-edit audit trail |

Layer 1 decides whether to answer directly or escalate. If the task is simple, it responds immediately. If complex, it calls `start_react_system` to enter Layer 2.

## Layer 2 Tools (25 tools)

Available during the ReAct loop (Layer 2). Defined by `layer2_tools()`. Includes these tools (note: not all L1 tools carry over — L2 has its own curated set):

### Control Tools

| Tool | Description |
|------|-------------|
| `reply_request` | Submit final answer to the user (terminates loop) |
| `refuse_request` | Explicitly refuse the request with a reason (terminates loop) |
| `extend_turns` | Request additional reasoning turns when the budget is low |

### Execution Tools

| Tool | File | Description |
|------|------|-------------|
| `run_bash_command` | `shell.rs` | Execute a shell command with optional working directory |
| `web_search` | `web_search.rs` | 8-engine waterfall: Brave → Serper → Tavily → SerpAPI → DuckDuckGo → Google → Wikipedia → Google News RSS |
| `codebase_search` | `codebase_search.rs` | Recursively search files in a directory for content matches |
| `file_read` | `file_read.rs` | Read the contents of a file |
| `file_write` | `file_write.rs` | Write content to a file (creates parent dirs) |
| `browser` | `browser_tool.rs` | Persistent headless browser with 10 interactive actions |

### Memory Tools

All memory tools route through `tool_dispatch.rs` which accesses `AppState.memory`.

| Tool | Dispatch | Actions |
|------|----------|---------|
| `memory` | `dispatch_memory` | `recall`, `status`, `search`, `reset` |
| `scratchpad` | `dispatch_scratchpad` | `pin`, `unpin`, `get`, `list` |
| `synaptic` | `dispatch_synaptic` | `store`, `store_relationship`, `search`, `beliefs`, `recent`, `stats`, `layers`, `co_activate` |
| `timeline` | `dispatch_timeline` | `recent`, `search`, `session` |
| `lessons` | `dispatch_lessons` | `add`, `remove`, `list`, `search` |
| `self_skills` | `dispatch_self_skills` | `list`, `view`, `create`, `refine`, `delete` |

### Learning & Introspection Tools

| Tool | File | Actions |
|------|------|---------|
| `learning` | `learning_tool.rs` | `status`, `buffer_stats`, `trigger_training`, `list_adapters`, `sleep` |
| `steering` | `steering_tool.rs` | `list`, `activate`, `deactivate`, `status` |
| `interpretability` | `interpretability_tool.rs` | `snapshot`, `top_features`, `encode`, `probe`, `labeled_features` |

### Content & Generation Tools

| Tool | File | Description |
|------|------|-------------|
| `create_artifact` | `artifact_tool.rs` | Create persistent rich documents (reports, analysis, plans) |
| `generate_image` | `image_tool.rs` | Generate images locally via Flux model |
| `spawn_sub_agent` | `sub_agent_tool.rs` | Spawn isolated agents with restricted tool sets |

### Self-Coding & Recompilation Tools

| Tool | File | Description |
|------|------|-------------|
| `codebase_edit` | `codebase_edit.rs` | Edit source files: patch, insert, multi_patch, delete (auto-checkpointed, containment-gated) |
| `system_recompile` | `compiler.rs` | 8-stage self-recompile: test → warning gate → build → changelog → resume → binary stage → log → hot-swap |
| `checkpoint` | `checkpoint.rs` | Manage file snapshots: list, rollback, prune |

## Tool Call Flow

```
1. Model emits tool_call in streaming response
2. ws.rs detects ToolCall via StreamEvent::ToolCalls
3. execute_tool_with_state(state, &tc) dispatches:
   ├─ "run_bash_command" → shell::run_command(cmd, wd)
   ├─ "web_search" → web_search::search(query) / visit(url)
   ├─ "memory" → dispatch_memory(state, args)
   ├─ "scratchpad" → dispatch_scratchpad(state, args)
   ├─ "synaptic" → dispatch_synaptic(state, args)
   ├─ "timeline" → dispatch_timeline(state, args)
   ├─ "lessons" → dispatch_lessons(state, args)
   ├─ "self_skills" → dispatch_self_skills(state, args)
   ├─ "learning" → learning_tool::execute(args, state)
   ├─ "steering" → steering_tool::execute(args)
   ├─ "interpretability" → interpretability_tool::execute(args)
   ├─ "codebase_search" → codebase_search::execute(args)
   ├─ "file_read" → file_read::execute(args)
   ├─ "file_write" → file_write::execute(args)
   ├─ "browser" → browser_tool::execute(&state.browser, action, args)
   ├─ "create_artifact" → artifact_tool::execute(args)
   ├─ "generate_image" → image_tool::execute(args)
   ├─ "spawn_sub_agent" → sub_agent spawn + isolated loop
   ├─ "codebase_edit" → dispatch_codebase_edit(state, args)
   ├─ "system_recompile" → compiler::run_recompile()
   ├─ "checkpoint" → dispatch_checkpoint(state, args)   └─ unknown → "Unknown tool: {name}"
4. ToolResult { tool_call_id, name, output, success } returned
5. Result injected into message history for next iteration
```

## Parallel Tool Execution

When the model emits multiple tool calls in a single turn, they are executed concurrently via `futures::join_all`. Results are collected and injected together, enabling efficient multi-tool information gathering.

## Web Search: 8-Engine Waterfall

`web_search::search(query)` tries each search provider in order until one succeeds:

1. **Brave Search** — requires `BRAVE_API_KEY`
2. **Serper** — requires `SERPER_API_KEY`
3. **Tavily** — requires `TAVILY_API_KEY`
4. **SerpAPI** — requires `SERPAPI_API_KEY`
5. **DuckDuckGo** — no key required (free fallback)
6. **Google Web Scrape** — no key required
7. **Wikipedia** — no key required (final knowledge fallback)
8. **Google News RSS** — no key required (news fallback)

All implementations in `src/tools/search_providers.rs`.

## Browser Tool Details

`browser_tool.rs` uses the `chromiumoxide` crate for headless Chromium control via CDP.

- Auto-detects Chrome at `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`
- Lazy initialization — Chrome only launches on first use
- Persistent pages — up to 5 concurrent pages managed via `HashMap`
- 10 Actions: `open`, `click`, `type`, `navigate`, `wait`, `extract`, `screenshot`, `evaluate`, `close`, `list`

## Voice & Video Endpoints

Real-time communication via WebSocket:

| Endpoint | Handler | Description |
|----------|---------|-------------|
| `/ws/voice` | `voice.rs` | Audio-only calls: mic → Gemma 4 native audio → Kokoro TTS |
| `/ws/video` | `video.rs` | Audio+video calls: camera frames + mic → Gemma 4 vision+audio → TTS |

## Tool Modules

```
src/tools/
├── schema.rs               — Tool JSON schemas (layer1_tools, layer2_tools)
├── executor.rs             — ToolCall/ToolResult types
├── shell.rs                — Shell command execution
├── web_search.rs           — Web search orchestrator
├── search_providers.rs     — 6 search engine implementations
├── browser_tool.rs         — Persistent headless Chromium browser (10 actions)
├── memory_tool.rs          — Memory recall/search
├── scratchpad_tool.rs      — Pinned notes
├── synaptic_tool.rs        — Knowledge graph operations
├── timeline_tool.rs        — Conversation history
├── lessons_tool.rs         — Behavioral rules
├── self_skills_tool.rs     — Procedural skill management
├── learning_tool.rs        — Training pipeline control
├── steering_tool.rs        — Steering vector activation
├── interpretability_tool.rs — SAE feature analysis
├── codebase_search.rs      — Recursive directory search
├── file_read.rs            — File reading
├── file_write.rs           — File writing
├── artifact_tool.rs        — Persistent artifact creation
├── image_gen_tool.rs       — Local Flux image generation
└── sub_agent_tool.rs       — Isolated sub-agent spawning (src/inference/sub_agent.rs)
```
