# Architecture

## System Overview

Ern-OS is a **WebUI-centric** agent engine with autonomous learning. The WebUI is the single public interface — all external consumers (Discord bots, Telegram adapters, mobile apps) connect via WebSocket or REST. No external client imports engine internals directly.

```
┌─────────────────────────────────────────────────────┐
│                   WebUI Hub (ws.rs)                  │
│   WebSocket (/ws)  ·  Static Frontend (index.html)   │
│   70 REST API endpoints · 12 Dashboard views         │
├─────────────────────────────────────────────────────┤
│                Internal Engine Services              │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌────────┐ ┌──────────┐  │
│  │ Inference │ │ Observer │ │ Tools  │ │ Sessions │  │
│  │  Engine   │ │  Audit   │ │27 Tools│ │ Manager  │  │
│  └──────────┘ └──────────┘ └────────┘ └──────────┘  │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐  │
│  │ Memory   │ │ Learning │ │  Interpretability    │  │
│  │ 7 Tiers  │ │ Pipeline │ │  (SAE + Steering)    │  │
│  └──────────┘ └──────────┘ └──────────────────────┘  │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐  │
│  │Scheduler │ │ Platform │ │  Agents & Teams      │  │
│  │ Cron     │ │ Adapters │ │  Custom personas     │  │
│  └──────────┘ └──────────┘ └──────────────────────┘  │
│                                                      │
│  ┌──────────┐ ┌──────────┐                           │
│  │ Kokoro   │ │ code-    │  Auto-starting services   │
│  │ TTS      │ │ server   │  (VS Code IDE)            │
│  └──────────┘ └──────────┘                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────────┐  │
│  │ Flux     │ │ Voice/   │ │  Sub-Agent            │  │
│  │ ImageGen │ │ Video WS │ │  Isolation            │  │
│  └──────────┘ └──────────┘ └──────────────────────┘  │
├─────────────────────────────────────────────────────┤
│              Provider Layer (trait-based)             │
│      llamacpp.rs  │  ollama.rs  │  openai_compat.rs  │
└─────────────────────────────────────────────────────┘
         ↓                ↓                ↓
    llama-server      Ollama API     Any OpenAI API
```

## Module Map

All source lives under `src/`. 18 top-level modules:

| Module | Path | Purpose |
|--------|------|---------|
| `agents` | `src/agents/` | Agent registry and team management |
| `config` | `src/config/mod.rs` | TOML config loading with defaults (includes CodesConfig, TtsConfig) |
| `inference` | `src/inference/` | Dual-layer engine: fast_reply.rs, react_loop.rs, react_observer.rs, router.rs |
| `interpretability` | `src/interpretability/` | SAE training (trainer.rs), feature extraction, steering bridge, live analysis, divergence, snapshots |
| `learning` | `src/learning/` | LoRA (6 loss functions, adapters, optimizer), GRPO (generation, rewards, training), sleep cycles, distillation, teacher |
| `logging` | `src/logging/mod.rs` | Structured tracing, rotating file appender |
| `memory` | `src/memory/` | 7-tier persistent memory: timeline, lessons, procedures, scratchpad, synaptic (graph + plasticity + relationships + query), embeddings, consolidation |
| `model` | `src/model/mod.rs` | ModelSpec struct (auto-derived from provider) |
| `observer` | `src/observer/` | Response audit (mod.rs), insight extraction (insights.rs), rule system (rules.rs), parser (parser.rs), skill synthesis (skills.rs) |
| `platform` | `src/platform/` | Platform adapter trait, registry, router — connects Discord/Telegram as WebSocket clients |
| `prompt` | `src/prompt/mod.rs` | System prompt management, identity loading |
| `provider` | `src/provider/` | Provider trait + 3 implementations: llamacpp (with embed), ollama, openai_compat, stream_parser |
| `scheduler` | `src/scheduler/` | Cron engine: job definitions (job.rs), persistent store (store.rs), 8 built-in system jobs |
| `session` | `src/session/mod.rs` | JSON-backed session CRUD with pin, archive, fork, search, reactions |
| `steering` | `src/steering/` | Activation steering vectors (vectors.rs), server interface (server.rs) |
| `tools` | `src/tools/` | 27-tool registry: schema definitions + tool implementation files |
| `web` | `src/web/` | Axum server (70 routes), WebSocket handler (chat + voice + video), 14 handler modules, state, static frontend |

## Data Flow: User Message → Response

```
1. User sends JSON over WebSocket to ws.rs
2. ws.rs parses message type ("chat")
3. Memory recall: MemoryManager.recall_context() builds system context
   └─ Includes: scratchpad (35%), lessons (25%), skills (15%),
      timeline (15%), knowledge graph (10%)
4. Message ingested into Timeline memory
5. Layer 1: Provider.chat() with layer1_tools (18 tools)
6. Stream consumed via consume_silently()
   ├─ TextDelta → buffered (NOT sent to user yet)
   ├─ ToolCall "start_react_system" → escalate to Layer 2
   └─ Other ToolCall → execute_tool_with_state() → re-infer
7. Observer audit gate (if enabled):
   ├─ Provider.chat_sync() with audit prompt
   ├─ parse_verdict() extracts JSON {approved, score, reason}
    ├─ Approved → deliver to user
    │   ├─ capture_approved() → GoldenBuffer (SFT training data)
    │   └─ spawn_insight_extraction() → Lessons memory tier
    └─ Rejected → inject feedback → silent retry
        └─ capture_rejection() → RejectionBuffer (DPO training pairs)
 8. Tool chain stashed in PendingToolChain for delayed reinforcement
 9. Response archived to Timeline memory
10. On NEXT user message, delayed reinforcement evaluates implicit feedback:
    ├─ Approved ("great", "now...", "thanks") → auto-create procedure + golden buffer
    ├─ Rejected ("wrong", "not what I asked") → rejection buffer
    └─ Neutral → no action
11. "done" signal sent to client
```

## Shared State

All components share state via `AppState` (defined in `src/web/state.rs`):

```rust
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub model_spec: Arc<ModelSpec>,
    pub memory: Arc<RwLock<MemoryManager>>,
    pub sessions: Arc<RwLock<SessionManager>>,
    pub provider: Arc<dyn Provider>,
    pub golden_buffer: Arc<RwLock<GoldenBuffer>>,
    pub rejection_buffer: Arc<RwLock<RejectionBuffer>>,
    pub scheduler: Arc<RwLock<JobStore>>,
    pub agents: Arc<RwLock<AgentRegistry>>,
    pub teams: Arc<RwLock<TeamRegistry>>,
    pub browser: Arc<RwLock<BrowserState>>,
    pub platforms: Arc<RwLock<PlatformRegistry>>,
    pub mutable_config: Arc<RwLock<AppConfig>>,
}
```

- `Arc<RwLock<MemoryManager>>` enables concurrent reads with exclusive writes
- `Arc<dyn Provider>` enables runtime provider selection
- `Arc<RwLock<AgentRegistry>>` manages custom agent personas
- `Arc<RwLock<BrowserState>>` lazily initialized headless Chromium instance
- `Arc<RwLock<PlatformRegistry>>` manages Discord/Telegram adapter connections
- `Arc<RwLock<AppConfig>>` (`mutable_config`) — runtime-updatable config for Settings UI changes

## Auto-Starting Services

On boot, `main.rs` spawns background services before starting the WebUI:

1. **Kokoro TTS** — `maybe_start_kokoro()` spawns `start-kokoro.py` on port 8880
2. **Flux Image Server** — `maybe_start_flux()` spawns `flux_server.py` on port 8890
3. **code-server** — `maybe_start_code_server()` spawns the VS Code binary on port 8443
4. **WebUI Hub** — Axum server on port 3000

All are supervised processes that log startup/failure via tracing.

## Real-Time Communication

### Chat (WebSocket `/ws`)
Standard bidirectional text/tool chat via the dual-layer inference engine.

### Voice Calls (WebSocket `/ws/voice`)
Mic audio → Gemma 4 (native audio via mmproj) → Kokoro TTS → speaker. Session state shared with chat.

### Video Calls (WebSocket `/ws/video`)
Camera frames + mic audio → Gemma 4 (vision + audio) → TTS response. Frame rate limited to 1fps.

## Sub-Agent Isolation

The `spawn_sub_agent` tool creates an isolated ReAct loop with:
- Restricted tool whitelist (prevents recursive spawning)
- Separate conversation context (no pollution of parent)
- Summary-only return (only the final answer flows back)

## Parallel Tool Execution

When the model emits multiple `ToolCalls` in a single response, they are dispatched concurrently via `futures::join_all`. All results are collected and injected as tool messages before the next inference turn.

## REST API (70 routes)

Organized by handler module in `src/web/handlers/`:

| Handler Module | Endpoints | Purpose |
|---------------|-----------|---------|
| `sessions.rs` | 12 routes | CRUD, search, pin, archive, export, fork, message delete, reactions |
| `memory.rs` | 6 routes | Stats + all 7 memory tiers |
| `system.rs` | 19 routes | Health, status, models, tools, training, interpretability (features/snapshots/sae), steering, learning (status/adapters/sleep-history), observer, logs, self-edits, checkpoints, prompts get/put, factory reset |
| `scheduler.rs` | 5 routes | Job CRUD, toggle, history |
| `agents.rs` | 8 routes | Agent/team CRUD |
| `onboarding.rs` | 3 routes | Status, save profile, complete |
| `api_keys.rs` | 3 routes | GET/PUT API key management, env loading |
| `tts.rs` | 2 routes | Synthesize, status |
| `codes.rs` | 1 route | code-server health |
| `platforms.rs` | 5 routes | List, config get/put, connect, disconnect, platform ingest |
| `content.rs` | 4 routes | Static file serving (index.html, app.css, app.js, images) |

## Background Scheduler (Cron Engine)

The scheduler (`src/scheduler/`) is a job-driven cron engine with 15-second tick intervals. Jobs are persistent (`data/scheduler.json`) and support three schedule types:

- **Interval** — execute every N seconds
- **Cron** — standard cron expressions (via `cron` crate)
- **Once** — execute at a specific UTC datetime

### 8 Built-in System Jobs

| Job | Schedule | Task |
|-----|----------|------|
| `sleep_cycle` | every 5m | Drain buffers → LoRA training |
| `lesson_decay` | every 5m | Hebbian forgetting (0.98 factor) |
| `memory_consolidate` | every 1h | Tier consolidation check |
| `snapshot_capture` | every 30m | Neural activation snapshot |
| `synaptic_prune` | every 2h | Weak edge decay |
| `buffer_flush` | every 10m | Persist training buffers |
| `log_rotate` | daily midnight | Remove logs >7 days old |
| `health_check` | every 60s | Self-diagnostic |

## File Size Governance

Per `agents/rust_code_governance.md`:
- All files ≤500 lines (excluding the operational kernel prompt)
- All functions ≤50 lines
- All `impl` blocks ≤15 methods
- Zero TODOs, zero `unimplemented!()`, zero `todo!()`
