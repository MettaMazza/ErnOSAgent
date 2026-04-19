<p align="center">
  <h1 align="center">Ern-OS</h1>
  <p align="center"><strong>Sovereign AI agent engine. Local-first. Written in Rust.</strong></p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> ·
    <a href="#architecture">Architecture</a> ·
    <a href="#tools">Tools</a> ·
    <a href="#memory-system">Memory</a> ·
    <a href="docs/">Documentation</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/language-Rust-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/tests-378%20passing-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/warnings-0-brightgreen?style=flat-square" />
  <img src="https://img.shields.io/badge/license-MIT-blue?style=flat-square" />
</p>

---

Ern-OS is a high-performance AI agent engine that runs entirely on your hardware. No cloud. No telemetry. No API keys required. Point it at any GGUF model via `llama-server`, and you get a full agentic system: a dual-layer inference engine with ReAct reasoning, a 22-tool executor, a 7-tier persistent memory system, an observer audit pipeline, autonomous learning, and a 12-tab WebUI dashboard — all compiled into a single Rust binary.

Created by [@mettamazza](https://github.com/mettamazza)

## Quick Start

```bash
# 1. Clone
git clone https://github.com/mettamazza/ErnosAgent.git
cd ErnosAgent

# 2. Place a GGUF model
mkdir -p models
# Copy your model to models/ (e.g., gemma-4-27b-it-Q4_K_M.gguf)

# 3. Configure (edit ern-os.toml with your model path)
# 4. Run
cargo run --release
```

Opens `http://localhost:3000` — the full dashboard with chat, memory explorer, tool logs, training controls, and more.

### Requirements

| Dependency | Purpose |
|-----------|---------|
| **Rust 1.75+** | Build the engine |
| **[llama-server](https://github.com/ggerganov/llama.cpp)** | Serve GGUF models locally |
| A GGUF model file | The brain (any model works — Gemma, Llama, Mistral, etc.) |

Optional: Kokoro TTS (voice), Flux (image generation), code-server (VS Code IDE) — each auto-launches if available.

## Architecture

```
User ──→ WebUI (localhost:3000)
           │
           ├─ WebSocket: Chat / Voice / Video
           │
    ┌──────┴──────────────────────────────────┐
    │         Dual-Layer Inference Engine      │
    │                                          │
    │  Layer 1 (L1): Fast single-shot reply    │
    │  ─ 7 core tools, streaming, sub-second   │
    │                                          │
    │  Layer 2 (L2): ReAct reasoning loop      │
    │  ─ 22 tools, multi-turn, autonomous      │
    │  ─ Model-driven turn management          │
    │  ─ Observer audit on every reply          │
    ├──────────────────────────────────────────┤
    │  22-Tool Executor                        │
    │  shell · web · files · browser · memory  │
    │  sub-agents · artifacts · codebase edit  │
    │  image gen · SAE · steering · learning   │
    ├──────────────────────────────────────────┤
    │  7-Tier Persistent Memory                │
    │  timeline · scratchpad · lessons ·       │
    │  synaptic · procedures · embeddings ·    │
    │  consolidation                           │
    ├──────────────────────────────────────────┤
    │  Learning Pipeline                       │
    │  golden buffer · rejection buffer ·      │
    │  LoRA · GRPO · sleep consolidation       │
    ├──────────────────────────────────────────┤
    │  Provider Trait (model-neutral)           │
    │  llamacpp · ollama · openai-compatible   │
    └──────────────────────────────────────────┘
```

### Dual-Layer Inference

**Layer 1** handles straightforward requests — the model gets a single inference call with 7 core tools. If it needs more than one tool call, or if the task requires reasoning, it escalates to Layer 2.

**Layer 2** runs a full ReAct loop: the model reasons, calls tools, observes results, and continues until it decides it's done. Turn management is model-driven — the model requests extensions when it needs more turns. An Observer audits every reply for quality, hallucination, and completeness before it reaches the user.

### Model Neutrality

Ern-OS doesn't care what model you run. The `Provider` trait abstracts all inference:

- **llamacpp** — local GGUF models via `llama-server` (default, recommended)
- **ollama** — Ollama-managed models
- **openai-compatible** — any OpenAI-compatible API endpoint

## Tools

22 native tools, all executing locally:

| Tool | What It Does |
|------|-------------|
| `run_bash_command` | Execute shell commands with working directory control |
| `web_search` | Search the web and visit URLs (Brave, Tavily, SearXNG, Serper) |
| `file_read` / `file_write` | Read and write files on the local filesystem |
| `codebase_search` | Recursive grep across directories |
| `codebase_edit` | Find-replace, insert, multi-patch with auto-checkpoint |
| `browser` | Headless Chrome — open, navigate, click, type, screenshot |
| `memory` | Store, recall, and search across the memory system |
| `scratchpad` / `timeline` / `lessons` / `synaptic` | Direct access to individual memory tiers |
| `self_skills` | Create, store, and execute learned skill procedures |
| `spawn_sub_agent` | Launch a child agent with scoped tool access |
| `create_artifact` | Generate structured documents and reports |
| `generate_image` | Text-to-image via local Flux server |
| `learning` | Trigger LoRA training, manage preference buffers |
| `interpretability` | SAE feature analysis, activation inspection |
| `steering` | Runtime steering vectors for behaviour modification |
| `system_recompile` | Hot-recompile the engine from its own source |
| `system_logs` | Read and search runtime logs |
| `checkpoint` | Create named restore points during codebase edits |

## Memory System

7 tiers of persistent memory, all stored locally as JSON:

| Tier | Purpose | Persistence |
|------|---------|-------------|
| **Timeline** | Chronological event log — every tool call, every interaction | Append-only |
| **Scratchpad** | Working memory for the current task | Session-scoped |
| **Lessons** | Distilled learnings from past mistakes and successes | Permanent |
| **Synaptic** | High-signal knowledge graph with weighted connections | Permanent |
| **Procedures** | Executable skill recipes synthesised from experience | Permanent |
| **Embeddings** | Vector store for semantic recall | Permanent |
| **Consolidation** | Sleep-cycle memory compression and pruning | Scheduled |

Memory is automatically recalled at inference time and injected into the system prompt. The consolidation engine runs on a configurable schedule to compress, prune, and strengthen memory based on access patterns.

## Observer

Every Layer 2 reply passes through the Observer before reaching the user. The Observer is a separate inference call that audits for:

- **Hallucination** — claims not supported by tool results
- **Sycophancy** — agreeing with the user when evidence says otherwise
- **Laziness** — incomplete, vague, or placeholder responses
- **Tool ignorance** — describing what it would do instead of using tools

If the Observer rejects a reply, the model gets structured feedback and tries again. This is not a filter — it's a quality loop.

## WebUI Dashboard

12 tabs accessible from `localhost:3000`:

| Tab | What's There |
|-----|-------------|
| **Chat** | Streaming chat with thinking blocks, tool execution cards, artifacts |
| **Memory** | Browse and search all 7 memory tiers |
| **Tools** | Live tool execution log with timing |
| **Training** | Golden/rejection buffer stats, trigger LoRA training |
| **Interpretability** | SAE feature analysis, activation heatmaps |
| **Steering** | Apply runtime steering vectors |
| **Logs** | Live system logs with filtering |
| **Identity** | View and edit the agent's persona |
| **Agents** | Manage sub-agent configurations |
| **Scheduler** | Cron-like job scheduling (health checks, consolidation, learning) |
| **Codes** | Embedded VS Code IDE (via code-server) |
| **Settings** | Platform adapters, provider config, system controls |

## Learning Pipeline

Ern-OS has a built-in learning pipeline for continuous self-improvement:

- **Golden Buffer** — captures high-quality interaction pairs for SFT fine-tuning
- **Rejection Buffer** — captures Observer-rejected responses for preference training (DPO/GRPO)
- **Sleep Consolidation** — scheduled memory compression, lesson extraction, and skill synthesis
- **LoRA Training** — native Candle-based LoRA on Apple Silicon (Metal-accelerated)

## Customisation

### Identity / Persona

Create `data/prompts/identity.md` to give your agent a custom personality. If absent, a default Ern-OS persona is used. The identity file supports full markdown and is injected into the system prompt at inference time.

### Configuration

All configuration lives in `ern-os.toml`:

```toml
[general]
active_provider = "llamacpp"
data_dir = "data"

[llamacpp]
server_binary = "/opt/homebrew/bin/llama-server"
port = 8080
model_path = "./models/your-model.gguf"
n_gpu_layers = 999

[observer]
enabled = true

[web]
port = 3000
open_browser = true

[prompt]
thinking_enabled = true
```

See [docs/configuration.md](docs/configuration.md) for the full reference.

## Project Stats

| Metric | Value |
|--------|-------|
| Language | Rust (Edition 2021) |
| Source files | 144 `.rs` files |
| Lines of code | ~20,000 |
| Tests | 378 passing (302 lib + 76 e2e) |
| Test failures | 0 |
| Compiler warnings | 0 |
| Tools | 22 |
| API endpoints | 68 REST + 3 WebSocket (chat, voice, video) |
| Dashboard tabs | 12 |
| Memory tiers | 7 |
| Providers | 3 (llamacpp, ollama, openai-compatible) |
| Auto-launching services | 4 (WebUI, Kokoro TTS, Flux image gen, code-server) |

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | System design, data flow, module responsibilities |
| [Configuration](docs/configuration.md) | All config options with types and defaults |
| [Memory System](docs/memory.md) | 7-tier memory architecture and consolidation |
| [Inference Pipeline](docs/inference.md) | Dual-layer engine, ReAct loop, observer audit |
| [Learning Pipeline](docs/learning.md) | LoRA, GRPO, sleep consolidation, preference training |
| [Tools](docs/tools.md) | 22-tool registry with schemas and parallel execution |
| [Interpretability](docs/interpretability.md) | SAE, feature analysis, steering vectors |
| [Provider Interface](docs/providers.md) | Provider trait, implementations, model neutrality |
| [Testing](docs/testing.md) | Test structure, coverage, running tests |

## License

MIT — do whatever you want with it.
