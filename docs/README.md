# Ern-OS

**High-performance, model-neutral Rust AI agent engine.**

Created by [@mettamazza](https://github.com/mettamazza) · License: MIT

---

## What Is Ern-OS?

Ern-OS is a local-first AI agent engine written in Rust. It runs any GGUF model via `llama-server`, exposes a WebSocket-based chat interface, and provides a full agentic pipeline: dual-layer inference, tool execution, observer audit, 7-tier persistent memory, autonomous learning (LoRA + sleep cycles), self-skill synthesis, SAE interpretability, a built-in VS Code IDE, headless browser, platform adapters, and TTS — all on your own hardware.

## Quick Start

```bash
cd /path/to/Ern-OS
cargo run --release
```

This starts `llama-server` with your configured model, auto-starts Kokoro TTS, Flux image generation, and code-server, waits for health, then opens `http://localhost:3000` in your browser.

## Requirements

- **Rust** 1.75+ (Edition 2021)
- **llama-server** (from [llama.cpp](https://github.com/ggerganov/llama.cpp)) — installed via Homebrew or built from source
- A GGUF model file in `models/`

## Configuration

All config lives in `ern-os.toml` at the project root. If absent, defaults are used.

See [configuration.md](configuration.md) for the full reference.

## Architecture

See [architecture.md](architecture.md) for the complete system design.

```
User → WebUI (localhost:3000)
         ↓ WebSocket (chat / voice / video)
    ┌────────────────────────────────┐
    │  Dual-Layer Inference Engine   │
    │  Layer 1: Fast Reply (7 tools) │
    │  Layer 2: ReAct Loop (22 tools)│
    ├────────────────────────────────┤
    │  Observer Audit + Insights     │
    ├────────────────────────────────┤
    │  22-Tool Executor (shell, web, │
    │  memory, browser, image, sub-  │
    │  agent, artifacts, files, SAE) │
    ├────────────────────────────────┤
    │  7-Tier Memory + Self-Skills   │
    ├────────────────────────────────┤
    │  Learning Pipeline + Scheduler │
    ├────────────────────────────────┤
    │  Platform Adapters + TTS       │
    │  Voice + Video Calls (WS)      │
    ├────────────────────────────────┤
    │  code-server (VS Code IDE)     │
    ├────────────────────────────────┤
    │  Provider Trait (model-neutral) │
    │  llamacpp │ ollama │ openai    │
    └────────────────────────────────┘
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | System design, data flow, module responsibilities |
| [Configuration](configuration.md) | All config options with types and defaults |
| [Memory System](memory.md) | 7-tier memory: timeline, scratchpad, lessons, synaptic, procedures, embeddings, consolidation |
| [Inference Pipeline](inference.md) | Dual-layer engine, ReAct loop, observer audit |
| [Learning Pipeline](learning.md) | LoRA, GRPO, sleep consolidation, preference training |
| [Tools](tools.md) | 22-tool registry with schemas, parallel execution |
| [Interpretability](interpretability.md) | SAE, feature analysis, steering vectors, divergence tracking |
| [Provider Interface](providers.md) | Provider trait, implementations, model neutrality |
| [Testing](testing.md) | Test structure, coverage, running tests |

## Project Stats

| Metric | Value |
|--------|-------|
| Language | Rust (Edition 2021) |
| Source files | 144 `.rs` files |
| Lines of code | ~20,000 (src only) |
| Tests | 378 passing (302 lib + 76 e2e) |
| Test failures | 0 |
| Compiler warnings | 0 |
| Modules | 18 top-level |
| Tools | 22 |
| API endpoints | 68 REST routes + 3 WebSocket (chat, voice, video) |
| Dashboard views | 12 tabs |
| Auto-starting services | 4 (WebUI, Kokoro TTS, Flux Image, code-server) |
