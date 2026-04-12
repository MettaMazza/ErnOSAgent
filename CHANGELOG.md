# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2026-04-12

### Added
- **Mesh Network** — QUIC transport, ed25519/x25519 crypto, binary attestation, 4-layer content filter, distributed compute pool, knowledge sync, LoRA weight exchange, DHT, MeshFS, WASM sandbox, governance engine, censorship-resistant web proxy. 157 unit + 19 integration/E2E tests
- **14-Tab Dashboard** — expanded from 12 to 14 tabs with full backend wiring
  - **Checkpoints tab** — create/list/restore/delete system snapshots
  - **Autonomy tab** — unified agent state view with feature toggles (Observer, TTS, Scheduler, Mesh), per-tool enable/disable for autonomy scope, and live scrollable activity log
  - **Observer tab redesign** — 6-card stats grid (total/allowed/blocked/false positives/confirmed/pending), accuracy progress bar, scrollable audit log
  - **Learning tab expansion** — rejection count, observer audit count, distilled lessons count, full adapter version history table
  - **Mesh Network tab** — added connected peers table with trust badges, latency, and status indicators
  - **Tools tab** — per-tool chat-scope toggles with visual disabled state
- **API Expansion** — 9 new REST endpoints: `/api/observer/stats`, `/api/features`, `/api/features/{feature}/toggle`, `/api/tools/{name}/toggle` (chat scope), `/api/tools/{name}/toggle/autonomy` (autonomy scope), `/api/autonomy/status`, `/api/autonomy/log`, `/api/mesh/peers`
- **Independent Tool Toggle Scopes** — chat and autonomy tools toggled independently; both enforced at the pipeline/scheduler execution layer
- **Live Autonomy Activity Log** — `GET /api/autonomy/log` reads `activity.jsonl` and streams structured entries (cycle, timestamp, job name, tools used, summary, success, duration)
- **Discord Autonomy Channel** — configurable channel ID in Platforms tab; autonomous job results automatically forwarded to designated Discord channel
- **Feature Governance** — runtime enable/disable of all major subsystems from the dashboard (all default ON)
- **Scratchpad Write** — write to scratchpad directly from the Memory tab
- **Observer Training Pipeline** — dedicated Observer SFT with retroactive correctness labeling, plus 4 additional training methods (SimPO, KTO, DPO, GRPO)
- **28 Tools** — added scheduler_tool, autonomy_history, distill_knowledge, performance_review (was 24 in v1.0)
- **1083 tests** — 943 unit + 140 E2E (was 718 in v1.0)

### Fixed
- Observer toggle endpoint now returns JSON body with new state
- Session create endpoint field name alignment (`id` not `session_id`)
- ReactEvent pattern match exhaustiveness for new telemetry fields
- Pre-existing `context_length` missing from e2e_llama.rs test config

### Changed
- Source files: 227 → 230 `.rs` files
- Lines of code: ~51,600 → ~52,500
- Web UI: JavaScript IIFE-scoped controllers for all new tabs (ObserverStats, Checkpoints, AutonomyDashboard)

## [1.0.0] - 2026-04-10

### Added
- **ReAct Loop** — multi-turn reasoning with tool dispatch and error recovery
- **17-Rule Observer** — LLM-based quality audit with rejection-feedback loop
- **24 Tools** — codebase (8), shell, git, compiler, forge, memory (4), steering, interpretability, reasoning, web, download, synaptic graph, turing grid, reply_request
- **7-Tier Memory** — scratchpad, lessons, timeline, knowledge graph, procedures, embeddings, consolidation
- **Multi-Provider** — llama.cpp (primary), Ollama, LM Studio, HuggingFace, plus OpenAI-compatible cloud fallbacks (Claude, Groq, OpenRouter)
- **Self-Improvement Pipeline** — golden/preference buffer capture, SFT+ORPO training, LoRA adapter management with PEFT-compatible safetensors output
- **Architecture-Agnostic LoRA Engine** — auto-detects model dimensions from safetensors headers, per-layer LoRA initialization, Metal GPU accelerated, verified on Gemma 4 27B real weights
- **SAE Framework** — sparse autoencoder with ReLU/JumpReLU/TopK, safetensors import/export
- **Steering Vectors** — SAE feature steering (in-memory) + GGUF control vectors (transparent restart)
- **Interpretability** — neural snapshots, cognitive profiles, emotional state, safety alerts, divergence detection
- **Reasoning Traces** — persistent searchable thought traces in JSONL format
- **Mobile Engine** — UniFFI-exported Rust core with 4 inference modes (local, remote, hybrid, chain-of-agents), 90 tests
- **Desktop Relay** — WebSocket relay running full ReAct+Observer loop for mobile clients
- **Web UI** — Axum server with WebSocket chat, 12-tab dashboard, REST API
- **TUI** — ratatui terminal with chat, sidebar, model picker, steering panel
- **Feature Flags** — `discord`, `telegram`, `interp`, `mobile-native`, `all-platforms`
- **718+ tests** — 645 unit tests + 47 E2E tool tests + 12 E2E LoRA + 7 E2E learning + 7 E2E interpretability, 0 warnings
- **Kernel Protocols** — Zero Assumption, Anti-Sycophancy, Anti-Confabulation, Systemic Awareness, Tool Failure Recovery, System Capabilities Summary
- **Cross-platform** — macOS (Metal), Linux (CUDA), Windows (CUDA/CPU). All platform-specific code behind cfg guards
- **Credit Attribution** — @mettamazza attribution headers on all 185 source files with anti-removal notices

### Security
- Path containment system blocks directory traversal in all file operations
- Git operations locked to `ernosagent/self-edit` branch
- Shell command execution with configurable timeout and output capture limits
- 3 verified-safe `unsafe` blocks (required by Candle/memmap2 upstream crates)

### Known Limitations
- SAE weights are randomised demo data (requires ~24-48h GPU training for real feature decomposition)
- Steering vectors are placeholder GGUFs (requires contrast-pair training for real behaviour control)
- Neural snapshots use hash-derived activations until SAE training produces real feature decompositions
- Mobile llama.cpp FFI requires native library linking via `--features mobile-native`
- WebSocket relay transport partially implemented (needs tokio-tungstenite integration)

## [0.1.0] - 2026-04-07

Initial prototype with ReAct loop, Observer audit, and basic tool framework.
