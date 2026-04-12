# ErnOSAgent — Architecture Reference

Created by [@mettamazza](https://github.com/mettamazza) | 227 Rust source files | ~58,000 lines | 950+ tests

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│  ┌──────────┐  ┌──────────────────────────────────────────────┐ │
│  │ TUI      │  │ Web UI (localhost:3000)                      │ │
│  │ ratatui  │  │ Axum + WebSocket + HTML/CSS/JS Dashboard     │ │
│  └──────────┘  └──────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                     Application Core                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Prompt   │  │ Session  │  │ Memory   │  │ Learning     │   │
│  │ Assembly │  │ Manager  │  │ Manager  │  │ (Buffers +   │   │
│  │ (kernel) │  │          │  │ (7-tier) │  │  Teacher +   │   │
│  └──────────┘  └──────────┘  └──────────┘  │  LoRA)       │   │
│                                             └──────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     ReAct Engine                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Reason → Act → Observe (loop until reply_request)       │   │
│  │                                                          │   │
│  │  ┌──────────┐                    ┌──────────────────┐    │   │
│  │  │ Tool     │                    │ Observer Audit    │    │   │
│  │  │ Executor │                    │ (17-rule LLM     │    │   │
│  │  │ (24)     │                    │  quality gate)    │    │   │
│  │  └──────────┘                    └──────────────────┘    │   │
│  │                                                          │   │
│  │  Golden Buffer ← (PASS, 0 rejections)                    │   │
│  │  Preference Buffer ← (PASS after rejections)             │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     Interpretability                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ SAE      │  │ Feature  │  │ Snapshot │  │ Divergence   │   │
│  │ (Sparse  │  │ Dict     │  │ Builder  │  │ Detector     │   │
│  │  AE)     │  │ (labels) │  │          │  │ (valence)    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     Providers                                    │
│  ┌──────────┐  ┌────────┐  ┌──────────┐  ┌──────────┐        │
│  │ llama.cpp│  │ Ollama │  │ LM Studio│  │ HF API   │        │
│  │ PRIMARY  │  │        │  │          │  │          │        │
│  └──────────┘  └────────┘  └──────────┘  └──────────┘        │
│       │              ┌──────────────────────────┐              │
│  ┌──────────────┐    │ OpenAI-compat (fallback) │              │
│  │ Steering     │    │ OpenAI, Claude, Groq,    │              │
│  │ Vectors      │    │ OpenRouter               │              │
│  └──────────────┘    └──────────────────────────┘              │
├─────────────────────────────────────────────────────────────────┤
│                     Platforms                                    │
│  ┌─────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ TUI │  │ Discord  │  │ Telegram │  │ WhatsApp │           │
│  └─────┘  └──────────┘  └──────────┘  └──────────┘           │
├─────────────────────────────────────────────────────────────────┤
│                     LoRA Training Engine                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Architecture auto-detection from safetensors headers      │   │
│  │ Per-layer LoRA dims (handles GQA, MoE, alternating attn)  │   │
│  │ Metal GPU (Apple) / CUDA / CPU training                   │   │
│  │ SFT (golden) + ORPO (preference) training modes           │   │
│  │ PEFT-compatible safetensors adapter output                 │   │
│  └──────────────────────────────────────────────────────────┘   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     Mesh Network (enabled by default)           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ QUIC     │  │ Crypto   │  │ Trust    │  │ Compute Pool │   │
│  │ Transport│  │ ed25519  │  │ Pipeline │  │ + Equality   │   │
│  │ (quinn)  │  │ x25519   │  │          │  │              │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Knowledge│  │ Weight   │  │ DHT +    │  │ Governance   │   │
│  │ Sync     │  │ Exchange │  │ MeshFS   │  │ + Web Proxy  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### Core Engine

| Module | Path | Description | Tests |
|--------|------|-------------|:-----:|
| config | `src/config/` | Hierarchical configuration (env → file → defaults) | 8 |
| main | `src/main.rs` | Binary entry point, startup orchestration | — |
| lib | `src/lib.rs` | Library crate (re-exports all modules) | — |

### Inference Pipeline

| Module | Path | Description | Tests |
|--------|------|-------------|:-----:|
| provider | `src/provider/` | Trait-based provider abstraction | 2 |
| provider::llamacpp | `src/provider/llamacpp.rs` | **PRIMARY** — llama-server lifecycle, SSE streaming, tool call parsing | 3 |
| provider::ollama | `src/provider/ollama/` | Ollama API client | 1 |
| provider::lmstudio | `src/provider/lmstudio.rs` | LM Studio API client | 1 |
| provider::huggingface | `src/provider/huggingface.rs` | HuggingFace Inference API | 1 |
| provider::openai_compat | `src/provider/openai_compat.rs` | OpenAI-compatible endpoint adapter (OpenAI, Claude, Groq, OpenRouter) | — |
| inference | `src/inference/` | Context window management, stream processing | 10 |
| model | `src/model/` | Model spec auto-derivation, registry, routing | 6 |

### Reasoning & Tools

| Module | Path | Description | Tests |
|--------|------|-------------|:-----:|
| react | `src/react/loop/` | Reason→Act→Observe loop engine + learning hooks | 2 |
| react::reply | `src/react/reply.rs` | `reply_request` tool definition (mandatory loop exit) | 2 |
| tools::executor | `src/tools/executor.rs` | Tool execution dispatcher | 5 |
| tools::schema | `src/tools/schema.rs` | `ToolCall`, `ToolResult` types, reply extraction | 5 |
| tools::tool_schemas | `src/tools/tool_schemas.rs` | All 24 tool definitions with JSON Schema params | — |
| observer | `src/observer/` | 17-rule LLM-based quality audit | 6 |

### Memory System

| Module | Path | Tier | Tests |
|--------|------|:----:|:-----:|
| memory | `src/memory/mod.rs` | — | 3 |
| scratchpad | `src/memory/scratchpad.rs` | 1 | 3 |
| lessons | `src/memory/lessons.rs` | 2 | 4 |
| timeline | `src/memory/timeline.rs` | 3 | 3 |
| knowledge_graph | `src/memory/knowledge_graph.rs` | 4 | 3 |
| procedures | `src/memory/procedures.rs` | 5 | 3 |
| embeddings | `src/memory/embeddings.rs` | 6 | 3 |
| consolidation | `src/memory/consolidation.rs` | 7 | 3 |
| synaptic | `src/memory/synaptic/` | — | 4+ |

### Learning & Self-Improvement

| Module | Path | Description | Tests |
|--------|------|-------------|:-----:|
| learning::buffers | `src/learning/buffers.rs` | JSONL golden + preference data capture | 8 |
| learning::teacher | `src/learning/teacher.rs` | Training lifecycle orchestrator (state machine) | 6 |
| learning::lora::mod | `src/learning/lora/mod.rs` | LoRA config, tokenization, estimation utilities | — |
| learning::lora::forward | `src/learning/lora/forward.rs` | Architecture-agnostic forward pass with GQA, per-layer dim probing | 15+ |
| learning::lora::weights | `src/learning/lora/weights.rs` | Per-layer LoRA VarMap from safetensors headers | — |
| learning::lora::training | `src/learning/lora/training.rs` | SFT + ORPO training loops (Metal/CPU) | — |
| learning::lora::loss | `src/learning/lora/loss.rs` | Cross-entropy + ORPO loss computation | 15 |
| learning::lora::optimizer | `src/learning/lora/optimizer.rs` | AdamW with warmup + cosine decay | — |
| learning::lora::adapters | `src/learning/lora/adapters.rs` | PEFT-compatible safetensors + adapter_config.json export | — |
| learning::manifest | `src/learning/manifest.rs` | Adapter version management, promote/rollback | 11 |
| learning::distill | `src/learning/distill.rs` | Observer → lesson auto-generation | 7 |

### Interpretability

| Module | Path | Description | Tests |
|--------|------|-------------|:-----:|
| interpretability::sae | `src/interpretability/sae.rs` | Sparse Autoencoder (ReLU/JumpReLU/TopK) | 3 |
| interpretability::features | `src/interpretability/features.rs` | Feature dictionary (cognitive, safety, emotion) | 6 |
| interpretability::extractor | `src/interpretability/extractor.rs` | Activation extraction | 1 |
| interpretability::snapshot | `src/interpretability/snapshot.rs` | Per-turn neural activity snapshot | — |
| interpretability::divergence | `src/interpretability/divergence.rs` | Internal/output state divergence detection | 7 |
| interpretability::steering_bridge | `src/interpretability/steering_bridge.rs` | Feature-level steering interface | 3 |
| interpretability::trainer | `src/interpretability/trainer.rs` | SAE training infrastructure | — |

### Prompt Engineering

| Module | Path | Description | Tests |
|--------|------|-------------|:-----:|
| prompt::core | `src/prompt/core.rs` | Operational kernel (Zero Assumption, Anti-Sycophancy, Systemic Awareness, System Capabilities) | 4 |
| prompt::context | `src/prompt/context.rs` | Dynamic context assembly | 4 |
| prompt::identity | `src/prompt/identity.rs` | Persona loading from file | 4 |

### Steering & Control

| Module | Path | Description | Tests |
|--------|------|-------------|:-----:|
| steering::vectors | `src/steering/vectors.rs` | Control vector GGUF management (load, scale, compose) | 3 |
| steering::server | `src/steering/server.rs` | llama-server restart + model hot-swap | 2 |

### User Interface

| Module | Path | Description | Tests |
|--------|------|-------------|:-----:|
| web::server | `src/web/server.rs` | Axum web server with open_browser (macOS/Linux/Windows) | — |
| web::ws | `src/web/ws/` | WebSocket chat + session management | — |
| web::routes | `src/web/routes/` | REST API endpoints (status, memory, steering, tools, etc.) | — |
| web::state | `src/web/state.rs` | Shared application state | 3 |

### Platform Adapters

| Module | Path | Feature Flag | Tests |
|--------|------|:------------:|:-----:|
| platform::adapter | `src/platform/adapter.rs` | — | 2 |
| platform::discord | `src/platform/discord.rs` | `discord` | — |
| platform::telegram | `src/platform/telegram.rs` | `telegram` | — |
| platform::whatsapp | `src/platform/whatsapp.rs` | — | — |
| platform::custom | `src/platform/custom.rs` | — | — |

### Mobile Engine

| Module | Path | Description | Tests |
|--------|------|-------------|:-----:|
| mobile::engine | `src/mobile/engine.rs` | Core mobile engine with 4 inference modes | 20+ |
| mobile::llama_ffi | `src/mobile/llama_ffi.rs` | llama.cpp C FFI bindings with Metal/OpenCL detection | 5 |
| mobile::native_build | `src/mobile/native_build.rs` | CMake build configuration for NDK/Xcode | 10+ |
| mobile::model_manager | `src/mobile/model_manager.rs` | On-device model lifecycle | 5+ |
| mobile::provider_* | `src/mobile/provider_*.rs` | Local, relay, hybrid, and chain providers | 20+ |
| mobile::desktop_discovery | `src/mobile/desktop_discovery.rs` | mDNS + QR code + manual pairing | 5+ |
| mobile::uniffi_scaffolding | `src/mobile/uniffi_scaffolding.rs` | UniFFI export layer for Kotlin/Swift | — |

### Advanced Compute

| Module | Path | Description | Tests |
|--------|------|-------------|:-----:|
| computer::alu | `src/computer/alu.rs` | Arithmetic logic unit | — |
| computer::turing_grid | `src/computer/turing_grid/` | Turing grid computation engine | 10+ |

### Mesh Network (default feature)

| Module | Path | Description | Tests |
|--------|------|-------------|:-----:|
| network::peer_id | `src/network/peer_id.rs` | Binary-hash peer identity | 7 |
| network::wire | `src/network/wire.rs` | MessagePack wire protocol (40+ message types) | 10 |
| network::crypto | `src/network/crypto.rs` | ed25519/x25519/ChaCha20-Poly1305 | 9 |
| network::transport | `src/network/transport.rs` | QUIC transport with `quinn` | 5 |
| network::trust | `src/network/trust.rs` | Binary attestation TrustGate | 8 |
| network::sanctions | `src/network/sanctions.rs` | Violation tracking SanctionEngine | 8 |
| network::neutralise | `src/network/neutralise.rs` | IntegrityWatchdog + self-destruct | 6 |
| network::content_filter | `src/network/content_filter.rs` | 4-layer security filter | 11 |
| network::discovery | `src/network/discovery.rs` | mDNS/Bootstrap/Gossip discovery | 8 |
| network::identity | `src/network/identity.rs` | Peer identity profiles | 4 |
| network::capabilities | `src/network/capabilities.rs` | Hardware/model capability registry | 8 |
| network::compute | `src/network/compute.rs` | Distributed compute pool + equality | 11 |
| network::knowledge_sync | `src/network/knowledge_sync.rs` | Knowledge exchange + PII stripping | 11 |
| network::weight_exchange | `src/network/weight_exchange.rs` | LoRA adapter sharing | 5 |
| network::code_propagation | `src/network/code_propagation.rs` | Code patch distribution | 8 |
| network::dht | `src/network/dht.rs` | Kademlia DHT | 6 |
| network::mesh_fs | `src/network/mesh_fs.rs` | Distributed chunked file system | 6 |
| network::sandbox | `src/network/sandbox.rs` | WASM sandbox execution | 3 |
| network::governance | `src/network/governance.rs` | Ban voting, alerts, OSINT | 7 |
| network::web_proxy | `src/network/web_proxy.rs` | Censorship-resistant proxy | 6 |
| network::human_mesh | `src/network/human_mesh.rs` | Human P2P PlatformAdapter | 6 |
| network::mesh_loop | `src/network/mesh_loop.rs` | Central event loop + MeshCoordinator | 5 |
| web::routes::mesh | `src/web/routes/mesh.rs` | Mesh API endpoints | — |

---

## Data Flow

### ReAct Loop + Learning Capture

```
User Message
    │
    ▼
Prompt Assembly (kernel + context + memory + tools)
    │
    ▼
LLM Inference (llama-server SSE)
    │
    ├── Tool Call → Tool Executor → Re-inject result → Loop
    │
    └── reply_request → Observer Audit
                            │
                ┌───────────┴───────────┐
                │                       │
              PASS                    FAIL
                │                       │
        ┌───────┴───────┐          Re-prompt with
        │               │          feedback (loop)
    0 rejections    >0 rejections        │
        │               │          On eventual PASS:
    Golden Buffer   Preference Buffer  ──┘
        │               │
        └───────┬───────┘
                │
        Teacher checks thresholds
                │
        LoRA Training (SFT/ORPO on Metal GPU)
                │
        Adapter Manifest → Hot-Swap
```

### Memory Tiers

```
Tier 1: Scratchpad   ─── In-memory working context (current session)
Tier 2: Lessons      ─── Distilled rules from past interactions (JSON)
Tier 3: Timeline     ─── Chronological event archive (JSON per-day)
Tier 4: Knowledge Graph ─ Neo4j entity-relation store with decay
Tier 5: Procedures   ─── Learned multi-step workflows (JSON)
Tier 6: Embeddings   ─── Semantic vector search
Tier 7: Consolidation ── Cross-tier compression & pruning
```

---

## Maturity Matrix

> Honest assessment of each subsystem's implementation status.

| Subsystem | Status | Detail |
|-----------|:------:|--------|
| ReAct Loop | **Production** | Full Reason→Act→Observe loop, tool dispatch, error recovery |
| Observer (17-rule audit) | **Production** | LLM-based audit with structured JSON parsing, fail-closed |
| 7-Tier Memory | **Production** | All tiers implemented, persistence verified, KG via Neo4j |
| Provider Abstraction | **Production** | 4 local backends + cloud fallbacks, auto-derivation, SSE streaming |
| Prompt Assembly | **Production** | Dynamic 3-layer prompt: kernel + context + identity |
| Session Management | **Production** | Persistence, history, multi-session |
| Steering Vectors | **Infrastructure** | Vector loading/management works; vectors themselves are placeholder GGUFs until contrast-pair training |
| Learning Buffers | **Production** | JSONL crash-safe, lock-free counters, drain/read_all |
| Teacher Orchestrator | **Production** | State machine, training lock, real LoRA calls |
| LoRA Training | **Production** | Architecture-agnostic, real weights, Metal GPU, per-layer dims, PEFT export. E2E verified on Gemma 4 27B |
| ORPO Loss | **Production** | Mathematically correct log-sigmoid formulation |
| Adapter Manifest | **Production** | Version tracking, promote/rollback, pruning, health checks |
| Distillation | **Production** | Failure pattern → lesson generation with dedup |
| SAE | **Infrastructure** | Encode/decode pipeline works; uses demo weights, needs GPU training |
| Feature Dictionary | **Reference** | 40+ labeled features; labels predefined from Anthropic's taxonomy |
| Divergence Detection | **Production** | Valence-based divergence with safety-refusal exemption |
| Snapshot Pipeline | **Infrastructure** | Full pipeline works; activations are hash-based until SAE is trained |
| Web UI | **Production** | Dashboard with 12 tabs (incl. Mesh Network), WebSocket chat, REST API |
| TUI | **Production** | Full interactive terminal UI with ratatui |
| Tool Implementations | **Production** | All 24 tools implemented and E2E tested |
| Mobile Engine | **Production** | Full 4-mode engine with 90 tests, needs cross-compilation for device deployment |
| Mesh Transport | **Production** | QUIC via quinn, binary attestation, ed25519/x25519 crypto. 176 tests |
| Mesh Security | **Production** | TrustGate, SanctionEngine, IntegrityWatchdog, 4-layer content filter |
| Mesh Compute | **Production** | Distributed job pool with equality enforcement, FIFO scheduling |
| Mesh Knowledge | **Production** | Lesson sync with PII stripping, weight exchange, code propagation |
| Mesh Storage | **Production** | Kademlia DHT, chunked MeshFS, WASM sandbox |
| Mesh Governance | **Production** | Phase-dependent ban voting, emergency alerts, web proxy |

---

## Test Coverage

| Suite | Tests | Runtime | Requires |
|-------|:-----:|--------:|----------|
| Unit tests (all modules) | 775 | ~1.3s | Nothing |
| Mesh unit tests | 157 | ~8s | Nothing (default) |
| Mesh integration tests | 7 | ~1.2s | Nothing (default) |
| Mesh E2E tests | 12 | ~1.3s | Nothing (default) |
| E2E Tools | 47 | ~0.3s | Nothing |
| E2E LoRA | 12 | ~0.4s | Nothing |
| E2E Learning | 7 | ~46s | Model weights |
| E2E Interpretability | 7 | ~0.03s | Nothing |
| E2E llama | 4 | ~5s | llama-server + model |
| **Total** | **950+** | — | — |

---

## File Structure

```
ErnOSAgent/
├── src/
│   ├── computer/             # ALU + Turing grid computation
│   ├── config/               # Hierarchical configuration (env → file → defaults)
│   ├── inference/            # Context window, stream processing
│   ├── interpretability/     # SAE, features, snapshots, divergence, steering bridge
│   ├── learning/             # Buffers, teacher, LoRA engine, manifest, distillation
│   ├── logging/              # Structured JSON logging
│   ├── memory/               # 7-tier memory + synaptic graph
│   ├── mobile/               # Mobile engine, FFI, providers, UniFFI
│   ├── model/                # Model spec, registry, routing
│   ├── network/              # Mesh network (default feature)
│   │   ├── peer_id.rs        #   Peer identity (binary hash)
│   │   ├── wire.rs           #   MessagePack wire protocol
│   │   ├── crypto.rs         #   ed25519/x25519/ChaCha20
│   │   ├── transport.rs      #   QUIC transport (quinn)
│   │   ├── trust.rs          #   Binary attestation trust gate
│   │   ├── compute.rs        #   Distributed compute pool
│   │   ├── knowledge_sync.rs #   Knowledge exchange + PII stripping
│   │   ├── dht.rs            #   Kademlia distributed hash table
│   │   ├── mesh_fs.rs        #   Chunked distributed file system
│   │   ├── governance.rs     #   Ban voting, alerts, resource ads
│   │   ├── web_proxy.rs      #   Censorship-resistant proxy
│   │   └── mesh_loop.rs      #   Central coordinator event loop
│   ├── observer/             # 17-rule quality audit
│   ├── platform/             # Platform adapters (Discord, Telegram, WhatsApp, Custom)
│   ├── prompt/               # System prompt assembly (core + context + identity)
│   ├── provider/             # Inference backends (llama.cpp, Ollama, LM Studio, HF, OpenAI-compat)
│   ├── react/                # ReAct loop engine
│   ├── scheduler/            # Job execution + persistent store
│   ├── session/              # Session lifecycle & persistence
│   ├── steering/             # Control vector management + hot-swap
│   ├── tools/                # 24-tool execution framework
│   ├── voice/                # Voice input infrastructure
│   └── web/                  # Axum web server, WebSocket, REST API, dashboard
├── tests/
│   ├── e2e_llama.rs          # E2E: model spec, inference, ReAct, Neo4j
│   ├── e2e_learning.rs       # E2E: buffers → teacher → LoRA → manifest (requires model weights)
│   ├── e2e_lora.rs           # E2E: config, VarMap, ORPO formula, Metal device
│   ├── e2e_tools.rs          # E2E: all 24 tools
│   ├── e2e_interpretability.rs  # E2E: SAE, features, divergence, steering
│   ├── mesh_integration.rs   # Integration: mesh security, compute, knowledge, governance
│   └── mesh_e2e.rs           # E2E: 12 multi-instance mesh tests
├── static/                   # Web UI HTML/CSS/JS assets
├── models/                   # Model weights (gitignored)
├── adapters/                 # Trained LoRA adapters (gitignored)
├── ARCHITECTURE.md           # This file
├── DEVELOPMENT.md            # Development guide
├── MEMORY.md                 # Memory system reference
├── VERIFICATION.md           # Manual verification checklist
├── CHANGELOG.md              # Version history
├── Cargo.toml
├── LICENSE
└── README.md
```
