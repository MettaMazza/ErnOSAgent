# Memory System

Ern-OS uses a 7-tier persistent memory system. All tiers are fields on `MemoryManager` (defined in `src/memory/mod.rs`) and are disk-persisted as JSON files under `data/`.

## Tier Overview

| Tier | Struct | File | Purpose |
|------|--------|------|---------|
| 1. Timeline | `TimelineStore` | `src/memory/timeline.rs` | Chronological message log with search |
| 2. Scratchpad | `ScratchpadStore` | `src/memory/scratchpad.rs` | Key-value pinned facts |
| 3. Lessons | `LessonStore` | `src/memory/lessons.rs` | Learned rules with confidence scores |
| 4. Synaptic | `SynapticGraph` | `src/memory/synaptic/mod.rs` | Knowledge graph with nodes, edges, plasticity |
| 5. Procedures | `ProcedureStore` | `src/memory/procedures.rs` | Named multi-step procedures |
| 6. Embeddings | `EmbeddingStore` | `src/memory/embeddings.rs` | Vector store for semantic search |
| 7. Consolidation | `ConsolidationEngine` | `src/memory/consolidation.rs` | Tracks context consolidation events |

## MemoryManager

```rust
pub struct MemoryManager {
    pub consolidation: ConsolidationEngine,
    pub timeline: TimelineStore,
    pub embeddings: EmbeddingStore,
    pub scratchpad: ScratchpadStore,
    pub lessons: LessonStore,
    pub procedures: ProcedureStore,
    pub synaptic: SynapticGraph,
}
```

### Key Methods

| Method | Description |
|--------|-------------|
| `new(data_dir)` | Initialise all 7 tiers, loading from disk |
| `recall_context(query, budget_tokens)` | Build context string with token budget allocation |
| `ingest_turn(role, content, session_id)` | Add a message to timeline |
| `status_summary()` | Human-readable status of all tiers |
| `clear()` | Reset all tiers |

## Context Recall Budget

`recall_context()` allocates the token budget across tiers:

| Tier | Allocation | Content |
|------|------------|---------|
| Scratchpad | 35% | Pinned key-value facts |
| Lessons | 25% | High-confidence rules (≥0.8) |
| Procedures (Skills) | 15% | Known skill names + descriptions (L0 loading) |
| Timeline | 15% | Recent 10 entries (120 char preview) |
| Knowledge Graph | 10% | Recent 5 nodes (id + layer) |

The budget is computed as `budget_tokens × 4` (approximating 4 chars per token).

## Tier Details

### 1. Timeline

- **Storage**: `Vec<TimelineEntry>` serialised to `data/timeline.json`
- **Entry fields**: `id`, `role`, `transcript`, `session_id`, `timestamp`
- **API**: `ingest()`, `recent(n)`, `search(query, limit)`, `entry_count()`
- **Search**: Case-insensitive substring matching on transcript

### 2. Scratchpad

- **Storage**: `Vec<ScratchpadEntry>` serialised to `data/scratchpad.json`
- **Entry fields**: `key`, `value`, `updated_at`
- **API**: `pin(key, value)`, `unpin(key)`, `get(key)`, `all()`, `count()`
- **Behaviour**: `pin()` upserts — if key exists, the value is replaced

### 3. Lessons

- **Storage**: `Vec<Lesson>` serialised to `data/lessons.json`
- **Entry fields**: `id`, `rule`, `source`, `confidence` (0.0–1.0), `times_applied`, `created_at`
- **API**: `add(rule, source, confidence)`, `add_if_new(rule, source, confidence)`, `remove(id)`, `search(query, limit)`, `high_confidence(threshold)`, `enforce_cap(max)`, `decay_unused(factor, min_confidence)`, `all()`, `count()`
- **Dedup**: `add_if_new()` uses Jaccard word-overlap similarity (≥0.7) to prevent duplicates
- **Decay**: `decay_unused()` reduces confidence of never-applied lessons by a factor, evicting below minimum. Triggered by the scheduler every 5 minutes.

### 4. Synaptic Graph

- **Storage**: Nodes + edges serialised to `data/synaptic.json`
- **Node fields**: `id`, `data` (HashMap), `layer`, `strength`, `last_activated`, `activation_count`
- **Edge fields**: `source`, `target`, `edge_type`, `weight`
- **API**: `upsert_node()`, `add_edge()`, `search_nodes()`, `get_node()`, `recent_nodes()`, `node_count()`, `edge_count()`, `layers()`
- **Plasticity**: `co_activate(a, b, delta)` strengthens both nodes; `decay_all(factor)` weakens inactive nodes
- **Submodules**: `plasticity.rs` (co-activation/decay), `query.rs` (search), `relationships.rs` (edge management)

### 5. Procedures (Self-Skills)

- **Storage**: `Vec<Procedure>` serialised to `data/procedures.json`
- **Procedure fields**: `id`, `name`, `description`, `steps: Vec<ProcedureStep>`, `success_count`, `last_used`
- **Step fields**: `tool`, `purpose`, `instruction`
- **API**: `add(name, steps)`, `add_if_new(name, description, steps)`, `refine(id, steps)`, `record_success(id)`, `record_success_by_name(name)`, `remove(id)`, `find_by_name(name)`, `all()`, `count()`
- **Dedup**: `add_if_new()` checks by name to prevent duplicate skills
- **Delayed reinforcement**: After a tool chain completes, the chain is stashed. On the NEXT user message, implicit approval/rejection signals are classified. Approved chains are auto-saved as procedures with `add_if_new()` and `record_success_by_name()`. Rejected chains are sent to the rejection buffer.
- **Tool access**: The `self_skills` tool provides CRUD operations for the agent to manage its own skills

### 6. Embeddings

- **Storage**: `Vec<EmbeddingEntry>` serialised to `data/embeddings.json`
- **Entry fields**: `id`, `text`, `source_type`, `vector: Vec<f32>`, `created_at`
- **API**: `insert(text, source_type, vector)`, `search(query_vector, limit)`, `count()`
- **Search**: Cosine similarity

### 7. Consolidation

- **Storage**: `Vec<ConsolidationEvent>` serialised to `data/consolidation.json`
- **Event fields**: `id`, `messages_consolidated`, `summary`, `context_length_at_time`, `timestamp`
- **API**: `record_consolidation()`, `history()`, `total_consolidations()`
- **Purpose**: Tracks when context was trimmed to respect model context window

## Persistence

All stores load from JSON on `MemoryManager::new()` and persist on every write operation. The data directory structure:

```
data/
├── timeline.json
├── scratchpad.json
├── lessons.json
├── synaptic.json
├── procedures.json
├── embeddings.json
├── consolidation.json
├── golden_buffer.json
├── rejection_buffer.json
├── scheduler.json
├── scheduler_history.json
├── snapshots/
│   └── snapshot_*.json
└── sessions/
    └── {uuid}.json
```
