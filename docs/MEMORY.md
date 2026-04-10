# Memory System

ErnOSAgent uses a 7-tier memory architecture that gives the agent persistent, queryable context across sessions. This document covers each tier, how they interact, and the data flow.

---

## Overview

```
              ┌───────────────────────────────────┐
              │        MemoryManager              │
              │   recall_context() → system prompt │
              │   ingest_turn()   → persist data   │
              └────────┬──────────────────────────┘
                       │
    ┌──────────────────┼──────────────────────┐
    │                  │                      │
    ▼                  ▼                      ▼
┌────────┐     ┌────────────┐          ┌───────────┐
│ Tier 1 │     │  Tier 2    │          │  Tier 3   │
│Scratch │     │  Lessons   │          │ Timeline  │
│  pad   │     │            │          │           │
└────────┘     └────────────┘          └───────────┘
    ▼                  ▼                      ▼
┌────────┐     ┌────────────┐          ┌───────────┐
│ Tier 4 │     │  Tier 5    │          │  Tier 6   │
│  KG    │     │ Procedures │          │Embeddings │
│(Neo4j) │     │            │          │           │
└────────┘     └────────────┘          └───────────┘
                       │
                       ▼
              ┌────────────────┐
              │    Tier 7      │
              │ Consolidation  │
              └────────────────┘
```

---

## Tier 1: Scratchpad

**Purpose:** Working memory for the current session.

| Property | Value |
|----------|-------|
| Persistence | Session only (lost on restart) |
| Backend | In-memory `Vec<String>` |
| Budget | Up to 40% of recall context |

The scratchpad holds short-lived notes, intermediate reasoning results, and user corrections from the current session. It is the fastest tier to query (in-memory) and the first to be pruned when context budget is exceeded.

### Operations

```rust
scratchpad.add("User prefers concise answers");
scratchpad.entries();    // → &[String]
scratchpad.clear();
```

---

## Tier 2: Lessons

**Purpose:** Distilled behavioral rules learned from past interactions.

| Property | Value |
|----------|-------|
| Persistence | Permanent (JSON file) |
| Backend | `lessons.json` |
| Budget | Up to 30% of recall context |

Lessons are high-level patterns extracted from conversation history. They answer: "What should I remember about how this user works, thinks, or communicates?"

### Format

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "rule": "User prefers code examples over explanations",
    "source_session": "session-2026-04-07",
    "created_at": "2026-04-07T12:00:00Z",
    "confidence": 0.85
  }
]
```

### Operations

```rust
lessons.add("User prefers code examples", session_id, 0.85)?;
lessons.search("coding style", 5);  // → Vec<Lesson>
lessons.all();
lessons.remove(id)?;
```

---

## Tier 3: Timeline

**Purpose:** Chronological archive of all conversation turns.

| Property | Value |
|----------|-------|
| Persistence | Permanent (daily JSON files) |
| Backend | `timeline/YYYY-MM-DD.json` |
| Budget | Up to 20% of recall context |

Every user message and assistant response is archived with timestamps. This provides raw material for:
- Continuity Recovery (greeting returning users by context)
- Lesson extraction
- Trend analysis

### Format

```json
{
  "date": "2026-04-07",
  "entries": [
    {
      "id": "...",
      "timestamp": "2026-04-07T18:30:00Z",
      "role": "user",
      "content": "What is the borrow checker?",
      "session_id": "session-123"
    },
    {
      "id": "...",
      "timestamp": "2026-04-07T18:30:05Z",
      "role": "assistant",
      "content": "The borrow checker is Rust's...",
      "session_id": "session-123"
    }
  ]
}
```

### Operations

```rust
timeline.record(role, content, session_id)?;
timeline.recent(n);        // → last N entries
timeline.search(query, n); // → fuzzy match
timeline.entry_count();
```

---

## Tier 4: Knowledge Graph

**Purpose:** Structured entity-relation storage for factual knowledge.

| Property | Value |
|----------|-------|
| Persistence | Permanent |
| Backend | Neo4j (`bolt://localhost:7687`) |
| Budget | Up to 10% of recall context |

The Knowledge Graph stores entities (people, concepts, tools, projects) and the relationships between them. It supports:

- **Weighted relations** — Each relation has a `weight` (0.0–1.0) representing confidence/relevance
- **Temporal decay** — Weights decay over time; stale relations are pruned
- **Reinforcement** — Re-encountered facts increase in weight
- **Fuzzy search** — Find entities by approximate label match

### Schema

```cypher
(:Entity {
  uid: String,         -- UUID
  label: String,       -- Human-readable name
  entity_type: String, -- "person", "concept", "tool", "project", etc.
  properties: String,  -- JSON-encoded metadata
  created_at: DateTime,
  updated_at: DateTime
})

(:Entity)-[:RELATION {
  relation_type: String,  -- "knows", "uses", "created", etc.
  weight: Float,          -- 0.0 to 1.0
  last_seen: DateTime
}]->(:Entity)
```

### Operations

```rust
let kg = KnowledgeGraph::connect(uri, user, pass, db).await?;

// Create/update entities
let id = kg.upsert_entity("Rust", "language", &json!({"paradigm": "systems"})).await?;

// Create/update relations
kg.upsert_relation(&alice_id, &rust_id, "knows", 0.9).await?;

// Recall with decay filtering
let relations = kg.recall("Rust", 10, 0.01).await?;

// Global decay (run periodically)
let pruned = kg.decay_all(0.95).await?; // Multiply all weights by 0.95
```

### Resilience

The Knowledge Graph is treated as optional. If Neo4j is unreachable:

```rust
if !mgr.kg_available() {
    // System continues without KG tier
    // Status shows "KG: offline"
    // No crash, no panic, no stub data
}
```

---

## Tier 5: Procedures

**Purpose:** Learned multi-step workflows.

| Property | Value |
|----------|-------|
| Persistence | Permanent (JSON file) |
| Backend | `procedures.json` |

Procedures are sequences of tool calls that successfully solved a class of problem. When a similar problem is encountered, the procedure is recalled to guide the agent's tool-calling strategy.

### Format

```json
[
  {
    "id": "...",
    "name": "Research a Technical Topic",
    "steps": [
      {"tool": "web_search", "purpose": "Find authoritative sources"},
      {"tool": "file_read", "purpose": "Check local documentation"},
      {"tool": "reply_request", "purpose": "Synthesize findings"}
    ],
    "success_count": 5,
    "last_used": "2026-04-07T18:00:00Z"
  }
]
```

---

## Tier 6: Embeddings

**Purpose:** Semantic similarity search across all persisted content.

| Property | Value |
|----------|-------|
| Persistence | Permanent (vector store) |
| Backend | Local file or provider embedding API |

Embeddings allow the system to find semantically relevant past interactions, lessons, or procedures even when exact keyword matches fail.

---

## Tier 7: Consolidation

**Purpose:** Cross-tier compression and garbage collection.

Consolidation runs periodically to:

1. **Merge duplicate lessons** — Multiple similar rules → one authoritative rule
2. **Extract lessons from timeline** — Recurring patterns → new lessons
3. **Prune stale KG relations** — Weight < threshold → delete
4. **Compress timeline** — Old entries → summaries
5. **Update embeddings** — New content → re-index

---

## Context Recall

When the user sends a message, `MemoryManager::recall_context()` assembles a memory snapshot:

```
Memory Context:
━━━ Lessons ━━━
• User prefers code examples over explanations
• Always verify tech claims with documentation

━━━ Recent Context ━━━
• [10min ago] User asked about Rust borrow checker
• [5min ago] Assistant explained ownership rules

━━━ Knowledge Graph ━━━
• User → knows → Rust (weight: 0.9)
• User → works_on → ErnOSAgent (weight: 0.85)
• Rust → paradigm → systems (weight: 0.7)
```

This string is injected into the system prompt, budgeted to 15% of the total context window.

---

## Data Flow

### Ingestion (after every assistant response)

```
User Message + Assistant Response
         │
         ├──→ Timeline: record both as entries
         ├──→ Scratchpad: note key topics (if applicable)
         └──→ KG: extract entities/relations (future: LLM-based extraction)
```

### Recall (before every inference call)

```
User Query
    │
    ├──→ Lessons: search for matching rules
    ├──→ Scratchpad: include all current-session notes
    ├──→ Timeline: fetch recent entries
    ├──→ KG: recall decay-weighted relations
    │
    ▼
Budget allocation (15% of context window):
  40% → Scratchpad
  30% → Lessons
  20% → Timeline
  10% → KG
    │
    ▼
Formatted string → injected into system prompt
```
