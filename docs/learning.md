# Learning Pipeline

The learning subsystem provides on-device incremental training. All code lives in `src/learning/`. The pipeline is **fully integrated** — Observer verdicts automatically fill training buffers, and a background scheduler triggers sleep cycles when thresholds are met.

## Module Map

| File | Purpose |
|------|---------|
| `mod.rs` | `TrainingSample`, `TrainingMethod`, `PipelineStatus` structs |
| `teacher.rs` | Training orchestrator — SFT and preference training |
| `sleep.rs` | Background sleep consolidation cycle |
| `buffers.rs` | `GoldenBuffer` — high-quality sample storage |
| `buffers_rejection.rs` | `RejectionBuffer` + `PreferencePair` for DPO/KTO |
| `observer_buffer.rs` | Observer-scored sample buffer |
| `manifest.rs` | Training run metadata tracking |
| `distill.rs` | Knowledge distillation utilities |

### LoRA Subsystem (`src/learning/lora/`)

| File | Purpose |
|------|---------|
| `mod.rs` | `LoraConfig` struct |
| `weights.rs` | `LoraLayer` — A/B matrix storage with forward pass |
| `forward.rs` | Forward pass through LoRA adapter |
| `training.rs` | `train_step()` and `train_epoch()` — finite-difference gradient computation |
| `loss.rs` | Cross-entropy loss |
| `loss_dpo.rs` | Direct Preference Optimization loss |
| `loss_kto.rs` | Kahneman-Tversky Optimization loss |
| `loss_simpo.rs` | Simple Preference Optimization loss |
| `optimizer.rs` | SGD optimizer |
| `adapters.rs` | Multi-adapter management |
| `ewc.rs` | Elastic Weight Consolidation (catastrophic forgetting prevention) |
| `training_alignment.rs` | Alignment training utilities |

### GRPO Subsystem (`src/learning/grpo/`)

| File | Purpose |
|------|---------|
| `mod.rs` | Module declarations |
| `generation.rs` | Candidate response generation (online + offline) |
| `rewards.rs` | Multi-signal reward scoring (length, relevance, structure) + advantage computation |
| `training.rs` | GRPO loss with KL penalty |

## Core Types

### TrainingSample

```rust
pub struct TrainingSample {
    pub id: String,
    pub input: String,
    pub output: String,
    pub method: TrainingMethod,
    pub quality_score: f32,
    pub timestamp: DateTime<Utc>,
}
```

### TrainingMethod

```rust
pub enum TrainingMethod { Sft, Orpo, SimPO, Kto, Dpo, Grpo }
```

### LoraConfig

```rust
pub struct LoraConfig {
    pub rank: usize,          // default: 4
    pub alpha: f32,           // default: 8.0
    pub learning_rate: f64,   // default: 1e-4
    pub model_dim: usize,     // default: 3584
}
```

### LoraLayer

```rust
pub struct LoraLayer {
    pub name: String,
    pub a_matrix: Vec<Vec<f32>>,  // [rank × input_dim]
    pub b_matrix: Vec<Vec<f32>>,  // [output_dim × rank]
    pub rank: usize,
    pub alpha: f32,
    pub input_dim: usize,
    pub output_dim: usize,
}
```

## Training Flow

### SFT (Supervised Fine-Tuning)

```
Teacher.train_sft(samples) →
  for each sample:
    input_vec = text_to_vec(sample.input)
    target_vec = text_to_vec(sample.output)
    loss = training::train_step(&mut lora, &input, &target, lr)
  → TrainingResult { method: "SFT", samples, loss }
```

### Preference Training (DPO)

```
Teacher.train_preference(pairs, method) →
  for each PreferencePair:
    chosen_vec → LoRA forward
    rejected_vec → LoRA forward
    loss = chosen_loss - rejected_loss margin
  → TrainingResult { method: "DPO", samples, loss }
```

### GRPO (Group Relative Policy Optimization)

```
1. generate_candidates_offline(prompt, n) → n candidate responses
2. score_group(candidates, query) → reward scores (length + relevance + structure)
3. compute_advantages(scores) → zero-mean advantages
4. train_step(candidates, query, kl_coeff) → policy loss with KL penalty
```

**Reward signals** in `rewards::score_group()`:
- **Length score**: `min(len / 200.0, 1.5)` — rewards substantive responses
- **Relevance score**: Keyword overlap between response and query
- **Structure score**: Presence of paragraphs, lists, code blocks

### Gradient Computation

LoRA training uses **finite-difference gradient estimation** (not autograd). In `training::train_step()`:

1. Compute forward loss at current B-matrix weights
2. For each weight, perturb by `epsilon` (1e-4), compute loss again
3. Gradient = `(perturbed_loss - base_loss) / epsilon`
4. Update: `weight -= learning_rate × gradient`

This approach requires no autograd library and works on any hardware.

## Sleep Consolidation

`sleep::run_sleep_cycle()` is a background task that:

1. Drains the `GoldenBuffer` (up to 32 samples)
2. Runs SFT training via `Teacher.train_sft()`
3. Drains the `RejectionBuffer` (up to 16 pairs)
4. Runs DPO training via `Teacher.train_preference()`
5. Calls `synaptic.decay_all(0.95)` to weaken inactive knowledge graph nodes
6. Calls `lessons.decay_unused(0.98, 0.3)` to prune stale lessons

The sleep cycle is triggered by the `sleep_cycle` job in the cron engine (`src/scheduler/`). The scheduler ticks every 15 seconds, checking all enabled jobs against their schedule. The `sleep_cycle` job runs every 5 minutes and triggers when buffer thresholds are met:
- GoldenBuffer ≥ 10 samples, OR
- RejectionBuffer ≥ 5 pairs

## GoldenBuffer

```rust
pub struct GoldenBuffer {
    samples: Vec<TrainingSample>,
    max_size: usize,
}
```

- `add(sample)` — append (drops oldest if at capacity)
- `drain_batch(count)` — remove and return up to `count` samples
- `count()` — current buffer size

## PreferencePair

```rust
pub struct PreferencePair {
    pub id: String,
    pub input: String,
    pub chosen: String,
    pub rejected: String,
    pub rejection_reason: String,
    pub timestamp: DateTime<Utc>,
}
```

Created when the Observer rejects a response — the original (rejected) and retry (chosen) form a preference pair for DPO training.

## Live Integration

### Observer → Training Buffers

`src/web/training_capture.rs` captures Observer verdicts as training signals:
- **Approved** responses → `capture_approved()` → `GoldenBuffer` (SFT)
- **Rejected** responses + retried replacement → `capture_rejection()` → `RejectionBuffer` (DPO pairs)

Both operate as fire-and-forget `tokio::spawn` background tasks.

### Learning Tool

The `learning` tool (`src/tools/learning_tool.rs`) provides live access to the training pipeline through `AppState`:
- `status` — real buffer counts and pipeline readiness
- `buffer_stats` — detailed capacity info
- `trigger_training` — queue a training run
- `sleep` — manually trigger the sleep consolidation cycle

### LoRA Adapter Loading

`LlamaCppConfig.lora_adapter` specifies a trained adapter to load at inference via `--lora` flag. After the scheduler completes a sleep cycle and saves an adapter, a server restart loads it into the model.
