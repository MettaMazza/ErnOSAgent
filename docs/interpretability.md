# Interpretability & Steering

The interpretability subsystem provides real-time feature analysis via Sparse Autoencoders (SAE), activation steering, and neural divergence tracking. All code lives in `src/interpretability/` and `src/steering/`.

## Module Map

### Interpretability (`src/interpretability/`)

| File | Purpose |
|------|---------|
| `mod.rs` | Shared types: `LabeledFeature`, `NeuralSnapshot`, `FeatureActivation` |
| `sae.rs` | SAE model — load `.safetensors`, encode activations, extract top features |
| `features.rs` | 195-feature labeled dictionary across 5 categories |
| `trainer.rs` | SAE training loop (pure f32 vector math, no tensor library) |
| `trainer_persist.rs` | Weight saving/loading for SAE training |
| `trainer_tests.rs` | Unit tests for SAE trainer |
| `train_runner.rs` | SAE training binary runner (batch activation collection + training) |
| `corpus.rs` | Training corpus management and activation extraction |
| `collector.rs` | Activation sample collection and JSONL flush |
| `snapshot.rs` | Capture and store neural snapshots |
| `divergence.rs` | KL divergence + cosine distance tracking |
| `probe.rs` | Linear probe for activation prediction |
| `live.rs` | Sliding-window live monitor for feature averages |
| `steering_bridge.rs` | Rule-based bridge between feature activations and steering vectors |
| `extractor.rs` | Feature extraction utilities |

### Steering (`src/steering/`)

| File | Purpose |
|------|---------|
| `mod.rs` | `SteeringVector`, `SteeringConfig` structs |
| `vectors.rs` | `VectorStore` — load, activate, deactivate steering vectors |
| `server.rs` | Steering server integration |

## Sparse Autoencoder (SAE)

### SaeModel (`src/interpretability/sae.rs`)

```rust
pub struct SaeModel {
    pub config: SaeConfig,
    pub encoder_weights: Vec<Vec<f32>>,  // [hidden_dim × input_dim]
    pub encoder_bias: Vec<f32>,          // [hidden_dim]
    pub decoder_weights: Vec<Vec<f32>>,  // [input_dim × hidden_dim]
    pub decoder_bias: Vec<f32>,          // [input_dim]
    pub is_loaded: bool,
}
```

### SaeConfig

```rust
pub struct SaeConfig {
    pub input_dim: usize,            // default: 3584
    pub hidden_dim: usize,           // default: 195
    pub sparsity_coefficient: f32,   // default: 0.04
}
```

### Key Methods

| Method | Description |
|--------|-------------|
| `empty()` | Create an unloaded SAE with zero weights |
| `load(path)` | Load from `.safetensors` file. Falls back to `empty()` if file missing |
| `encode(activations)` | Forward pass: input → encoder → ReLU → sparse hidden |
| `top_features(activations, k)` | Encode, then return top-k features by activation magnitude |
| `decoder_direction(feature_idx)` | Extract decoder column for a specific feature (for steering) |

### Safetensors Loading

`load()` parses the safetensors binary format:
1. Read 8-byte header length
2. Parse JSON header for tensor metadata (name, shape, dtype, data_offsets)
3. Extract F32 tensors: `encoder.weight`, `encoder.bias`, `decoder.weight`, `decoder.bias`
4. Construct weight matrices from flat float arrays

If the file is missing, truncated, or corrupt, the SAE degrades to `empty()` (feature disabled, not degraded).

## Feature Dictionary (`src/interpretability/features.rs`)

195 labeled features across 5 categories:

| Category | Index Range | Examples |
|----------|-------------|----------|
| Cognitive | 0–24 | Analytical Reasoning, Pattern Recognition, Code Generation, Task Decomposition |
| Emotional | 40–45 | Empathy Activation, Curiosity Drive, Frustration Detection |
| Behavioral | 80–86 | Tool Selection, Verification Impulse, Teaching Mode |
| Factual | 120–125 | Science Knowledge, Programming Knowledge, Machine Learning |
| Meta | 160–165 | Confidence Level, Uncertainty Signal, Hallucination Risk |

### FeatureActivation

```rust
pub struct FeatureActivation {
    pub feature_index: usize,
    pub activation: f32,
    pub label: String,
    pub category: String,
}
```

- `is_elevated()` — activation > 0.5
- `is_suppressed()` — activation < -0.3

## SAE Training (`src/interpretability/trainer.rs`)

Trains SAE on collected activation data using pure f32 vector math.

### SaeTrainingConfig

```rust
pub struct SaeTrainingConfig {
    pub learning_rate: f64,            // default: 1e-4
    pub epochs: usize,                 // default: 10
    pub batch_size: usize,             // default: 32
    pub sparsity_coefficient: f32,     // default: 0.04
    pub input_dim: usize,              // default: 3584
    pub hidden_dim: usize,             // default: 195
    pub data_dir: String,              // default: "data/activations"
    pub output_path: String,           // default: "data/sae_weights.safetensors"
}
```

### Training Flow

```
1. load_activations(data_dir) → Vec<Vec<f32>> from JSONL files
2. init_weights() → Xavier-init encoder + decoder matrices
3. For each epoch:
   a. For each batch:
      - forward_pass() → hidden (encoder + ReLU) → reconstruction (decoder)
      - compute_loss() → MSE + L1 sparsity penalty
      - backward_pass() → update decoder, then encoder through ReLU gate
4. save_weights(output_path) → JSON serialized weights
```

**Loss function**: `MSE(input, reconstruction) + λ × L1(hidden)` where `λ = sparsity_coefficient`

## Activation Collection (`src/interpretability/collector.rs`)

### ActivationCollector

```rust
pub struct ActivationCollector {
    samples: Vec<ActivationSample>,
    dir: PathBuf,
    max_samples: usize,
}
```

- `add(activations, context)` — buffer a sample
- `flush()` — write buffered samples to JSONL files in `data_dir`
- `count()` — current buffer size

## Neural Snapshots (`src/interpretability/snapshot.rs`)

### NeuralSnapshot

```rust
pub struct NeuralSnapshot {
    pub id: String,
    pub top_features: Vec<(usize, f32)>,
    pub context: String,
    pub divergence: f32,
    pub timestamp: DateTime<Utc>,
}
```

- `capture(top_features, context, divergence)` → saves snapshot to disk
- `recent(n)` → last n snapshots

## Divergence Tracking (`src/interpretability/divergence.rs`)

| Function | Description |
|----------|-------------|
| `kl_divergence(baseline, current)` | KL divergence between two activation distributions |
| `cosine_distance(a, b)` | 1 - cosine_similarity between two vectors |

### DivergenceTracker

Maintains a rolling history of divergence values:
- `new(baseline)` — set the baseline activation vector
- `record(current)` → compute and store KL divergence
- `trend()` → slope of recent divergence values
- `mean_divergence()` → average across all recorded values

## Live Monitor (`src/interpretability/live.rs`)

Sliding-window monitor for real-time feature analysis:

- `new(window_size)` — create monitor with fixed window
- `push(activations)` — add a frame of feature activations
- `averages()` → compute mean activation per feature across the window

## Steering Vectors (`src/steering/`)

### SteeringVector

```rust
pub struct SteeringVector {
    pub name: String,
    pub direction: Vec<f32>,
    pub default_strength: f32,
    pub active: bool,
    pub current_strength: f32,
}
```

### VectorStore

- `new(dir)` — load vectors from disk
- `list()` → all vectors
- `activate(name, strength)` — enable a steering vector at given strength
- `deactivate(name)` — disable a steering vector
- `active_vectors()` → currently active vectors

### Steering Bridge (`src/interpretability/steering_bridge.rs`)

Connects SAE feature activations to steering actions via `SteeringRule`:

```rust
pub struct SteeringRule {
    pub feature_index: usize,
    pub threshold: f32,
    pub vector_name: String,
    pub strength: f32,
}
```

`evaluate_rules(rules, activations)` — checks each rule against current activations, returns which steering vectors to activate.
