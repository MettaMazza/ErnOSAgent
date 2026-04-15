---
license: mit
library_name: safetensors
tags:
  - sparse-autoencoder
  - interpretability
  - gemma-4
  - mechanistic-interpretability
  - sae
  - neural-interpretability
datasets:
  - self-generated
language:
  - en
base_model: google/gemma-4-27b-it
---

# Gemma 4 Sparse Autoencoder (SAE)

**First open-source Sparse Autoencoder trained on Gemma 4 26B activations.**

Built for the [ErnOSAgent](https://github.com/MettaMazza/ErnOSAgent) neural interpretability pipeline.

## Architecture

| Parameter | Value |
|---|---|
| Features | 131,072 |
| Model Dimension | 2,816 |
| Expansion Factor | 46.6× |
| Format | SafeTensors |
| Source Model | Gemma 4 26B IT (Q4_K_M) |
| Training Hardware | Apple M3 Ultra (512GB RAM) |
| Extraction Layer | Last-layer residual stream |

## Files

- `gemma4_sae_1m.safetensors` — SAE encoder/decoder weights (2.8GB)
- `feature_map.json` — 195 labeled features via automated probing

## Usage

### With ErnOSAgent (Rust)

```bash
# Place weights in the data directory
mkdir -p ~/.ernosagent/sae_training/
# Download weights
huggingface-cli download MettaMazza/gemma4-sae gemma4_sae_1m.safetensors --local-dir ~/.ernosagent/sae_training/
# Download feature map
huggingface-cli download MettaMazza/gemma4-sae feature_map.json --local-dir ~/.ernosagent/sae_training/

# Run ErnOS — SAE loads automatically
cd ~/Desktop/ErnOSAgent && cargo run --release
```

### With Python

```python
from safetensors import safe_open
import numpy as np

with safe_open("gemma4_sae_1m.safetensors", framework="numpy") as f:
    encoder = f.get_tensor("encoder.weight")  # [131072, 2816]
    decoder = f.get_tensor("decoder.weight")  # [2816, 131072]
    bias = f.get_tensor("encoder.bias")       # [131072]

# Encode activations → sparse features
activations = np.random.randn(2816).astype(np.float32)  # from model
features = np.maximum(0, encoder @ activations + bias)    # ReLU

# Top-k active features
top_k = np.argsort(features)[-20:][::-1]
for idx in top_k:
    if features[idx] > 0:
        print(f"Feature {idx}: {features[idx]:.3f}")
```

## Feature Map

The `feature_map.json` contains 195 human-interpretable labels mapped to SAE feature indices via automated probing. Categories include:

- **Reasoning**: Chain-of-thought, logical deduction, mathematical reasoning
- **Safety**: Refusal, deception detection, bias detection, power-seeking
- **Cognitive**: Creativity, recall, planning, context integration
- **Emotional**: Valence, arousal, emotional tone detection
- **Technical**: Code generation, technical depth, language detection

## Training

Trained using ErnOSAgent's native SAE training pipeline (`cargo run -- --train-sae`):

1. **Activation Collection**: Extract 2816-dim residual stream vectors from Gemma 4 26B via llama.cpp's native `/embedding` endpoint
2. **Training**: TopK sparse autoencoder with gradient descent (k=64, LR=3e-4)
3. **Probing**: Automated feature labeling via targeted prompt pairs

## License

MIT — same as ErnOSAgent.

## Citation

```bibtex
@misc{mettamazza2026gemma4sae,
  title={Gemma 4 Sparse Autoencoder for Neural Interpretability},
  author={MettaMazza},
  year={2026},
  url={https://huggingface.co/MettaMazza/gemma4-sae}
}
```
