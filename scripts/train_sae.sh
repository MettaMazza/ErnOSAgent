#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# ErnOSAgent SAE Training Script
#
# Train a Sparse Autoencoder on Gemma 4 31B residual stream activations.
# Designed for Apple Silicon M3 Ultra (512GB RAM).
#
# Usage:
#   ./scripts/train_sae.sh [--steps 50000] [--features 16384]
#
# Prerequisites:
#   - Gemma 4 31B GGUF model downloaded and configured
#   - llama-server available on PATH
#   - Rust toolchain with Metal support
# ═══════════════════════════════════════════════════════════════════

set -euo pipefail

STEPS="${1:-50000}"
FEATURES="${2:-16384}"
DATA_DIR="${ERNOSAGENT_DATA_DIR:-$HOME/.ernosagent}"
LOG_FILE="${DATA_DIR}/training/sae_training.log"
WEIGHTS_FILE="${DATA_DIR}/training/sae_weights.safetensors"

echo "═══════════════════════════════════════════════════════"
echo "  ErnOSAgent SAE Training"
echo "  Steps: ${STEPS} | Features: ${FEATURES}"
echo "  Output: ${WEIGHTS_FILE}"
echo "  Log:    ${LOG_FILE}"
echo "═══════════════════════════════════════════════════════"
echo ""

mkdir -p "$(dirname "$LOG_FILE")"

echo "[$(date)] Starting SAE training..." | tee "$LOG_FILE"

# Build with interpretability features enabled
echo "[$(date)] Building with --features interp..." | tee -a "$LOG_FILE"
cargo build --release --features interp 2>&1 | tee -a "$LOG_FILE"

# Run the SAE training CLI
echo "[$(date)] Launching SAE training (this will take 24-48 hours)..." | tee -a "$LOG_FILE"
echo ""

# The training runs using the interpretability trainer module.
# For the initial version, we use CPU training with the demo dataset.
# Once Candle Metal acceleration is fully integrated, this will use GPU.
./target/release/ernosagent train-sae \
  --steps "$STEPS" \
  --features "$FEATURES" \
  --output "$WEIGHTS_FILE" \
  2>&1 | tee -a "$LOG_FILE"

RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "[$(date)] ✅ SAE training completed!" | tee -a "$LOG_FILE"
    echo "  Weights saved to: ${WEIGHTS_FILE}" | tee -a "$LOG_FILE"
    echo "  Restart the server with --features interp to use real SAE weights." | tee -a "$LOG_FILE"
else
    echo ""
    echo "[$(date)] ❌ SAE training failed (exit code: ${RESULT})" | tee -a "$LOG_FILE"
    echo "  Check the log file: ${LOG_FILE}" | tee -a "$LOG_FILE"
fi
