#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "diffusers>=0.32.0",
#     "torch>=2.4.0",
#     "transformers>=4.44.0",
#     "sentencepiece>=0.2.0",
#     "accelerate>=1.0.0",
#     "protobuf>=5.0.0",
#     "flask>=3.0.0",
# ]
# ///
"""
Persistent Flux image generation server.

Loads the Flux pipeline ONCE into MPS memory and serves generation
requests over HTTP. Configured via environment variables:

    FLUX_MODEL_PATH  — HF repo or local path (default: black-forest-labs/FLUX.1-dev)
    FLUX_HOST        — bind address (default: 127.0.0.1)
    FLUX_PORT        — port (default: 7860)

Usage:
    uv run scripts/flux_server.py

API:
    POST /generate
    {
        "prompt": "a sunset over the ocean",
        "width": 1024,
        "height": 1024,
        "steps": 50,
        "guidance": 3.5,
        "filename": "/path/to/output.png"
    }

    Response: { "path": "/abs/path/output.png", "width": 1024, "height": 1024 }

    GET /health
    Response: { "status": "ready", "model": "...", "device": "mps" }
"""

import base64
import io
import os
import sys
import time
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

# Global pipeline — loaded once at startup
_pipe = None
_device = None
_model_path = None


def get_device():
    """Detect best available device."""
    import torch
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dtype(device: str):
    import torch
    if device == "mps":
        return torch.float16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def _patch_scheduler(scheduler):
    """Monkeypatch for MPS IndexError bug."""
    if getattr(scheduler, "_is_patched", False):
        return
    original_step = scheduler.step
    def safe_step(model_output, timestep, sample, **kwargs):
        try:
            return original_step(model_output, timestep, sample, **kwargs)
        except IndexError:
            print("Warning: Scheduler IndexError caught (MPS bug).", file=sys.stderr)
            if not kwargs.get("return_dict", True):
                return (sample,)
            from diffusers.schedulers.scheduling_utils import SchedulerOutput
            return SchedulerOutput(prev_sample=sample)
    scheduler.step = safe_step
    scheduler._is_patched = True


def load_pipeline():
    """Load pipeline into memory. Called once at startup."""
    global _pipe, _device, _model_path

    _model_path = os.environ.get("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev")
    _device = get_device()
    dtype = get_dtype(_device)

    print(f"Loading Flux pipeline: {_model_path} on {_device} ({dtype})")
    start = time.time()

    from diffusers import FluxPipeline
    _pipe = FluxPipeline.from_pretrained(_model_path, torch_dtype=dtype)
    _pipe.to(_device)
    _patch_scheduler(_pipe.scheduler)

    elapsed = time.time() - start
    print(f"Pipeline loaded in {elapsed:.1f}s")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ready" if _pipe is not None else "loading",
        "model": _model_path,
        "device": _device,
    })


@app.route("/generate", methods=["POST"])
def generate():
    if _pipe is None:
        return jsonify({"error": "Pipeline not loaded"}), 503

    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' field"}), 400

    prompt = data["prompt"]
    width = data.get("width", 1024)
    height = data.get("height", 1024)
    steps = data.get("steps", 50)
    guidance = data.get("guidance", 3.5)
    filename = data.get("filename")
    seed = data.get("seed")

    # Clamp dimensions to reasonable bounds
    width = max(256, min(2048, width))
    height = max(256, min(2048, height))
    steps = max(1, min(100, steps))

    import torch

    generator = None
    if seed is not None:
        generator = torch.Generator("cpu").manual_seed(seed)

    print(f"Generating: {width}x{height}, {steps} steps, guidance {guidance}")
    print(f"Prompt: {prompt[:100]}...")
    start = time.time()

    image = _pipe(
        prompt,
        guidance_scale=guidance,
        num_inference_steps=steps,
        width=width,
        height=height,
        max_sequence_length=512,
        generator=generator,
    ).images[0]

    elapsed = time.time() - start
    print(f"Generated in {elapsed:.1f}s")

    # Save to disk if filename provided
    if filename:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(str(output_path))
        abs_path = str(output_path.resolve())
    else:
        abs_path = None

    # Always return base64 for multimodal feedback
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return jsonify({
        "path": abs_path,
        "width": width,
        "height": height,
        "elapsed_s": round(elapsed, 1),
        "base64": b64,
    })


if __name__ == "__main__":
    load_pipeline()

    host = os.environ.get("FLUX_HOST", "127.0.0.1")
    port = int(os.environ.get("FLUX_PORT", "7860"))
    print(f"Flux server listening on {host}:{port}")
    app.run(host=host, port=port, threaded=False)
