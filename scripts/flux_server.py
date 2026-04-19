#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "fastapi>=0.115.0",
#     "uvicorn>=0.30.0",
#     "diffusers>=0.32.0",
#     "torch>=2.4.0",
#     "transformers>=4.44.0",
#     "sentencepiece>=0.2.0",
#     "accelerate>=1.0.0",
#     "protobuf>=5.0.0",
#     "pillow>=10.0.0",
# ]
# ///
"""
Flux image generation server for Ern-OS.

Serves a local FLUX.1-dev model via FastAPI on port 8890.
Reuses weights from ~/.cache/huggingface/hub/ (no re-download).

Usage:
    uv run flux_server.py
"""

import base64
import io
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


# ─── Device Detection ───
def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_dtype(device: str):
    if device == "mps":
        return torch.float16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


# ─── Scheduler MPS Bugfix ───
def patch_scheduler(scheduler):
    if getattr(scheduler, "_is_patched", False):
        return
    original_step = scheduler.step

    def safe_step(model_output, timestep, sample, **kwargs):
        try:
            return original_step(model_output, timestep, sample, **kwargs)
        except IndexError:
            print("Warning: Scheduler IndexError caught (MPS bug)", file=sys.stderr)
            if not kwargs.get("return_dict", True):
                return (sample,)
            from diffusers.schedulers.scheduling_utils import SchedulerOutput
            return SchedulerOutput(prev_sample=sample)

    scheduler.step = safe_step
    scheduler._is_patched = True


# ─── Global Pipeline ───
pipe = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
    device = get_device()
    dtype = get_dtype(device)
    model_path = os.environ.get("FLUX_MODEL_PATH", "black-forest-labs/FLUX.1-dev")
    print(f"Loading Flux pipeline: {model_path} on {device} ({dtype})")

    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe.to(device)
    patch_scheduler(pipe.scheduler)
    print("Flux pipeline ready.")
    yield
    pipe = None


app = FastAPI(title="Ern-OS Flux Server", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance: float = 3.5
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    image_base64: str
    width: int
    height: int


@app.get("/health")
def health():
    return {"status": "ready" if pipe is not None else "loading"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if pipe is None:
        return {"error": "Pipeline not loaded"}

    generator = None
    if req.seed is not None:
        generator = torch.Generator("cpu").manual_seed(req.seed)

    image = pipe(
        req.prompt,
        guidance_scale=req.guidance,
        num_inference_steps=req.steps,
        width=req.width,
        height=req.height,
        max_sequence_length=512,
        generator=generator,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return GenerateResponse(image_base64=b64, width=req.width, height=req.height)


if __name__ == "__main__":
    port = int(os.environ.get("FLUX_PORT", "8890"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
