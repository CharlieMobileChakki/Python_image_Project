"""Dedicated RTX 3090 GPU Microservice for Arangtik Flow B (Qwen-Image-Edit).

Run on AIC Cloud RTX 3090 GPU Server (24GB VRAM):
    uvicorn server:app --host 0.0.0.0 --port 8001

Endpoints:
    GET  /health    -> GPU VRAM & Qwen model status
    POST /generate  -> Reference Photo + ChatGPT Prompt -> Clean Studio Garment PNG
"""
from __future__ import annotations

import io
import logging
import os
import threading
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps
import torch

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen-gpu-worker")

# Configuration via environment variables
MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-Edit").strip()
API_KEY = os.getenv("QWEN_API_KEY", "").strip()
QUANTIZE_4BIT = os.getenv("QWEN_QUANTIZE_4BIT", "true").strip().lower() in ("1", "true", "yes")
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "false").strip().lower() in ("1", "true", "yes")
DEFAULT_NEGATIVE_PROMPT = (
    "person, human, model, face, head, hands, fingers, arms, legs, skin, hair, mannequin, "
    "hanger, cluttered background, shadows, watermark, text, blurry, distorted pattern"
)

app = FastAPI(title="Arangtik Qwen GPU Worker", version="2.0.0")

_pipeline = None
_pipe_lock = threading.Lock()


def _verify_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and (x_api_key or "").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


def _get_gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "gpu_name": "CPU (No CUDA)", "vram_total_gb": 0.0, "vram_used_gb": 0.0}
    props = torch.cuda.get_device_properties(0)
    total_gb = round(props.total_memory / (1024 ** 3), 2)
    used_gb = round(torch.cuda.memory_allocated(0) / (1024 ** 3), 2)
    return {
        "cuda_available": True,
        "gpu_name": props.name,
        "vram_total_gb": total_gb,
        "vram_used_gb": used_gb,
    }


def load_qwen_pipeline():
    """Load Qwen-Image-Edit optimized for 24GB VRAM (RTX 3090)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipe_lock:
        if _pipeline is not None:
            return _pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not detected on this server.")

        from diffusers import QwenImageEditPipeline

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info("Loading %s (dtype=%s, 4bit=%s)...", MODEL_ID, dtype, QUANTIZE_4BIT)
        t0 = time.perf_counter()

        load_kwargs = {"torch_dtype": dtype}

        # 4-bit NF4 quantization keeps 20B Qwen-Image-Edit safely within RTX 3090 24GB VRAM
        if QUANTIZE_4BIT:
            try:
                from diffusers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                )
                logger.info("Enabled 4-bit NF4 quantization for 24GB VRAM optimization.")
            except Exception as exc:
                logger.warning("4-bit BitsAndBytesConfig unavailable (%s); falling back to CPU offload.", exc)

        pipe = QwenImageEditPipeline.from_pretrained(MODEL_ID, **load_kwargs)

        # CPU offload guarantees zero OOM spikes on 24GB VRAM + 31GB System RAM
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        if hasattr(pipe, "enable_vae_slicing"):
            pipe.enable_vae_slicing()
        if hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        _pipeline = pipe
        logger.info("Model %s loaded in %.2fs.", MODEL_ID, time.perf_counter() - t0)
        return _pipeline


@app.on_event("startup")
def _startup_preload():
    if PRELOAD_MODEL:
        try:
            load_qwen_pipeline()
        except Exception as exc:
            logger.error("Startup preload failed (will retry on first request): %s", exc)


@app.get("/health")
def health(x_api_key: Optional[str] = Header(default=None)):
    _verify_api_key(x_api_key)
    gpu = _get_gpu_info()
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "model_loaded": _pipeline is not None,
        "quantize_4bit": QUANTIZE_4BIT,
        **gpu,
    }


@app.post("/generate")
async def generate_garment(
    image: UploadFile = File(..., description="Original model-worn reference image."),
    prompt: str = Form(..., description="Detailed garment extraction prompt from ChatGPT Vision + studio rules."),
    negative_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
    steps: int = Form(28),
    guidance_scale: float = Form(4.0),
    seed: Optional[int] = Form(None),
    x_api_key: Optional[str] = Header(default=None),
):
    """Run Qwen-Image-Edit on RTX 3090 and return clean studio garment PNG."""
    _verify_api_key(x_api_key)

    raw_bytes = await image.read()
    await image.close()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image uploaded.")

    try:
        ref_img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}") from exc

    # Cap max input edge to 1024px for crisp, fast 24GB VRAM inference
    w, h = ref_img.size
    if max(w, h) > 1024:
        scale = 1024 / float(max(w, h))
        ref_img = ref_img.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.Resampling.LANCZOS)

    try:
        pipe = load_qwen_pipeline()
    except Exception as exc:
        logger.exception("Model load failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Failed to load Qwen model: {exc}") from exc

    used_seed = seed if seed is not None else int(time.time()) % 1000000
    generator = torch.Generator(device="cuda").manual_seed(used_seed)

    t0 = time.perf_counter()
    try:
        with _pipe_lock, torch.inference_mode():
            result = pipe(
                image=ref_img,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=max(8, min(50, int(steps))),
                true_cfg_scale=float(guidance_scale),
                generator=generator,
            )
            out_img: Image.Image = result.images[0]
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.exception("Qwen inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"GPU inference failed: {exc}") from exc

    elapsed = round(time.perf_counter() - t0, 2)
    gpu_info = _get_gpu_info()

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={
            "X-Model-Name": MODEL_ID,
            "X-GPU-Name": gpu_info["gpu_name"],
            "X-Inference-Time-Sec": str(elapsed),
            "X-Seed": str(used_seed),
        },
    )

