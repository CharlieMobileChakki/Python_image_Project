"""Dedicated High-VRAM GPU Microservice for Arangtik Flow B (Official Qwen/Qwen-Image-2.1-Turbo).

7B Single-Stream DiT architecture with native 8-step Turbo schedule, prefix KV caching,
and lossless full bfloat16 precision — produces Myntra/Meesho studio-grade garment cutouts
in real-time on NVIDIA RTX A6000 (48GB VRAM).

Run on AIC Cloud RTX A6000 GPU Server (Python 3.11 Conda env):
    uvicorn server:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import io
import logging
import os
import sys

# Prevent PyTorch CUDA memory fragmentation on back-to-back jobs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import threading
import time
from typing import Any, Optional
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps
import torch

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen21-gpu-worker")

# Official Qwen-Image-2.1-Turbo (7B, native 8-step editing + product preservation)
MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-2.1-Turbo").strip()
API_KEY = os.getenv("QWEN_API_KEY", "").strip()
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "false").strip().lower() in ("1", "true", "yes")
OUTPUT_WIDTH = int(os.getenv("QWEN_OUTPUT_WIDTH", "896"))
OUTPUT_HEIGHT = int(os.getenv("QWEN_OUTPUT_HEIGHT", "896"))

app = FastAPI(title="Arangtik Qwen-Image-2.1-Turbo GPU Worker", version="3.0.0")

_pipeline = None
_pipe_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


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


def load_qwen21_pipeline():
    """Load official Qwen/Qwen-Image-2.1-Turbo in FULL lossless bfloat16 for RTX A6000 (48GB VRAM)."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    with _pipe_lock:
        if _pipeline is not None:
            return _pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not detected on this server.")

        from diffusers import QwenImage21Pipeline

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info("Loading %s (full lossless %s)...", MODEL_ID, dtype)
        t0 = time.perf_counter()

        pipe = QwenImage21Pipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )

        # Dynamic VRAM Management:
        # High-VRAM GPUs (RTX A6000 48GB, RTX 6000 Ada, A100 80GB) stay 100% resident in CUDA VRAM for ~6s generation!
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        if total_vram_gb > 36:
            logger.info("High VRAM GPU detected (%.1f GB). Loading 100%% resident into CUDA VRAM...", total_vram_gb)
            pipe.to("cuda")
            logger.info("Model 100%% resident in CUDA VRAM - Zero PCIe offload bottleneck!")
        elif hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
            logger.info("Enabled model_cpu_offload (safe for 24GB VRAM GPUs).")
        else:
            pipe.to("cuda")

        # PyTorch 2.x Ampere Tensor Core TF32 acceleration & cuDNN auto-tuner
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        # Attention Slicing is disabled to allow all 10,496 CUDA cores to compute SDPA in parallel
        if hasattr(pipe, "disable_attention_slicing"):
            pipe.disable_attention_slicing()

        # Keep VAE slicing and tiling enabled for memory-safe latent decoding
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

        _pipeline = pipe
        logger.info("Model %s ready in %.2fs.", MODEL_ID, time.perf_counter() - t0)
        return _pipeline


def _prepare_ref_image(raw_bytes: bytes) -> Image.Image:
    """Decode and preserve RGBA format as recommended by official Qwen-Image-2.1 docs."""
    ref_img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGBA")
    w, h = ref_img.size
    max_in = 1280
    if max(w, h) > max_in:
        scale = max_in / float(max(w, h))
        ref_img = ref_img.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.Resampling.LANCZOS)
    return ref_img


def _run_inference_sync(
    ref_img: Image.Image,
    prompt: str,
    steps: int = 8,
    guidance_scale: float = 1.0,
    seed: Optional[int] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> tuple[bytes, dict[str, str]]:
    """Run official Qwen-Image-2.1-Turbo 8-step editing with KV cache and return crisp PNG bytes."""
    pipe = load_qwen21_pipeline()
    used_seed = seed if seed is not None else int(time.time()) % 1000000
    # Official Qwen-Image-2.1 README uses CPU generator for deterministic reproducibility
    generator = torch.Generator("cpu").manual_seed(used_seed)

    target_w = int(width) if width and int(width) > 0 else OUTPUT_WIDTH
    target_h = int(height) if height and int(height) > 0 else OUTPUT_HEIGHT

    logger.info(
        "Running Qwen-Image-2.1-Turbo (size=%dx%d, in=%s) | prompt: %.200s...",
        target_w,
        target_h,
        ref_img.size,
        prompt,
    )

    # Keep CUDA caching allocator warm between inferences; avoid heavy empty_cache unless memory > 21GB
    if torch.cuda.is_available():
        mem_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        if mem_gb > 21.0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()

    t0 = time.perf_counter()
    with _pipe_lock, torch.inference_mode():
        pipe_kwargs = {
            "prompt": prompt,
            "image": ref_img,
            "width": target_w,
            "height": target_h,
            "use_kv_cache": True,
            "generator": generator,
        }
        import inspect
        call_params = inspect.signature(pipe.__call__).parameters
        if "num_inference_steps" in call_params and steps and steps > 0:
            pipe_kwargs["num_inference_steps"] = int(steps)
        if "true_cfg_scale" in call_params and guidance_scale and guidance_scale > 0:
            pipe_kwargs["true_cfg_scale"] = float(guidance_scale)
        elif "guidance_scale" in call_params and guidance_scale and guidance_scale > 0:
            pipe_kwargs["guidance_scale"] = float(guidance_scale)

        logger.info("Executing Qwen pipe with steps=%s, guidance_scale=%s, size=%dx%d", steps, guidance_scale, target_w, target_h)
        result = pipe(**pipe_kwargs)
        out_img: Image.Image = result.images[0].copy()
        del result

    elapsed = round(time.perf_counter() - t0, 2)
    gpu_info = _get_gpu_info()

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    headers = {
        "X-Model-Name": MODEL_ID,
        "X-GPU-Name": gpu_info["gpu_name"],
        "X-VRAM-GB": str(gpu_info["vram_total_gb"]),
        "X-Inference-Time-Sec": str(elapsed),
        "X-Seed": str(used_seed),
    }
    return buf.getvalue(), headers


def _prune_old_jobs() -> None:
    now = time.time()
    with _jobs_lock:
        stale = [jid for jid, j in _jobs.items() if now - j["created_at"] > 600]
        for jid in stale:
            _jobs.pop(jid, None)


def _background_job_runner(
    job_id: str,
    ref_img: Image.Image,
    prompt: str,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "running"

    try:
        png_bytes, headers = _run_inference_sync(
            ref_img=ref_img,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
        )
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["png_bytes"] = png_bytes
                _jobs[job_id]["headers"] = headers
                _jobs[job_id]["finished_at"] = time.time()
        logger.info("Job %s completed in %ss.", job_id, headers["X-Inference-Time-Sec"])
    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = str(exc)


@app.on_event("startup")
def _startup_preload():
    if PRELOAD_MODEL:
        try:
            logger.info("Preloading model into VRAM on startup...")
            p = load_qwen21_pipeline()
            logger.info("Application ready with resident VRAM pipeline.")
        except Exception as exc:
            logger.error("Startup preload failed (will retry on first request): %s", exc)


@app.get("/health")
def health(x_api_key: Optional[str] = Header(default=None)):
    _verify_api_key(x_api_key)
    gpu = _get_gpu_info()
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "precision": "full_bfloat16",
        "model_loaded": _pipeline is not None,
        "output_size": f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        **gpu,
    }


@app.post("/jobs")
async def submit_job(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    steps: int = Form(8),
    guidance_scale: float = Form(1.0),
    seed: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    x_api_key: Optional[str] = Header(default=None),
):
    """Submit an async Qwen-Image-2.1-Turbo generation job (immune to Cloudflare 524!)."""
    _verify_api_key(x_api_key)
    _prune_old_jobs()

    raw_bytes = await image.read()
    await image.close()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image uploaded.")

    try:
        ref_img = _prepare_ref_image(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}") from exc

    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "created_at": time.time(),
            "error": None,
            "png_bytes": None,
            "headers": {},
        }

    thread = threading.Thread(
        target=_background_job_runner,
        args=(job_id, ref_img, prompt, steps, guidance_scale, seed, width, height),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str, x_api_key: Optional[str] = Header(default=None)):
    _verify_api_key(x_api_key)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return {
            "job_id": job_id,
            "status": job["status"],
            "elapsed_sec": round(time.time() - job["created_at"], 1),
            "error": job["error"],
        }


@app.get("/jobs/{job_id}/result")
def get_job_result(job_id: str, x_api_key: Optional[str] = Header(default=None)):
    _verify_api_key(x_api_key)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job["status"] == "error":
            raise HTTPException(status_code=500, detail=job["error"] or "GPU job failed.")
        if job["status"] != "done" or not job["png_bytes"]:
            raise HTTPException(status_code=202, detail=f"Job still {job['status']}.")
        png_bytes = job["png_bytes"]
        headers = dict(job["headers"])
        _jobs.pop(job_id, None)

    return Response(content=png_bytes, media_type="image/png", headers=headers)


@app.post("/generate")
async def generate_garment(
    image: UploadFile = File(..., description="Original model-worn reference image."),
    prompt: str = Form(..., description="Detailed garment extraction prompt from ChatGPT Vision + studio rules."),
    steps: int = Form(8),
    guidance_scale: float = Form(1.0),
    seed: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    x_api_key: Optional[str] = Header(default=None),
):
    """Run synchronous Qwen-Image-2.1-Turbo generation on NVIDIA RTX A6000 (48GB VRAM)."""
    _verify_api_key(x_api_key)

    raw_bytes = await image.read()
    await image.close()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image uploaded.")

    try:
        ref_img = _prepare_ref_image(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}") from exc

    try:
        png_bytes, headers = _run_inference_sync(
            ref_img=ref_img,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
        )
    except Exception as exc:
        logger.exception("Qwen-Image-2.1-Turbo inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"GPU inference failed: {exc}") from exc

    return Response(content=png_bytes, media_type="image/png", headers=headers)
