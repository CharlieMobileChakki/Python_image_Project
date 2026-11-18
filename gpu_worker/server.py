"""Dedicated RTX 3090 GPU Microservice for Arangtik Flow B (Qwen-Image-Edit + Lightning 8-Step LoRA).

Run on AIC Cloud RTX 3090 GPU Server (24GB VRAM):
    uvicorn server:app --host 0.0.0.0 --port 8001

Endpoints:
    GET  /health               -> GPU VRAM & Qwen model status
    POST /jobs                 -> Async submit (immune to Cloudflare 100s 524 timeout!)
    GET  /jobs/{job_id}        -> Poll job status
    GET  /jobs/{job_id}/result -> Fetch generated PNG bytes
    POST /generate             -> Direct synchronous generation
"""
from __future__ import annotations

import io
import logging
import math
import os
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
logger = logging.getLogger("qwen-gpu-worker")

# Configuration via environment variables
MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen-Image-Edit").strip()
USE_LIGHTNING_LORA = os.getenv("QWEN_USE_LIGHTNING", "true").strip().lower() in ("1", "true", "yes")
LIGHTNING_LORA_REPO = os.getenv("QWEN_LORA_REPO", "lightx2v/Qwen-Image-Lightning").strip()
LIGHTNING_LORA_WEIGHT = os.getenv("QWEN_LORA_WEIGHT", "Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors").strip()
API_KEY = os.getenv("QWEN_API_KEY", "").strip()
QUANTIZE_4BIT = os.getenv("QWEN_QUANTIZE_4BIT", "true").strip().lower() in ("1", "true", "yes")
PRELOAD_MODEL = os.getenv("PRELOAD_MODEL", "false").strip().lower() in ("1", "true", "yes")
MAX_EDGE = int(os.getenv("QWEN_MAX_EDGE", "1024"))
DEFAULT_STEPS = int(os.getenv("QWEN_DEFAULT_STEPS", "8" if USE_LIGHTNING_LORA else "40"))
DEFAULT_CFG = float(os.getenv("QWEN_DEFAULT_CFG", "1.0" if USE_LIGHTNING_LORA else "4.0"))
# Official Qwen-Image-Edit requires a single space negative_prompt for razor-sharp texture fidelity
DEFAULT_NEGATIVE_PROMPT = " "

app = FastAPI(title="Arangtik Qwen GPU Worker (Lightning 8-Step)", version="2.2.0")

_pipeline = None
_lora_active = False
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


def load_qwen_pipeline():
    """Load Qwen-Image-Edit + Lightning 8-Step LoRA optimized for RTX 3090 (24GB VRAM)."""
    global _pipeline, _lora_active
    if _pipeline is not None:
        return _pipeline

    with _pipe_lock:
        if _pipeline is not None:
            return _pipeline

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU not detected on this server.")

        from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageEditPipeline

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info("Loading %s (dtype=%s, 4bit=%s, lightning=%s)...", MODEL_ID, dtype, QUANTIZE_4BIT, USE_LIGHTNING_LORA)
        t0 = time.perf_counter()

        load_kwargs: dict[str, Any] = {"torch_dtype": dtype}

        # Distilled FlowMatch scheduler required for Lightning 8-step LoRA sharpness
        if USE_LIGHTNING_LORA:
            scheduler_config = {
                "base_image_seq_len": 256,
                "base_shift": math.log(3),
                "invert_sigmas": False,
                "max_image_seq_len": 8192,
                "max_shift": math.log(3),
                "num_train_timesteps": 1000,
                "shift": 1.0,
                "shift_terminal": None,
                "stochastic_sampling": False,
                "time_shift_type": "exponential",
                "use_beta_sigmas": False,
                "use_dynamic_shifting": True,
                "use_exponential_sigmas": False,
                "use_karras_sigmas": False,
            }
            load_kwargs["scheduler"] = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        # 4-bit NF4 quantization keeps 20B Qwen-Image-Edit safely within RTX 3090 24GB VRAM
        if QUANTIZE_4BIT:
            try:
                from diffusers.quantizers import PipelineQuantizationConfig

                load_kwargs["quantization_config"] = PipelineQuantizationConfig(
                    quant_backend="bitsandbytes_4bit",
                    quant_kwargs={
                        "load_in_4bit": True,
                        "bnb_4bit_quant_type": "nf4",
                        "bnb_4bit_compute_dtype": dtype,
                    },
                    components_to_quantize=["transformer", "text_encoder"],
                )
                logger.info("Enabled PipelineQuantizationConfig (4-bit NF4) for 24GB VRAM optimization.")
            except Exception as exc:
                logger.warning("PipelineQuantizationConfig unavailable (%s); falling back to CPU offload.", exc)

        pipe = QwenImageEditPipeline.from_pretrained(MODEL_ID, **load_kwargs)

        # Attach official Lightning 8-Step LoRA (unfused — compatible with 4-bit NF4 weights!)
        if USE_LIGHTNING_LORA:
            try:
                logger.info("Loading Lightning LoRA: %s / %s", LIGHTNING_LORA_REPO, LIGHTNING_LORA_WEIGHT)
                pipe.load_lora_weights(LIGHTNING_LORA_REPO, weight_name=LIGHTNING_LORA_WEIGHT)
                _lora_active = True
                logger.info("Lightning 8-Step LoRA attached successfully!")
            except Exception as exc:
                _lora_active = False
                logger.warning("Failed to load Lightning LoRA (%s); continuing with base model.", exc)

        # CPU offload guarantees zero OOM spikes on 24GB VRAM + 31GB System RAM
        if hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

        _pipeline = pipe
        logger.info("Model %s ready in %.2fs (lightning_lora=%s).", MODEL_ID, time.perf_counter() - t0, _lora_active)
        return _pipeline


def _prepare_ref_image(raw_bytes: bytes) -> Image.Image:
    ref_img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw_bytes))).convert("RGB")
    w, h = ref_img.size
    if max(w, h) > MAX_EDGE:
        scale = MAX_EDGE / float(max(w, h))
        ref_img = ref_img.resize((max(64, int(w * scale)), max(64, int(h * scale))), Image.Resampling.LANCZOS)
    return ref_img


def _run_inference_sync(
    ref_img: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> tuple[bytes, dict[str, str]]:
    """Run Qwen-Image-Edit under GPU lock and return crisp PNG bytes + headers."""
    pipe = load_qwen_pipeline()
    used_seed = seed if seed is not None else int(time.time()) % 1000000
    generator = torch.Generator(device="cuda").manual_seed(used_seed)

    # Enforce optimal steps & CFG for Lightning 8-step LoRA vs base model
    actual_steps = max(4, min(50, int(steps)))
    actual_cfg = float(guidance_scale)
    if _lora_active and actual_steps <= 12:
        actual_cfg = 1.0
    elif not _lora_active and actual_steps < 28:
        actual_steps = 36
        actual_cfg = 4.0

    # Official Qwen-Image-Edit works sharpest with blank single-space negative_prompt
    neg = negative_prompt if negative_prompt and negative_prompt.strip() else " "

    t0 = time.perf_counter()
    with _pipe_lock, torch.inference_mode():
        result = pipe(
            image=ref_img,
            prompt=prompt,
            negative_prompt=neg,
            num_inference_steps=actual_steps,
            true_cfg_scale=actual_cfg,
            generator=generator,
        )
        out_img: Image.Image = result.images[0]
        torch.cuda.empty_cache()

    elapsed = round(time.perf_counter() - t0, 2)
    gpu_info = _get_gpu_info()

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    headers = {
        "X-Model-Name": f"{MODEL_ID}+Lightning-8step" if _lora_active else MODEL_ID,
        "X-GPU-Name": gpu_info["gpu_name"],
        "X-Inference-Time-Sec": str(elapsed),
        "X-Seed": str(used_seed),
    }
    return buf.getvalue(), headers


def _prune_old_jobs() -> None:
    """Keep memory tidy by dropping finished jobs older than 10 minutes."""
    now = time.time()
    with _jobs_lock:
        stale = [jid for jid, j in _jobs.items() if now - j["created_at"] > 600]
        for jid in stale:
            _jobs.pop(jid, None)


def _background_job_runner(
    job_id: str,
    ref_img: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "running"

    try:
        png_bytes, headers = _run_inference_sync(
            ref_img=ref_img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
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
        "lightning_lora": _lora_active,
        "model_loaded": _pipeline is not None,
        "quantize_4bit": QUANTIZE_4BIT,
        "default_steps": DEFAULT_STEPS,
        "max_edge": MAX_EDGE,
        **gpu,
    }


@app.post("/jobs")
async def submit_job(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
    steps: int = Form(DEFAULT_STEPS),
    guidance_scale: float = Form(DEFAULT_CFG),
    seed: Optional[int] = Form(None),
    x_api_key: Optional[str] = Header(default=None),
):
    """Submit an async GPU generation job (returns immediately — never hits Cloudflare 524!)."""
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
        args=(job_id, ref_img, prompt, negative_prompt, steps, guidance_scale, seed),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job_status(job_id: str, x_api_key: Optional[str] = Header(default=None)):
    """Check status of an async GPU generation job."""
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
    """Download the generated PNG once job status is 'done'."""
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
    negative_prompt: str = Form(DEFAULT_NEGATIVE_PROMPT),
    steps: int = Form(DEFAULT_STEPS),
    guidance_scale: float = Form(DEFAULT_CFG),
    seed: Optional[int] = Form(None),
    x_api_key: Optional[str] = Header(default=None),
):
    """Run synchronous Qwen-Image-Edit generation on RTX 3090."""
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
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
    except Exception as exc:
        logger.exception("Qwen inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"GPU inference failed: {exc}") from exc

    return Response(content=png_bytes, media_type="image/png", headers=headers)
