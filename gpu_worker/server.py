"""FastAPI HTTP Microservice for Arangtik Flow B (Qwen/Qwen-Image-2.1-Turbo + BRIA RMBG-1.4).

Used for HTTP-based GPU servers (RunPod Pods, Vast.ai, Dedicated Multi-GPU Nodes, Modal ASGI).
RunPod Serverless Queue mode (`rp_handler.py`) shares the exact same `engine.py` inference core.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

import io
import logging
import os
import pathlib
import sys
import threading
import time
from typing import Any, Optional
import uuid

_THIS_DIR = str(pathlib.Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from starlette.concurrency import run_in_threadpool

from engine import (
    API_KEY,
    MODEL_ID,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    get_gpu_info,
    is_pipeline_loaded,
    is_pipeline_warming,
    is_rmbg_loaded,
    preload_models,
    prepare_ref_image,
    run_inference_sync,
    run_rmbg_sync,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen21-gpu-worker")

app = FastAPI(title="Arangtik Qwen-Image-2.1-Turbo GPU Worker", version="4.0.0")

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _verify_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and (x_api_key or "").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


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
    transparent: bool = False,
    pre_clean: bool = False,
) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id]["status"] = "running"

    try:
        png_bytes, headers, _ = run_inference_sync(
            ref_img=ref_img,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
            transparent=transparent,
            pre_clean=pre_clean,
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
def _startup_preload() -> None:
    preload_models(background_qwen=True)


@app.get("/health")
def health(x_api_key: Optional[str] = Header(default=None)):
    _verify_api_key(x_api_key)
    gpu = get_gpu_info()
    return {
        "status": "ok",
        "model_id": MODEL_ID,
        "precision": "full_bfloat16",
        "model_loaded": is_pipeline_loaded(),
        "model_warming": is_pipeline_warming(),
        "rmbg_loaded": is_rmbg_loaded(),
        "pre_clean_supported": True,
        "workers_total": int(os.getenv("GPU_WORKERS_TOTAL", "1")),
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
    transparent: bool = Form(False),
    pre_clean: bool = Form(False),
    x_api_key: Optional[str] = Header(default=None),
):
    """Submit an async Qwen-Image-2.1-Turbo generation job."""
    _verify_api_key(x_api_key)
    _prune_old_jobs()

    raw_bytes = await image.read()
    await image.close()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image uploaded.")

    try:
        ref_img = prepare_ref_image(raw_bytes)
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
        args=(job_id, ref_img, prompt, steps, guidance_scale, seed, width, height, transparent, pre_clean),
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
    transparent: bool = Form(False),
    pre_clean: bool = Form(False),
    x_api_key: Optional[str] = Header(default=None),
):
    """Run synchronous Qwen-Image-2.1-Turbo generation in worker thread so /health stays responsive."""
    _verify_api_key(x_api_key)

    raw_bytes = await image.read()
    await image.close()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image uploaded.")

    try:
        ref_img = prepare_ref_image(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}") from exc

    try:
        png_bytes, headers, _ = await run_in_threadpool(
            run_inference_sync,
            ref_img=ref_img,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
            transparent=transparent,
            pre_clean=pre_clean,
        )
    except Exception as exc:
        logger.exception("Qwen-Image-2.1-Turbo inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"GPU inference failed: {exc}") from exc

    return Response(content=png_bytes, media_type="image/png", headers=headers)


@app.post("/remove-background")
async def remove_background_endpoint(
    image: UploadFile = File(..., description="Image to remove background."),
    x_api_key: Optional[str] = Header(default=None),
):
    """Fast standalone GPU background removal endpoint."""
    _verify_api_key(x_api_key)
    raw_bytes = await image.read()
    await image.close()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty image uploaded.")

    try:
        img = Image.open(io.BytesIO(raw_bytes))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}") from exc

    try:
        png_bytes, elapsed = await run_in_threadpool(run_rmbg_sync, img)
    except Exception as exc:
        logger.exception("GPU background removal failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"GPU background removal failed: {exc}") from exc

    gpu_info = get_gpu_info()
    headers = {
        "X-Inference-Time-Sec": str(elapsed),
        "X-GPU-Name": str(gpu_info["gpu_name"]),
    }
    return Response(content=png_bytes, media_type="image/png", headers=headers)
