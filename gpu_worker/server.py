"""FastAPI HTTP Microservice for Arangtik Studio (Qwen/Qwen-Image-2.1-Turbo).

Used for HTTP-based GPU servers (RunPod Pods, Vast.ai, Dedicated Multi-GPU Nodes, Modal ASGI).
RunPod Serverless Queue mode (`rp_handler.py`) shares the exact same `engine.py` inference core.

Run:
    uvicorn server:app --host 0.0.0.0 --port 8001
"""
from __future__ import annotations

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
    preload_models,
    prepare_ref_image,
    prepare_ref_images,
    run_inference_sync,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen21-gpu-worker")

app = FastAPI(title="Arangtik Qwen-Image-2.1-Turbo GPU Worker", version="4.0.0")

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _verify_api_key(x_api_key: Optional[str]) -> None:
    if API_KEY and (x_api_key or "").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")


async def _read_uploaded_ref_images(
    image: Optional[UploadFile],
    images: Optional[list[UploadFile]],
) -> Image.Image | list[Image.Image]:
    """Decode one or multiple uploaded reference images cleanly via shared engine helpers."""
    uploads = [f for f in ([image] if image else []) + list(images or []) if f is not None]
    if not uploads:
        raise HTTPException(status_code=400, detail="At least one reference image is required.")

    raw_bytes_list: list[bytes] = []
    for upload in uploads[:10]:
        try:
            raw = await upload.read()
        finally:
            await upload.close()
        if not raw:
            raise HTTPException(status_code=400, detail=f"Empty image uploaded ({upload.filename or 'file'}).")
        raw_bytes_list.append(raw)

    try:
        return (
            prepare_ref_images(raw_bytes_list)
            if len(raw_bytes_list) > 1
            else prepare_ref_image(raw_bytes_list[0])
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image: {exc}") from exc


def _prune_old_jobs() -> None:
    now = time.time()
    with _jobs_lock:
        stale = [jid for jid, j in _jobs.items() if now - j["created_at"] > 600]
        for jid in stale:
            _jobs.pop(jid, None)


def _background_job_runner(
    job_id: str,
    ref_img: Image.Image | list[Image.Image],
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
        png_bytes, headers, _ = run_inference_sync(
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
        "workers_total": int(os.getenv("GPU_WORKERS_TOTAL", "1")),
        "output_size": f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        **gpu,
    }


@app.post("/jobs")
async def submit_job(
    image: Optional[UploadFile] = File(None),
    images: Optional[list[UploadFile]] = File(None),
    prompt: str = Form(...),
    steps: int = Form(12),
    guidance_scale: float = Form(1.4),
    seed: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    x_api_key: Optional[str] = Header(default=None),
):
    """Submit an async Qwen-Image-2.1-Turbo generation job."""
    _verify_api_key(x_api_key)
    _prune_old_jobs()

    ref_img = await _read_uploaded_ref_images(image, images)

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
    image: Optional[UploadFile] = File(None, description="Primary reference image."),
    images: Optional[list[UploadFile]] = File(None, description="Multiple reference images (up to 10) for multi-image generation."),
    prompt: str = Form(..., description="Detailed studio generation prompt from ChatGPT Vision."),
    steps: int = Form(12),
    guidance_scale: float = Form(1.4),
    seed: Optional[int] = Form(None),
    width: Optional[int] = Form(None),
    height: Optional[int] = Form(None),
    x_api_key: Optional[str] = Header(default=None),
):
    """Run synchronous Qwen-Image-2.1-Turbo generation in worker thread so /health stays responsive."""
    _verify_api_key(x_api_key)

    ref_img = await _read_uploaded_ref_images(image, images)

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
        )
    except Exception as exc:
        logger.exception("Qwen-Image-2.1-Turbo inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"GPU inference failed: {exc}") from exc

    return Response(content=png_bytes, media_type="image/png", headers=headers)
