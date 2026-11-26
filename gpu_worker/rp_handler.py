"""RunPod Serverless Queue Worker for Arangtik Studio (Qwen-Image-2.1-Turbo).

Entry point for RunPod Serverless queue endpoints (`run` / `runsync`).
When launched outside RunPod (or with `WORKER_MODE=http`), serves `server:app` via Uvicorn.
Shares all inference logic cleanly with `engine.py` (zero duplicated code).
"""
from __future__ import annotations

import base64
import logging
import os
import pathlib
import subprocess
import sys
from typing import Any

_THIS_DIR = str(pathlib.Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import runpod

from engine import (
    MODEL_ID,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    get_gpu_info,
    is_pipeline_loaded,
    preload_models,
    prepare_ref_image,
    run_inference_sync,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("runpod-qwen21-worker")


def _decode_base64_image(b64_payload: str) -> bytes:
    """Decode raw base64 or data-URI encoded image string into bytes."""
    cleaned = (b64_payload or "").strip()
    if not cleaned:
        raise ValueError("Missing 'image_base64' in job input.")
    if "," in cleaned and cleaned[:64].lower().startswith("data:"):
        cleaned = cleaned.split(",", 1)[1]
    return base64.b64decode(cleaned)


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Handle RunPod Serverless queued inference jobs (`generate`, `health`)."""
    job_input = job.get("input") or {}
    action = str(job_input.get("action") or "generate").strip().lower()

    if action == "health":
        return {
            "status": "ok",
            "model_id": MODEL_ID,
            "precision": "full_bfloat16",
            "model_loaded": is_pipeline_loaded(),
            "output_size": f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
            **get_gpu_info(),
        }

    try:
        raw_bytes = _decode_base64_image(str(job_input.get("image_base64") or ""))
    except Exception as exc:
        return {"error": f"Invalid image_base64 input: {exc}"}

    prompt = str(job_input.get("prompt") or "").strip()
    if not prompt:
        return {"error": "Missing 'prompt' in job input."}

    try:
        ref_img = prepare_ref_image(raw_bytes)
        steps = int(job_input.get("steps") or 8)
        guidance_scale = float(job_input.get("guidance_scale") or 1.0)
        raw_seed = job_input.get("seed")
        seed = int(raw_seed) if raw_seed is not None else None
        raw_w = job_input.get("width")
        raw_h = job_input.get("height")
        width = int(raw_w) if raw_w else None
        height = int(raw_h) if raw_h else None

        png_bytes, _, metadata = run_inference_sync(
            ref_img=ref_img,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
        )
        return {
            "image_base64": base64.b64encode(png_bytes).decode("ascii"),
            "metadata": metadata,
        }
    except Exception as exc:
        logger.exception("RunPod Qwen-Image-2.1-Turbo inference error: %s", exc)
        return {"error": f"GPU inference failed: {exc}"}


if __name__ == "__main__":
    mode = os.getenv("WORKER_MODE", "auto").strip().lower()
    if mode == "http":
        port = os.getenv("PORT", "8001").strip() or "8001"
        raise SystemExit(
            subprocess.call([sys.executable, "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", port])
        )

    if os.getenv("RUNPOD_SKIP_STARTUP_PRELOAD", "false").strip().lower() not in ("1", "true", "yes"):
        logger.info("Preloading Qwen-Image-2.1-Turbo before accepting RunPod jobs...")
        preload_models(background_qwen=False)
    runpod.serverless.start({"handler": handler})
