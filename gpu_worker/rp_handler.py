"""RunPod Serverless Queue Worker for Arangtik Flow B (Qwen-Image-2.1-Turbo + RMBG-1.4).

Entry point for RunPod Serverless queue endpoints (`run` / `runsync`).
Shares all inference logic cleanly with `engine.py` (zero duplicated code).
"""
from __future__ import annotations

import base64
import io
import logging
import os
import pathlib
import sys
from typing import Any

_THIS_DIR = str(pathlib.Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import runpod
from PIL import Image

from engine import (
    MODEL_ID,
    get_gpu_info,
    is_pipeline_loaded,
    is_rmbg_loaded,
    preload_models,
    prepare_ref_image,
    run_inference_sync,
    run_rmbg_sync,
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


def _to_bool(val: Any, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes")


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Handle RunPod Serverless queued inference jobs (`generate`, `remove_background`, `health`)."""
    job_input = job.get("input") or {}
    action = str(job_input.get("action") or "generate").strip().lower()

    if action == "health":
        return {
            "status": "ok",
            "model_id": MODEL_ID,
            "precision": "full_bfloat16",
            "model_loaded": is_pipeline_loaded(),
            "rmbg_loaded": is_rmbg_loaded(),
            **get_gpu_info(),
        }

    try:
        raw_bytes = _decode_base64_image(str(job_input.get("image_base64") or ""))
    except Exception as exc:
        return {"error": f"Invalid image_base64 input: {exc}"}

    if action == "remove_background":
        try:
            img = Image.open(io.BytesIO(raw_bytes))
            png_bytes, elapsed = run_rmbg_sync(img)
            gpu_info = get_gpu_info()
            return {
                "image_base64": base64.b64encode(png_bytes).decode("ascii"),
                "metadata": {
                    "model": "briaai/RMBG-1.4",
                    "gpu_name": gpu_info["gpu_name"],
                    "vram_total_gb": gpu_info["vram_total_gb"],
                    "inference_time_sec": str(elapsed),
                },
            }
        except Exception as exc:
            logger.exception("RunPod remove_background error: %s", exc)
            return {"error": f"GPU background removal failed: {exc}"}

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
        transparent = _to_bool(job_input.get("transparent"), default=False)
        pre_clean = _to_bool(job_input.get("pre_clean"), default=False)

        png_bytes, _, metadata = run_inference_sync(
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
        return {
            "image_base64": base64.b64encode(png_bytes).decode("ascii"),
            "metadata": metadata,
        }
    except Exception as exc:
        logger.exception("RunPod Qwen-Image-2.1-Turbo inference error: %s", exc)
        return {"error": f"GPU inference failed: {exc}"}


if __name__ == "__main__":
    # Warm VRAM models synchronously on worker startup unless explicitly deferred
    if os.getenv("RUNPOD_SKIP_STARTUP_PRELOAD", "false").strip().lower() not in ("1", "true", "yes"):
        logger.info("Preloading RMBG-1.4 and Qwen-Image-2.1-Turbo before accepting RunPod jobs...")
        preload_models(background_qwen=False)
    runpod.serverless.start({"handler": handler})
