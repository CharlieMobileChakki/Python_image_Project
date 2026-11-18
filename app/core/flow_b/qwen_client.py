"""Remote GPU Worker client for Flow B Qwen-Image-Edit generation.

Connects Server 1 (Contabo CPU API) to Server 2 (AIC Cloud RTX 3090 GPU Worker)
using an async job-submit + poll pattern (POST /jobs -> GET /jobs/{id} -> GET /jobs/{id}/result),
making Cloudflare 100s (HTTP 524) proxy timeouts completely impossible.
"""
from __future__ import annotations

import io
import logging
import os
import time
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


def _get_remote_url() -> str:
    return os.getenv("QWEN_REMOTE_URL", "").strip().rstrip("/")


def _get_api_key() -> str:
    return os.getenv("QWEN_API_KEY", "").strip()


def _get_timeout() -> float:
    try:
        return float(os.getenv("QWEN_TIMEOUT_SECONDS", "300"))
    except ValueError:
        return 300.0


def is_gpu_worker_configured() -> tuple[bool, str]:
    """Check whether remote GPU worker URL is configured in environment."""
    url = _get_remote_url()
    if not url:
        return (
            False,
            "QWEN_REMOTE_URL is not configured in .env. Set QWEN_REMOTE_URL=http://<GPU_SERVER_IP>:8001 to enable Flow B GPU generation.",
        )
    return True, url


def check_gpu_worker_health() -> dict[str, Any]:
    """Probe the remote GPU worker /health endpoint (non-blocking short timeout)."""
    configured, info = is_gpu_worker_configured()
    if not configured:
        return {"configured": False, "online": False, "detail": info}

    headers = {}
    api_key = _get_api_key()
    if api_key:
        headers["X-API-Key"] = api_key

    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{info}/health", headers=headers)
            if resp.status_code == 200:
                payload = resp.json()
                return {"configured": True, "online": True, "url": info, **payload}
            return {
                "configured": True,
                "online": False,
                "url": info,
                "status_code": resp.status_code,
            }
    except Exception as exc:
        return {"configured": True, "online": False, "url": info, "detail": str(exc)}


def generate_garment_via_gpu(
    reference_image: Image.Image,
    prompt: str,
    steps: int = 8,
    guidance_scale: float = 1.0,
    seed: int | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Submit reference photo + ChatGPT prompt to RTX 3090 GPU Worker and poll until PNG is ready.

    Uses short-lived HTTP calls (submit -> poll -> fetch) so Cloudflare Tunnel never hits HTTP 524.

    Args:
        reference_image: Original model-worn PIL image.
        prompt: Detailed garment extraction prompt (from ChatGPT Vision + studio rules).
        steps: Denoising inference steps on GPU (default 18 for fast, crisp RTX 3090 output).
        guidance_scale: Guidance scale for Qwen-Image-Edit.
        seed: Optional deterministic random seed.

    Returns:
        Tuple of (generated PIL Image, metadata dict).

    Raises:
        RuntimeError: If GPU worker is not configured, offline, or fails.
    """
    configured, remote_url = is_gpu_worker_configured()
    if not configured:
        raise RuntimeError(remote_url)

    ref_rgb = reference_image.convert("RGB")
    buf = io.BytesIO()
    ref_rgb.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    headers: dict[str, str] = {}
    api_key = _get_api_key()
    if api_key:
        headers["X-API-Key"] = api_key

    form_data: dict[str, str] = {
        "prompt": prompt,
        "steps": str(steps),
        "guidance_scale": str(guidance_scale),
    }
    if seed is not None:
        form_data["seed"] = str(seed)

    max_wait_sec = _get_timeout()
    t_start = time.perf_counter()

    logger.info("Submitting Flow B async job to GPU worker: %s/jobs", remote_url)
    try:
        with httpx.Client(timeout=30.0) as client:
            # 1. Submit job (completes in < 0.5s — immune to Cloudflare 524!)
            submit_resp = client.post(
                f"{remote_url}/jobs",
                headers=headers,
                files={"image": ("reference.png", img_bytes, "image/png")},
                data=form_data,
            )
            if submit_resp.status_code != 200:
                raise RuntimeError(f"GPU Worker /jobs returned HTTP {submit_resp.status_code}: {submit_resp.text[:300]}")

            job_id = submit_resp.json()["job_id"]
            logger.info("GPU job %s queued; polling status...", job_id)

            # 2. Poll job status every 2.5s until done or error
            while True:
                if time.perf_counter() - t_start > max_wait_sec:
                    raise RuntimeError(f"GPU Worker job {job_id} timed out after {max_wait_sec}s.")

                time.sleep(2.5)
                status_resp = client.get(f"{remote_url}/jobs/{job_id}", headers=headers)
                if status_resp.status_code != 200:
                    raise RuntimeError(f"GPU Worker status check failed ({status_resp.status_code}): {status_resp.text[:200]}")

                state = status_resp.json()
                job_status = state.get("status")
                if job_status == "done":
                    break
                if job_status == "error":
                    raise RuntimeError(f"GPU Worker job failed: {state.get('error', 'Unknown GPU error')}")

            # 3. Fetch final PNG bytes
            result_resp = client.get(f"{remote_url}/jobs/{job_id}/result", headers=headers, timeout=60.0)
            if result_resp.status_code != 200:
                raise RuntimeError(f"GPU Worker result fetch failed ({result_resp.status_code}): {result_resp.text[:200]}")

    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot connect to GPU Worker at {remote_url}. Ensure the AIC Cloud RTX 3090 instance is started."
        ) from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"GPU Worker communication error: {exc}") from exc

    try:
        generated_img = Image.open(io.BytesIO(result_resp.content)).convert("RGBA")
    except Exception as exc:
        raise RuntimeError(f"Invalid image received from GPU Worker: {exc}") from exc

    metadata = {
        "model": result_resp.headers.get("X-Model-Name", "Qwen-Image-Edit"),
        "inference_time_sec": result_resp.headers.get("X-Inference-Time-Sec", "0"),
        "gpu_name": result_resp.headers.get("X-GPU-Name", "RTX 3090"),
        "seed": result_resp.headers.get("X-Seed", str(seed or "")),
    }
    return generated_img, metadata
