"""Remote GPU Worker client for Flow B Qwen-Image-Edit generation.

Connects Server 1 (Contabo CPU API) to Server 2 (AIC Cloud RTX 3090 GPU Worker)
over HTTP with API Key authentication. Zero heavy diffusers/GPU weights are loaded
on Server 1.
"""
from __future__ import annotations

import io
import logging
import os
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
        return float(os.getenv("QWEN_TIMEOUT_SECONDS", "180"))
    except ValueError:
        return 180.0


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
    steps: int = 28,
    guidance_scale: float = 4.0,
    seed: int | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Send reference photo + ChatGPT prompt to the RTX 3090 GPU Worker and receive generated garment PNG.

    Args:
        reference_image: Original model-worn PIL image.
        prompt: Detailed garment extraction prompt (from ChatGPT Vision + studio rules).
        steps: Denoising inference steps on GPU.
        guidance_scale: Guidance scale for Qwen-Image-Edit.
        seed: Optional deterministic random seed.

    Returns:
        Tuple of (generated PIL Image, metadata dict).

    Raises:
        RuntimeError: If GPU worker is not configured, offline, or returns an error.
    """
    configured, remote_url = is_gpu_worker_configured()
    if not configured:
        raise RuntimeError(remote_url)

    # Encode reference image as high-quality PNG/JPEG for upload
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

    endpoint = f"{remote_url}/generate"
    timeout_sec = _get_timeout()

    logger.info("Sending Flow B garment generation request to GPU worker: %s", endpoint)
    try:
        with httpx.Client(timeout=timeout_sec) as client:
            resp = client.post(
                endpoint,
                headers=headers,
                files={"image": ("reference.png", img_bytes, "image/png")},
                data=form_data,
            )
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"Cannot connect to GPU Worker at {remote_url}. Ensure the AIC Cloud RTX 3090 instance is started and port 8001 is open."
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(
            f"GPU Worker timed out after {timeout_sec}s while generating garment."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"GPU Worker communication error: {exc}") from exc

    if resp.status_code != 200:
        detail = resp.text[:300]
        raise RuntimeError(f"GPU Worker returned HTTP {resp.status_code}: {detail}")

    try:
        generated_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception as exc:
        raise RuntimeError(f"Invalid image received from GPU Worker: {exc}") from exc

    metadata = {
        "model": resp.headers.get("X-Model-Name", "Qwen-Image-Edit"),
        "inference_time_sec": resp.headers.get("X-Inference-Time-Sec", "0"),
        "gpu_name": resp.headers.get("X-GPU-Name", "RTX 3090"),
        "seed": resp.headers.get("X-Seed", str(seed or "")),
    }
    return generated_img, metadata

