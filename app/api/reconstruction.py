"""FastAPI router for Direct Qwen-Image-2.1-Turbo Garment Generation."""
from __future__ import annotations

import io
import logging
import os
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

from app.core.flow_b import (
    analyze_garment_and_build_prompts,
    generate_garment_via_gpu,
    is_gpu_worker_configured,
)

logger = logging.getLogger(__name__)
reconstruction_router = APIRouter(tags=["Qwen Studio Generation"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def _decode_pil_rgb(data: bytes, filename: str) -> Image.Image:
    """Decode uploaded image bytes into EXIF-corrected PIL RGB."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image {filename}: {exc}") from exc


async def _read_upload(upload: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    raw = await upload.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds upload limit.")
    if not raw:
        raise HTTPException(status_code=400, detail=f"{upload.filename} is empty.")
    return raw


def _run_qwen_pipeline_sync(
    original_pil: Image.Image,
    target_garment: str,
    custom_prompt: Optional[str],
    steps: int,
    guidance_scale: float,
    seed: Optional[int],
) -> tuple[bytes, dict]:
    """Execute ChatGPT Vision -> Remote GPU Qwen -> Direct 1024x1024 White-BG PNG."""
    plans = analyze_garment_and_build_prompts(
        image_rgb=original_pil,
        target_garment=target_garment,
        custom_prompt=custom_prompt,
    )
    primary = plans[0]

    apparel_size = max(512, int(os.getenv("QWEN_APPAREL_SIZE", "1024")))
    generated_raw, gpu_meta = generate_garment_via_gpu(
        reference_image=original_pil,
        prompt=primary["prompt"],
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        width=apparel_size,
        height=apparel_size,
    )

    direct_img = generated_raw.convert("RGB")
    if direct_img.size != (1024, 1024):
        direct_img = direct_img.resize((1024, 1024), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    direct_img.save(buf, format="PNG", compress_level=1)
    meta = {
        "model": gpu_meta.get("model", "Qwen/Qwen-Image-2.1-Turbo"),
        "gpu_name": gpu_meta.get("gpu_name", "Remote GPU"),
        "inference_time_sec": str(gpu_meta.get("inference_time_sec", "0")),
        "seed": str(gpu_meta.get("seed", "")),
        "prompt_source": primary.get("prompt_source", "default"),
        "category": primary.get("category", "dress"),
        "recognized_label": primary.get("recognized_label", "Studio Garment"),
    }
    return buf.getvalue(), meta


@reconstruction_router.post(
    "/qwen-image",
    summary="Direct ChatGPT Vision + Remote GPU Qwen-Image-2.1-Turbo Garment Generation",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def qwen_image_endpoint(
    original_image: UploadFile = File(..., description="Original reference photograph."),
    target_garment: str = Form("all", description="Target garment filter: 'all', 'upper_clothes', 'pants', 'dress', 'skirt', 'saree', 'lehenga'."),
    prompt: Optional[str] = Form(None, description="Optional custom prompt override (auto-generated via ChatGPT Vision if omitted)."),
    steps: int = Form(8, description="Denoising inference steps on GPU."),
    guidance_scale: float = Form(1.0, description="Guidance scale."),
    seed: Optional[int] = Form(None, description="Optional random seed."),
):
    """Generate a direct 1024x1024 studio white-background garment PNG from a photo."""
    configured, info = is_gpu_worker_configured()
    if not configured:
        raise HTTPException(status_code=503, detail=info)

    raw_bytes = await _read_upload(original_image)
    original_pil = _decode_pil_rgb(raw_bytes, original_image.filename or "original_image")

    try:
        png_bytes, meta = await run_in_threadpool(
            _run_qwen_pipeline_sync,
            original_pil,
            target_garment,
            prompt,
            steps,
            guidance_scale,
            seed,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Flow B Qwen generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Qwen garment generation failed: {exc}") from exc
    finally:
        await original_image.close()

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Model-Name": meta["model"],
            "X-Model-Device": meta["gpu_name"],
            "X-Inference-Time-Sec": meta["inference_time_sec"],
            "X-Prompt-Source": meta["prompt_source"],
            "X-Garment-Category": meta["category"],
            "X-Garment-Label": meta["recognized_label"],
            "X-Seed": meta["seed"],
        },
    )
