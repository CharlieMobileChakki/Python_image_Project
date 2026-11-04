"""FastAPI endpoint router for Qwen-Image-2.1 Garment Generation."""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Optional
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

from app.core.pipeline.pipeline_config import OUTPUT_DIR
from app.experimental.qwen_generation import get_qwen_garment_engine
from app.schemas.qwen_garment_generation import QwenGarmentGenerationResponse

logger = logging.getLogger(__name__)

qwen_router = APIRouter(tags=["Qwen-Image-2.1 Garment Generation"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


async def _read_and_validate_upload(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Supported image formats are JPG, PNG, and WEBP.")
    image_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 25 MB upload limit.")
    return image_bytes


@qwen_router.post(
    "/qwen-generate-garment",
    response_model=QwenGarmentGenerationResponse,
    summary="Generate standalone garment from person photo using Qwen-Image-2.1",
)
async def qwen_generate_garment_endpoint(
    original_image: UploadFile = File(..., description="Photograph of person wearing the garment."),
    prompt: Optional[str] = Form(None, description="Optional custom text prompt. Default garment prompt used if empty."),
    resolution: int = Form(1024, ge=512, le=2048, description="Output image resolution (width & height)."),
    inference_steps: int = Form(28, ge=8, le=50, description="Denoising steps (default: 28)."),
    seed: Optional[int] = Form(None, description="Reproducible random seed."),
    transparent_background: bool = Form(True, description="Enable native transparency / RGBA output."),
    response_format: str = Form("json", description="'json' for full metadata or 'image_direct' for binary PNG stream."),
) -> Any:
    """Execute Qwen-Image-2.1 garment synthesis conditioned on reference image.

    - Reconstructs occluded fabric where arms/hands overlap.
    - Completely removes the human wearer without generating mannequins or hangers.
    - Generates native transparent RGBA PNG or clean white background.
    """
    image_bytes = await _read_and_validate_upload(original_image)
    try:
        raw_pil = Image.open(io.BytesIO(image_bytes))
        raw_pil = ImageOps.exif_transpose(raw_pil).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid image file: {exc}") from exc
    finally:
        await original_image.close()

    engine = get_qwen_garment_engine()
    is_avail, reason = engine.is_available()

    if not is_avail:
        # Development host lacking CUDA/diffusers support: return transparent informational response
        logger.warning("Qwen-Image-2.1 called on unsupported environment: %s", reason)
        raise HTTPException(
            status_code=503,
            detail=f"Qwen-Image-2.1 is unavailable in this development environment ({reason}). Ready for deployment on RTX 4090 GPU server.",
        )

    try:
        result = await run_in_threadpool(
            engine.generate,
            reference_image=raw_pil,
            prompt=prompt,
            resolution=resolution,
            inference_steps=inference_steps,
            seed=seed,
            transparent_background=transparent_background,
        )
    except Exception as exc:
        logger.exception("Qwen-Image-2.1 generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Qwen-Image-2.1 execution failed: {exc}") from exc

    # Save output to outputs directory
    job_id = uuid.uuid4().hex[:10]
    filename = f"{job_id}_qwen_garment.png"
    out_path = OUTPUT_DIR / filename
    result.image.save(out_path, format="PNG", optimize=True)

    if response_format.lower() == "image_direct":
        return FileResponse(
            path=out_path,
            media_type="image/png",
            filename=filename,
            headers={
                "X-Model-Name": result.model_name,
                "X-Inference-Seconds": str(result.inference_time_seconds),
                "X-Output-Width": str(result.width),
                "X-Output-Height": str(result.height),
                "X-Has-Alpha": str(result.has_alpha_channel),
            },
        )

    # Base64 encoding if requested
    import base64
    buf = io.BytesIO()
    result.image.save(buf, format="PNG")
    b64_str = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    return QwenGarmentGenerationResponse(
        success=(result.status == "success"),
        status=result.status,
        model_name=result.model_name,
        image_format="PNG",
        image_width=result.width,
        image_height=result.height,
        has_alpha_channel=result.has_alpha_channel,
        inference_time_seconds=result.inference_time_seconds,
        gpu_memory_peak_mb=result.gpu_memory_peak_mb,
        output_png_url=f"/api/v1/outputs/{filename}",
        output_png_base64=b64_str,
        error_detail=result.error_detail,
    )

