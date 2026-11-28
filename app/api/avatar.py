"""API router for AI Stylized Avatar Profile Picture Generator ("Avatar Studio")."""
from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool

from app.api.routes import (
    _resolve_base_url,
    get_semaphore,
    validate_and_read_image,
)
from app.core.avatar import process_avatar_pipeline
from app.core.pipeline import OUTPUT_DIR
from app.schemas import AvatarGenerationResponse

logger = logging.getLogger(__name__)

avatar_router = APIRouter(tags=["Avatar Studio"])


@avatar_router.post(
    "/generate-avatar",
    response_model=AvatarGenerationResponse,
    response_model_exclude_none=True,
    summary="Generate Stylized 2D Vector Avatar Profile Picture",
    description=(
        "Transform an uploaded user photo into an iconic, charming 2D digital vector art avatar "
        "profile picture framed by a pastel circular halo badge with minimalist hand-drawn doodle accents. "
        "Preserves exact facial identity, hair texture/volume, eyeglasses, facial hair, and upper outfit."
    ),
)
@avatar_router.post(
    "/avatar-studio",
    response_model=AvatarGenerationResponse,
    response_model_exclude_none=True,
    summary="Generate Stylized 2D Vector Avatar Profile Picture (Alias)",
    include_in_schema=False,
)
async def generate_avatar_endpoint(
    request: Request,
    file: UploadFile = File(..., description="User photo (portrait, selfie, or full photo)"),
    style: str = Form("vector_art", description="Style: 'vector_art', 'digital_illustration', 'anime_portrait'"),
    halo_color: str = Form("auto", description="Pastel halo badge color: 'auto', 'soft_blue', 'periwinkle', 'lavender', 'blush_pink', 'mint_green', 'warm_amber'"),
    doodle_accent: str = Form("auto", description="Doodle accent: 'auto', 'crown', 'heart', 'sparks', 'stars', 'none'"),
    quality: str = Form("standard", description="Quality: 'standard' or 'high' (1024x1024)"),
    response_format: str = Form("json", description="'json' (default for API / UI) or 'image_direct' for binary PNG stream"),
    include_base64: bool = Form(False, description="Include Base64 data URI in JSON"),
    custom_prompt: str | None = Form(None, description="Optional extra custom creative direction"),
) -> Any:
    """Direct API to generate a stylized vector avatar profile card from a user photo."""
    try:
        image_bytes = await validate_and_read_image(file)
    finally:
        await file.close()

    client_host = request.client.host if request.client else "unknown"
    t_req = time.perf_counter()
    logger.info(
        "Incoming avatar request | client=%s bytes=%d style=%s halo=%s doodle=%s format=%s",
        client_host,
        len(image_bytes),
        style,
        halo_color,
        doodle_accent,
        response_format,
    )

    try:
        async with get_semaphore():
            result = await run_in_threadpool(
                process_avatar_pipeline,
                image_bytes=image_bytes,
                style=style,
                halo_color=halo_color,
                doodle_accent=doodle_accent,
                quality=quality,
                include_base64=include_base64,
                custom_prompt=custom_prompt,
            )
    except ValueError as exc:
        logger.warning("Validation error on %s (%s): %s", request.url.path, client_host, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("GPU/Service error on %s (%s): %s", request.url.path, client_host, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled error on %s (%s): %s", request.url.path, client_host, exc)
        raise HTTPException(status_code=500, detail=f"Avatar pipeline error: {exc}") from exc

    logger.info(
        "Completed avatar request in %.2fs | job_id=%s file=%s",
        time.perf_counter() - t_req,
        result.get("job_id"),
        result.get("avatar_url"),
    )

    # Check for direct image streaming request
    accept_hdr = request.headers.get("accept", "").lower()
    wants_image = (
        response_format.lower() in ("image_direct", "direct", "image", "stream", "png")
        or ("image/" in accept_hdr and "application/json" not in accept_hdr)
    )

    if wants_image and result.get("output_png_url"):
        png_path = OUTPUT_DIR / Path(result["output_png_url"]).name
        if png_path.exists():
            return FileResponse(
                path=png_path,
                media_type="image/png",
                filename=png_path.name,
                headers={
                    "X-Avatar-Style": result.get("style", "vector_art"),
                    "X-Avatar-Halo": result.get("halo_color", "soft_blue"),
                    "X-Avatar-Doodle": result.get("doodle_accent", "crown"),
                    "X-Execution-Time-Ms": str(result.get("execution_time_ms", 0)),
                    "X-Canvas-Size": "1024x1024",
                },
            )

    # Format absolute URLs
    base_url = _resolve_base_url(request)
    raw_url = result.get("avatar_url", "")
    full_url = f"{base_url}{raw_url}" if raw_url.startswith("/") else raw_url

    result["avatar_url"] = full_url
    result["output_png_url"] = full_url

    return result
