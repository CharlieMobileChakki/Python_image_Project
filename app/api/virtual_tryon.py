"""API router for Multi-Garment & Accessory Virtual Try-On endpoint."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.api.routes import _resolve_base_url, get_semaphore, validate_and_read_image
from app.core.pipeline import process_virtual_tryon_pipeline
from app.schemas import VirtualTryOnResponse

logger = logging.getLogger(__name__)
virtual_tryon_router = APIRouter()


async def _read_upload_group(
    uploads: list[UploadFile] | None,
    role_hint: str,
    max_items: int,
) -> list[dict[str, Any]]:
    """Validate and decode a group of uploaded files cleanly, always closing file handles."""
    if not uploads:
        return []
    entries: list[dict[str, Any]] = []
    for upload in uploads[:max_items]:
        try:
            raw = await validate_and_read_image(upload)
        finally:
            await upload.close()
        entries.append({
            "bytes": raw,
            "role_hint": role_hint,
            "filename": upload.filename or f"{role_hint}.jpg",
        })
    return entries


@virtual_tryon_router.post(
    "/virtual-tryon",
    response_model=VirtualTryOnResponse,
    summary="Virtual Try-On & AI Outfit Stylist",
    description=(
        "Analyze uploaded user avatar, multiple wardrobe garments (shirts, pants, dresses, etc.), "
        "and accessories (watch, shoes, eyewear, jewelry, bags) via ChatGPT Vision, curate stylish "
        "combinations, and generate photorealistic virtual try-on looks preserving the user's exact face."
    ),
)
async def virtual_tryon(
    request: Request,
    response: Response,
    avatar: UploadFile = File(..., description="User portrait/avatar image whose exact face must be preserved."),
    garments: list[UploadFile] | None = File(None, description="Wardrobe garment images (shirts, pants, dresses, kurtas, etc.)."),
    tops: list[UploadFile] | None = File(None, description="Optional upper-body garments (shirts, t-shirts, jackets, tops)."),
    bottoms: list[UploadFile] | None = File(None, description="Optional lower-body garments (pants, jeans, trousers, skirts)."),
    accessories: list[UploadFile] | None = File(None, description="Optional accessories (watch, shoes, sunglasses, bag, jewelry)."),
    max_combinations: int = Form(2, ge=1, le=4, description="Maximum curated outfit combinations to generate (1 to 4)."),
    occasion: str = Form("smart_casual", description="Target styling occasion."),
    photography_style: str = Form("auto", description="Photography aesthetic: auto, premium_studio, editorial_lookbook, natural_casual, traditional_royal, streetwear_lifestyle."),
    pose_preference: str = Form("auto", description="Pose style: auto (natural contrapposto), relaxed_casual, confident_executive, traditional_graceful, classic_catalog."),
    quality: str = Form("auto", description="Quality preset: auto (832x1152 12-step), high (896x1280 14-step), ultra (1024x1344 18-step), standard (832x1152 10-step)."),
    custom_prompt: str | None = Form(None, description="Optional extra styling prompt note."),
    include_base64: bool = Form(False),
    seed: int | None = Form(None),
):
    """Curate outfit pairings from multiple garments + accessories and render avatar try-on images."""
    try:
        avatar_bytes = await validate_and_read_image(avatar)
    finally:
        await avatar.close()

    garment_entries = (
        await _read_upload_group(tops, role_hint="top", max_items=6)
        + await _read_upload_group(bottoms, role_hint="bottom", max_items=6)
        + await _read_upload_group(garments, role_hint="auto", max_items=8)
    )[:12]
    if not garment_entries:
        raise HTTPException(
            status_code=400,
            detail="Please upload at least one garment photo in 'garments', 'tops', or 'bottoms'.",
        )

    accessory_entries = await _read_upload_group(accessories, role_hint="accessory", max_items=6)
    public_base = _resolve_base_url(request)

    try:
        async with get_semaphore():
            result = await run_in_threadpool(
                process_virtual_tryon_pipeline,
                avatar_bytes=avatar_bytes,
                garment_entries=garment_entries,
                accessory_entries=accessory_entries,
                max_combinations=max_combinations,
                occasion=occasion,
                custom_prompt=custom_prompt,
                base_url=public_base,
                include_base64=include_base64,
                seed=seed,
                photography_style=photography_style,
                pose_preference=pose_preference,
                quality_preset=quality,
            )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Virtual try-on execution error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during virtual try-on synthesis")
        raise HTTPException(status_code=500, detail=f"Virtual try-on failed: {exc}") from exc

    first_url = None
    if result.get("combinations"):
        first_url = result["combinations"][0].get("url") or result["combinations"][0].get("output_png_url")
    result.setdefault("image_url", first_url)
    result.setdefault("output_png_url", first_url)

    response.headers["X-Job-Id"] = result["job_id"]
    response.headers["X-Total-Combinations"] = str(result["total_combinations"])
    response.headers["X-Execution-Time-Ms"] = str(result["execution_time_ms"])
    return result
