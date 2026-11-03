"""FastAPI router for optional SDXL Garment Reconstruction."""
from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from app.services.inpainting.garment_reconstruction import get_sdxl_reconstruction_engine
from app.services.inpainting.reconstruction_utils import load_pil_image

logger = logging.getLogger(__name__)
reconstruction_router = APIRouter(tags=["Garment Reconstruction"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


async def _read_upload(upload: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    raw = await upload.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds upload limit.")
    if not raw:
        raise HTTPException(status_code=400, detail=f"{upload.filename} is empty.")
    return raw


@reconstruction_router.post(
    "/reconstruct-garment",
    summary="Reconstruct missing garment fabric using SDXL Inpainting + IP-Adapter Plus (Optional)",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def reconstruct_garment_endpoint(
    original_image: UploadFile = File(..., description="Original person photo used as garment reference."),
    damaged_garment: UploadFile = File(..., description="Person-removed garment PNG, preferably transparent."),
    repair_mask: UploadFile = File(..., description="Black/white mask: white=missing garment to generate, black=preserve."),
    garment_reference: Optional[UploadFile] = File(None, description="Optional tight crop of visible garment from original photo."),
    steps: Optional[int] = Form(None),
    guidance_scale: Optional[float] = Form(None),
    strength: Optional[float] = Form(None),
    ip_adapter_scale: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
):
    """Optional deep generative reconstruction endpoint using SDXL + IP-Adapter Plus.

    Falls back cleanly with HTTP 503 when running in CPU-only or diffusers-free environments,
    leaving the zero-overhead LaMa ONNX engine as primary.
    """
    engine = get_sdxl_reconstruction_engine()
    is_avail, reason = engine.is_available()
    if not is_avail:
        raise HTTPException(
            status_code=503,
            detail=f"SDXL Garment Reconstruction is optional and unavailable in this environment ({reason}). Use default LaMa ONNX inpainting.",
        )

    try:
        engine.load()
    except Exception as exc:
        logger.exception("Failed to load SDXL Reconstruction engine: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to initialize SDXL reconstruction model: {exc}",
        ) from exc

    try:
        original_raw = await _read_upload(original_image)
        damaged_raw = await _read_upload(damaged_garment)
        mask_raw = await _read_upload(repair_mask)
        reference_raw = await _read_upload(garment_reference) if garment_reference else None

        original_pil = load_pil_image(original_raw, original_image.filename or "original_image")
        damaged_pil = load_pil_image(damaged_raw, damaged_garment.filename or "damaged_garment")
        mask_pil = load_pil_image(mask_raw, repair_mask.filename or "repair_mask")
        reference_pil = load_pil_image(reference_raw, garment_reference.filename or "garment_reference") if reference_raw else None

        result = engine.reconstruct(
            original_reference=original_pil,
            damaged_garment=damaged_pil,
            repair_mask=mask_pil,
            garment_reference=reference_pil,
            steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            ip_adapter_scale=ip_adapter_scale,
            seed=seed,
        )

        buffer = io.BytesIO()
        result.image.save(buffer, format="PNG", optimize=True)
        png_bytes = buffer.getvalue()

        headers = {
            "X-Inference-Seconds": str(result.inference_seconds),
            "X-Model-Device": result.device,
            "X-Seed": str(result.seed),
            "X-Output-Width": str(result.width),
            "X-Output-Height": str(result.height),
            "X-Output-Format": "PNG-RGBA",
        }
        return Response(content=png_bytes, media_type="image/png", headers=headers)

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Garment reconstruction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Reconstruction execution failed: {exc}") from exc
    finally:
        await original_image.close()
        await damaged_garment.close()
        await repair_mask.close()
        if garment_reference:
            await garment_reference.close()

