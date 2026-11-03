from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from PIL import Image
import torch

from app.services.image_utils import load_pil_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Garment Reconstruction"])


async def _read_upload(upload: UploadFile, max_bytes: int) -> bytes:
    raw = await upload.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds upload limit.")
    if not raw:
        raise HTTPException(status_code=400, detail=f"{upload.filename} is empty.")
    return raw


@router.post(
    "/reconstruct-garment",
    summary="Reconstruct missing garment fabric using SDXL Inpainting + IP-Adapter Plus",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def reconstruct_garment(
    request: Request,
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
    settings = request.app.state.settings
    engine = request.app.state.reconstruction_engine
    max_bytes = settings.max_upload_mb * 1024 * 1024

    if not engine.is_loaded():
        raise HTTPException(status_code=503, detail="Reconstruction model is not ready.")

    try:
        original_raw = await _read_upload(original_image, max_bytes)
        damaged_raw = await _read_upload(damaged_garment, max_bytes)
        mask_raw = await _read_upload(repair_mask, max_bytes)
        reference_raw = await _read_upload(garment_reference, max_bytes) if garment_reference else None

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
    except torch.cuda.OutOfMemoryError as exc:
        logger.exception("CUDA out of memory during garment reconstruction")
        raise HTTPException(
            status_code=507,
            detail="GPU ran out of memory. Reduce MAX_IMAGE_SIDE or enable USE_MODEL_CPU_OFFLOAD.",
        ) from exc
    except Exception as exc:
        logger.exception("Garment reconstruction failed")
        raise HTTPException(status_code=500, detail=f"Garment reconstruction failed: {exc}") from exc
