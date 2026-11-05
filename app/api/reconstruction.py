"""FastAPI router for Qwen-Image-2.1 Primary Garment Reconstruction."""
from __future__ import annotations

import io
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
import numpy as np
from PIL import Image, ImageOps

from app.core.background_removal import extract_isolated_flow_a
from app.core.garment_extraction import (
    analyze_semantic_map,
    extract_garment_cutout_guided,
)
from app.core.inpainting import compute_arm_occlusion_mask
from app.core.person_parsing import run_segformer_inference
from app.experimental.qwen_generation import get_qwen_garment_engine

logger = logging.getLogger(__name__)
reconstruction_router = APIRouter(tags=["Garment Reconstruction"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


def load_pil_image(data: bytes, filename: str) -> Image.Image:
    """Decode raw bytes into sanitized, EXIF-transposed PIL Image."""
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGBA" if "A" in img.mode else "RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode image {filename}: {exc}") from exc


async def _read_upload(upload: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Safely read upload stream enforcing size constraints."""
    raw = await upload.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise HTTPException(status_code=413, detail=f"{upload.filename} exceeds upload limit.")
    if not raw:
        raise HTTPException(status_code=400, detail=f"{upload.filename} is empty.")
    return raw


@reconstruction_router.post(
    "/qwen-image",
    summary="Fabric Reconstruction & Generative Synthesis using Qwen-Image-2.1",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
async def qwen_image_endpoint(
    original_image: UploadFile = File(..., description="Original person photo used as garment visual reference."),
    damaged_garment: Optional[UploadFile] = File(None, description="Person-removed garment PNG. Auto-generated if omitted."),
    repair_mask: Optional[UploadFile] = File(None, description="Black/white occlusion mask. Auto-generated if omitted."),
    garment_reference: Optional[UploadFile] = File(None, description="Optional tight crop of visible garment from original photo."),
    prompt: Optional[str] = Form(None, description="Optional prompt describing specific garment preservation requirements."),
    steps: Optional[int] = Form(None, description="Number of denoising inference steps (default: 28)."),
    guidance_scale: Optional[float] = Form(None, description="Classifier-free guidance scale."),
    strength: Optional[float] = Form(None, description="Reconstruction strength."),
    ip_adapter_scale: Optional[float] = Form(None, description="Legacy parameter retained for backward compatibility."),
    seed: Optional[int] = Form(None, description="Random seed for reproducible generation."),
):
    """Primary deep generative reconstruction endpoint using Qwen-Image-2.1.

    Workflow:
    1. Ingest original photograph and visible garment reference.
    2. Auto-extract garment cutout and occlusion mask via SegFormer B2 if not supplied.
    3. Condition Qwen-Image-2.1 visual diffusion on visible garment fabric, weave, embroidery, and borders.
    4. Run mandatory final background verification with BiRefNet Lite.
    5. Composite newly synthesized fabric strictly into repair region, preserving 100% of original pixels outside.
    6. Return clean transparent RGBA PNG.
    """
    engine = get_qwen_garment_engine()
    is_avail, reason = engine.is_available()
    if not is_avail:
        raise HTTPException(
            status_code=503,
            detail=f"Qwen-Image-2.1 Garment Reconstruction is unavailable in this environment ({reason}). Ready for deployment on RTX 3090 GPU server.",
        )

    try:
        engine.load()
    except Exception as exc:
        logger.exception("Failed to load Qwen-Image-2.1 Reconstruction engine: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Failed to initialize Qwen-Image-2.1 reconstruction model: {exc}",
        ) from exc

    try:
        original_raw = await _read_upload(original_image)
        original_pil = load_pil_image(original_raw, original_image.filename or "original_image")

        damaged_pil = None
        if damaged_garment is not None:
            damaged_raw = await _read_upload(damaged_garment)
            damaged_pil = load_pil_image(damaged_raw, damaged_garment.filename or "damaged_garment")

        mask_pil = None
        if repair_mask is not None:
            mask_raw = await _read_upload(repair_mask)
            mask_pil = load_pil_image(mask_raw, repair_mask.filename or "repair_mask")

        reference_raw = await _read_upload(garment_reference) if garment_reference else None
        reference_pil = load_pil_image(reference_raw, garment_reference.filename or "garment_reference") if reference_raw else None

        # Auto-extract damaged cutout and repair mask from original image if omitted
        if damaged_pil is None or mask_pil is None:
            fg_cutout = extract_isolated_flow_a(original_pil)
            fg_alpha = np.array(fg_cutout.split()[3])
            person_rgb = Image.new("RGB", original_pil.size, (255, 255, 255))
            person_rgb.paste(original_pil, mask=Image.fromarray(fg_alpha))

            pred_map, probs_tensor = run_segformer_inference(person_rgb, fg_alpha=fg_alpha)
            scene_info = analyze_semantic_map(pred_map)
            target_cids = []
            for item in scene_info.get("detected_garments", []):
                target_cids.extend(item.get("class_ids", []))
            if not target_cids:
                target_cids = [4, 5, 6, 7]

            auto_cutout = extract_garment_cutout_guided(
                image_rgb=person_rgb,
                probs_tensor=probs_tensor,
                target_class_ids=target_cids,
                foreground_cutout=fg_cutout,
            )
            alpha_arr = np.array(auto_cutout.split()[3])
            notch_mask = compute_arm_occlusion_mask(pred_map, garment_class_id=target_cids, cutout_alpha=alpha_arr)

            if damaged_pil is None:
                damaged_pil = auto_cutout
            if mask_pil is None:
                mask_pil = Image.fromarray(notch_mask)

        # Execute reference-conditioned reconstruction
        result = engine.reconstruct_garment(
            original_image=original_pil,
            garment_cutout=damaged_pil,
            repair_mask=mask_pil,
            garment_reference=reference_pil,
            prompt=prompt,
            inference_steps=steps or 28,
            seed=seed,
        )

        buffer = io.BytesIO()
        result.image.save(buffer, format="PNG", optimize=True)
        png_bytes = buffer.getvalue()

        headers = {
            "X-Model-Name": result.model_name,
            "X-Inference-Seconds": str(result.inference_time_seconds),
            "X-Model-Device": engine.configured_device,
            "X-Seed": str(result.seed or seed or 0),
            "X-Output-Width": str(result.width),
            "X-Output-Height": str(result.height),
            "X-Output-Format": "PNG-RGBA",
            "X-Status": result.status,
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
        if damaged_garment:
            await damaged_garment.close()
        if repair_mask:
            await repair_mask.close()
        if garment_reference:
            await garment_reference.close()
