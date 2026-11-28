"""End-to-end AI Stylized Avatar Profile Picture Pipeline.

Orchestrates:
1. Intelligent bust-up portrait cropping via YuNet neural face detection.
2. Forensic facial and outfit analysis via ChatGPT Vision (with deterministic fallback).
3. 2D vector art prompt synthesis with pastel circular halo badge and minimalist doodles.
4. Conditioned image generation via remote Qwen-Image-2.1-Turbo GPU worker.
5. High-definition 1024x1024 PNG persistence and response serialization.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any
import uuid

from PIL import Image

from app.core.avatar.avatar_cropper import crop_bust_up_portrait
from app.core.avatar.avatar_stylist import (
    analyze_avatar_identity,
    build_avatar_stylized_prompt,
)
from app.core.flow_b import generate_garment_via_gpu, is_gpu_worker_configured
from app.core.image_processing import image_to_base64_data_uri, load_image_bytes
from app.core.pipeline.pipeline_config import OUTPUT_DIR
from app.schemas.avatar_schemas import AvatarIdentityFeatures

logger = logging.getLogger(__name__)


def process_avatar_pipeline(
    image_bytes: bytes,
    style: str = "vector_art",
    halo_color: str = "auto",
    doodle_accent: str = "auto",
    quality: str = "standard",
    include_base64: bool = False,
    custom_prompt: str | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Execute the complete stylized avatar generation pipeline for a user photo."""
    t_start = time.perf_counter()
    job_id = job_id or f"avatar_{uuid.uuid4().hex[:12]}"

    logger.info(
        "Starting avatar pipeline | job_id=%s style=%s halo=%s doodle=%s bytes=%d",
        job_id,
        style,
        halo_color,
        doodle_accent,
        len(image_bytes),
    )

    # 1. Decode user photo into RGB PIL image
    raw_img = load_image_bytes(image_bytes).convert("RGB")

    # 2. Extract visual identity details via ChatGPT Vision
    avatar_info = analyze_avatar_identity(raw_img)

    # 3. Crop centered bust-up portrait reference using YuNet / Vision coordinates
    face_box_2d = avatar_info.get("face_box_2d")
    cropped_ref = crop_bust_up_portrait(raw_img, face_box_2d=face_box_2d, target_size=1024)

    # 4. Synthesize stylized prompt with halo badge and doodle accents
    prompt, resolved_halo, resolved_doodle = build_avatar_stylized_prompt(
        avatar_info=avatar_info,
        style=style,
        halo_color=halo_color,
        doodle_accent=doodle_accent,
        custom_prompt=custom_prompt,
    )

    logger.info(
        "Generated avatar prompt | job_id=%s halo=%s doodle=%s prompt_len=%d",
        job_id,
        resolved_halo,
        resolved_doodle,
        len(prompt),
    )

    # 5. Call GPU worker (Qwen-Image-2.1-Turbo)
    # 1024x1024 resolution is the gold standard for avatar profile pictures (PFP)
    steps = 16 if quality in ("high", "ultra") else 14
    guidance = 2.0

    try:
        generated_img, diagnostics = generate_garment_via_gpu(
            reference_image=cropped_ref,
            prompt=prompt,
            steps=steps,
            guidance_scale=guidance,
            width=1024,
            height=1024,
        )
    except Exception as exc:
        logger.error("GPU worker synthesis failed for avatar | job_id=%s error=%s", job_id, exc)
        raise RuntimeError(f"Avatar synthesis failed on GPU worker: {exc}") from exc

    # 6. Save generated 1024x1024 PNG to outputs directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_filename = f"{job_id}_avatar.png"
    out_path = OUTPUT_DIR / out_filename
    generated_img.save(out_path, format="PNG", optimize=True)

    elapsed_sec = round(time.perf_counter() - t_start, 2)
    elapsed_ms = round(elapsed_sec * 1000.0, 1)

    logger.info(
        "Completed avatar generation | job_id=%s file=%s time=%.2fs",
        job_id,
        out_filename,
        elapsed_sec,
    )

    base64_uri = image_to_base64_data_uri(generated_img) if include_base64 else None

    identity_obj = AvatarIdentityFeatures(
        gender=avatar_info.get("gender", "unspecified"),
        expression=avatar_info.get("expression", "warm smile"),
        hair_details=avatar_info.get("hair_details", ""),
        eyewear=avatar_info.get("eyewear"),
        facial_hair=avatar_info.get("facial_hair"),
        upper_garment=avatar_info.get("upper_garment", ""),
        distinctive_features=avatar_info.get("distinctive_features"),
    )

    return {
        "status": "success",
        "success": True,
        "job_id": job_id,
        "avatar_url": f"/outputs/{out_filename}",
        "output_png_url": f"/outputs/{out_filename}",
        "avatar_base64": base64_uri,
        "style": style,
        "halo_color": resolved_halo,
        "doodle_accent": resolved_doodle,
        "identity_features": identity_obj.model_dump(),
        "resolution": "1024x1024",
        "processing_time_sec": elapsed_sec,
        "execution_time_ms": elapsed_ms,
        "prompt_used": prompt,
        "engine": diagnostics.get("engine", "Qwen/Qwen-Image-2.1-Turbo"),
        "gpu_name": diagnostics.get("gpu_name"),
        "diagnostics": diagnostics,
    }
