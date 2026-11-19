"""Master Pipeline Orchestrator: Coordinates Flow A (Without Person) and Flow B (With Person)."""
from __future__ import annotations

import logging
import time
from typing import Any
import uuid

import cv2
import numpy as np
from PIL import Image

from app.core.background_removal import (
    extract_isolated_flow_a,
    validate_garment_background,
)
from app.core.flow_b import (
    analyze_garment_and_build_prompts,
    generate_garment_via_gpu,
)
from app.core.garment_extraction import detect_multi_garment_layout
from app.core.image_processing import (
    detect_rotation_angle,
    image_to_base64_data_uri,
    load_image_bytes,
    normalize_to_1024,
    straighten_garment,
)
from app.core.pipeline.pipeline_config import OUTPUT_DIR
from app.core.pipeline.pipeline_router import resolve_execution_flow

logger = logging.getLogger(__name__)


def execute_flow_a_without_person(
    image_rgb: Image.Image,
    image_bytes: bytes,
    job_id: str,
    target_garment: str = "all",
    auto_straighten: bool = True,
    include_base64: bool = False,
    t_start: float | None = None,
) -> dict[str, Any]:
    """Execute Flow A: Pure background removal for isolated garments (without person).

    Designed for flat-lay, ghost mannequin, hanger, or catalog product photos where
    no human is present. Completely bypasses person parsing and human detection overhead.

    Workflow:
    1. Direct background removal via BiRefNet Lite.
    2. Check for multi-garment catalog grid vs single garment layout.
    3. Auto-deskew angle correction if requested.
    4. 1024x1024 transparent canvas normalization and background quality validation.
    """
    if t_start is None:
        t_start = time.perf_counter()

    orig_w, orig_h = image_rgb.size

    # Background Removal with BiRefNet Lite
    t_fg_start = time.perf_counter()
    fg_cutout = extract_isolated_flow_a(image_rgb)
    fg_alpha = np.array(fg_cutout.split()[3])
    fg_time_ms = round((time.perf_counter() - t_fg_start) * 1000, 2)

    # Check for multiple garments on flat-lay canvas (e.g. catalog collection grid)
    is_multi, multi_comps = detect_multi_garment_layout(image_rgb, fg_alpha=fg_alpha)
    processed_items: list[dict[str, Any]] = []

    if is_multi and len(multi_comps) >= 2:
        # Multi-Garment Extraction Flow
        arr_rgb = np.array(image_rgb)
        total_multi_area = max(1, sum(c["area"] for c in multi_comps))

        for idx, comp in enumerate(multi_comps):
            g_id = f"garment_{idx+1:03d}"
            c_mask = comp["comp_mask"]
            # Preserve BiRefNet Lite's exact anti-aliased alpha boundary
            g_alpha = np.where(c_mask, fg_alpha, 0).astype(np.uint8)

            x0, y0, x1, y1 = comp["bbox"]
            pad = 8
            cx0 = max(0, x0 - pad)
            cy0 = max(0, y0 - pad)
            cx1 = min(orig_w, x1 + pad)
            cy1 = min(orig_h, y1 + pad)

            sub_rgb = arr_rgb[cy0:cy1, cx0:cx1]
            sub_a = g_alpha[cy0:cy1, cx0:cx1]
            sub_rgba = Image.fromarray(np.dstack([sub_rgb, sub_a]), mode="RGBA")

            angle_deg = 0.0
            if auto_straighten:
                angle_deg = detect_rotation_angle(sub_rgba)
                if abs(angle_deg) > 0.5:
                    sub_rgba = straighten_garment(sub_rgba, angle_deg)

            normalized_1024 = normalize_to_1024(sub_rgba, target_size=1024, fill_ratio=0.82)
            val_res = validate_garment_background(normalized_1024, original_rgb=image_rgb)
            final_cutout = val_res.get("refined_cutout", normalized_1024)

            out_filename = f"{job_id}_{g_id}.png"
            out_path = OUTPUT_DIR / out_filename
            final_cutout.save(out_path, format="PNG", optimize=True)

            b64_data = image_to_base64_data_uri(final_cutout) if include_base64 else None
            coverage_pct = round((comp["area"] / total_multi_area) * 100.0, 1)

            processed_items.append({
                "garment_id": g_id,
                "category": comp["category"],
                "recognized_label": comp["label_name"],
                "image": out_filename,
                "confidence": val_res["confidence"],
                "coverage_percentage": coverage_pct,
                "priority": "first_priority",
                "is_fused": False,
                "rotation_angle_deg": angle_deg,
                "repaired": False,
                "repair_method": None,
                "width": 1024,
                "height": 1024,
                "image_dimensions": [1024, 1024],
                "output_png_url": f"/api/outputs/{out_filename}",
                "output_png_base64": b64_data,
                "background_removal_status": val_res["status"],
                "background_removal_confidence": val_res["confidence"],
                "garment_preservation_status": val_res["preservation_status"],
                "validation_errors": val_res["validation_errors"],
            })

        scene_type = "multi_garment"
        outfit_mode = "mode_a_independent"
        ensemble_type = "multi_garment_collection"

    else:
        # Single Product Flat-lay / Hanger background removal
        cutout_rgba = fg_cutout

        # Prune detached non-garment specks (such as stand feet, stray hanger artifacts)
        cutout_arr = np.array(cutout_rgba)
        alpha_channel = cutout_arr[:, :, 3]
        bin_cutout = (alpha_channel > 40).astype(np.uint8)
        num_l, lab, st, _ = cv2.connectedComponentsWithStats(bin_cutout)
        if num_l > 2:
            max_comp_area = max(st[i, cv2.CC_STAT_AREA] for i in range(1, num_l))
            # Keep components that are at least 12% of the main garment area
            keep_m = np.zeros_like(bin_cutout, dtype=bool)
            for i in range(1, num_l):
                if st[i, cv2.CC_STAT_AREA] >= 0.12 * max_comp_area:
                    keep_m |= (lab == i)
            cutout_arr[:, :, 3] = np.where(keep_m, alpha_channel, 0)
            cutout_rgba = Image.fromarray(cutout_arr, mode="RGBA")

        angle_deg = 0.0
        if auto_straighten:
            angle_deg = detect_rotation_angle(cutout_rgba)
            if abs(angle_deg) > 0.5:
                cutout_rgba = straighten_garment(cutout_rgba, angle_deg)

        normalized_1024 = normalize_to_1024(cutout_rgba, target_size=1024, fill_ratio=0.82)
        val_res = validate_garment_background(normalized_1024, original_rgb=image_rgb)
        final_cutout = val_res.get("refined_cutout", normalized_1024)

        w, h = cutout_rgba.size
        aspect = h / max(1, w)
        if target_garment and target_garment != "all":
            cat_key = target_garment
            cat_label = target_garment.replace("_", " ").title()
        elif aspect > 1.35:
            cat_key = "dress"
            cat_label = "Dress / Gown / Full Outfit"
        elif aspect > 1.10 and w > h * 0.75:
            cat_key = "dress"
            cat_label = "Dress / Kurta / Outfit"
        elif aspect > 1.15:
            cat_key = "pants"
            cat_label = "Pants / Jeans"
        else:
            cat_key = "upper_clothes"
            cat_label = "Top / Shirt / Garment"

        out_filename = f"{job_id}_{cat_key}.png"
        out_path = OUTPUT_DIR / out_filename
        final_cutout.save(out_path, format="PNG", optimize=True)

        b64_data = image_to_base64_data_uri(final_cutout) if include_base64 else None

        processed_items.append({
            "garment_id": f"{cat_key}_0",
            "category": cat_key,
            "recognized_label": cat_label,
            "image": out_filename,
            "confidence": val_res["confidence"],
            "coverage_percentage": 100.0,
            "priority": "first_priority",
            "is_fused": False,
            "rotation_angle_deg": angle_deg,
            "repaired": False,
            "repair_method": None,
            "width": 1024,
            "height": 1024,
            "image_dimensions": [1024, 1024],
            "output_png_url": f"/api/outputs/{out_filename}",
            "output_png_base64": b64_data,
            "background_removal_status": val_res["status"],
            "background_removal_confidence": val_res["confidence"],
            "garment_preservation_status": val_res["preservation_status"],
            "validation_errors": val_res["validation_errors"],
        })

        scene_type = "flat_lay"
        outfit_mode = "mode_b_single_piece"
        ensemble_type = "single_product"

    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return {
        "success": True,
        "job_id": job_id,
        "execution_time_ms": total_time_ms,
        "total_garments": len(processed_items),
        "garments": processed_items,
        "status": "success",
        "mode": "without_person",
        "has_human": False,
        "scene_type": scene_type,
        "outfit_mode": outfit_mode,
        "execution_time_ms": total_time_ms,
        "total_detected": len(processed_items),
        "total_extracted": len(processed_items),
        "total_background_removed": sum(1 for g in processed_items if g.get("background_removal_status") == "success"),
        "total_partial": sum(1 for g in processed_items if g.get("background_removal_status") == "partial"),
        "total_failed": sum(1 for g in processed_items if g.get("background_removal_status") == "failed"),
        "garments_found": len(processed_items),
        "garments": processed_items,
        "diagnostics": {
            "original_width": orig_w,
            "original_height": orig_h,
            "scene_type": scene_type,
            "outfit_mode": outfit_mode,
            "ensemble_type": ensemble_type,
            "segmentation_time_ms": 0.0,
            "foreground_time_ms": fg_time_ms,
        },
    }


def execute_flow_b_with_person(
    image_rgb: Image.Image,
    image_bytes: bytes,
    job_id: str,
    target_garment: str = "all",
    auto_repair: bool = True,
    auto_straighten: bool = True,
    include_base64: bool = False,
    t_start: float | None = None,
    custom_prompt: str | None = None,
) -> dict[str, Any]:
    """Execute Flow B: Direct ChatGPT Vision + Remote RTX 3090 Qwen-Image-Edit Garment Extraction.

    Clean generative workflow for model-worn photos (zero SegFormer/LaMa bloat):
    1. ChatGPT Vision analyzes the model-worn photo and creates garment-specific prompt(s)
       combined with common studio background-removal rules.
    2. Remote RTX 3090 GPU Worker runs Qwen-Image-Edit using the original photo as visual
       reference + the ChatGPT prompt to generate a clean standalone studio garment.
    3. BiRefNet Lite strips the solid white studio background into a crisp transparent PNG.
    4. Auto-deskew + 1024x1024 transparent canvas normalization.
    """
    if t_start is None:
        t_start = time.perf_counter()

    orig_w, orig_h = image_rgb.size

    # Step 1: ChatGPT Vision garment analysis & prompt generation
    t_prompt_start = time.perf_counter()
    planned_garments = analyze_garment_and_build_prompts(
        image_rgb=image_rgb,
        target_garment=target_garment,
        custom_prompt=custom_prompt,
    )
    prompt_time_ms = round((time.perf_counter() - t_prompt_start) * 1000, 2)

    processed_items: list[dict[str, Any]] = []
    gpu_time_ms = 0.0

    for idx, plan in enumerate(planned_garments):
        cat_key = plan["category"]
        cat_label = plan["recognized_label"]
        qwen_prompt = plan["prompt"]
        g_id = f"garment_{idx + 1:03d}"

        # Step 2: Generate clean studio garment on remote RTX 3090 GPU (Qwen-Image-Edit)
        t_gpu_start = time.perf_counter()
        generated_raw, gpu_meta = generate_garment_via_gpu(
            reference_image=image_rgb,
            prompt=qwen_prompt,
        )
        gpu_time_ms += round((time.perf_counter() - t_gpu_start) * 1000, 2)

        # Step 3: Strip studio white background cleanly via BiRefNet Lite
        cutout_rgba = extract_isolated_flow_a(generated_raw.convert("RGB"))

        # Step 4: 1024x1024 transparent canvas normalization
        # Note: Qwen-Image-2.1-Turbo already generates upright studio garments; skip PCA deskewing
        # so asymmetric dupattas/drapes/flared hems are never erroneously tilted.
        angle_deg = 0.0
        normalized_1024 = normalize_to_1024(cutout_rgba, target_size=1024, fill_ratio=0.82)
        val_res = validate_garment_background(normalized_1024, original_rgb=generated_raw.convert("RGB"))
        final_cutout = val_res.get("refined_cutout", normalized_1024)

        out_filename = f"{job_id}_{g_id}.png"
        out_path = OUTPUT_DIR / out_filename
        final_cutout.save(out_path, format="PNG", optimize=True)

        b64_data = image_to_base64_data_uri(final_cutout) if include_base64 else None

        processed_items.append({
            "garment_id": g_id,
            "category": cat_key,
            "recognized_label": cat_label,
            "image": out_filename,
            "confidence": val_res["confidence"],
            "coverage_percentage": round(100.0 / max(1, len(planned_garments)), 1),
            "priority": "first_priority",
            "is_fused": False,
            "rotation_angle_deg": angle_deg,
            "repaired": True,
            "repair_method": f"chatgpt_vision + {gpu_meta.get('model', 'qwen_gpu')}",
            "width": 1024,
            "height": 1024,
            "image_dimensions": [1024, 1024],
            "output_png_url": f"/api/outputs/{out_filename}",
            "output_png_base64": b64_data,
            "background_removal_status": val_res["status"],
            "background_removal_confidence": val_res["confidence"],
            "garment_preservation_status": val_res["preservation_status"],
            "validation_errors": val_res["validation_errors"],
        })

    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return {
        "success": True,
        "status": "success",
        "message": "Background removed successfully",
        "job_id": job_id,
        "mode": "with_person",
        "has_human": True,
        "scene_type": "person",
        "outfit_mode": "flow_b_chatgpt_qwen_gpu",
        "execution_time_ms": total_time_ms,
        "total_garments": len(processed_items),
        "total_detected": len(processed_items),
        "total_extracted": len(processed_items),
        "total_background_removed": sum(1 for g in processed_items if g.get("background_removal_status") == "success"),
        "total_partial": sum(1 for g in processed_items if g.get("background_removal_status") == "partial"),
        "total_failed": sum(1 for g in processed_items if g.get("background_removal_status") == "failed"),
        "garments_found": len(processed_items),
        "garments": processed_items,
        "diagnostics": {
            "original_width": orig_w,
            "original_height": orig_h,
            "scene_type": "person",
            "prompt_time_ms": prompt_time_ms,
            "gpu_generation_time_ms": round(gpu_time_ms, 2),
        },
    }


def process_garment_pipeline(
    image_bytes: bytes,
    person_mode: str = "with_person",
    has_human: bool | None = None,
    mode_hint: str | None = None,
    target_garment: str = "all",
    auto_repair: bool = True,
    auto_straighten: bool = True,
    include_base64: bool = False,
) -> dict[str, Any]:
    """Execute the Unified 3D Garment Studio Pipeline.

    Single unified production workflow for all images (model photos, worn outfits,
    or flat-lays on bed/floor/hanger):
    1. ChatGPT Vision analyzes the image and creates garment-specific textile prompt(s).
    2. Remote RTX 3090 GPU synthesizes the clean 3D ghost-mannequin product photo.
    3. BiRefNet Lite strips the solid white studio background into a crisp 1024x1024 transparent PNG.

    Args:
        image_bytes: Raw binary image payload.
        person_mode: Optional execution mode (defaults to unified studio flow).
        has_human: Optional explicit boolean flag from caller.
        mode_hint: Optional legacy hint ('worn', 'isolated', 'auto').
        target_garment: Garment filter: 'all', 'upper_clothes', 'pants', 'dress', 'skirt'.
        auto_repair: Enable fabric inpainting for occlusions.
        auto_straighten: Enable deskewing of tilted garments.
        include_base64: Whether to include Base64 data URI in response.

    Returns:
        Structured dictionary matching ProcessGarmentResponse schema.
    """
    t_start = time.perf_counter()
    job_id = uuid.uuid4().hex[:10]

    # Ingest and validate image
    image_rgb = load_image_bytes(image_bytes)

    # Legacy raw-cutout-only fallback (if explicitly requested)
    if person_mode == "flow_a_legacy_raw" or mode_hint == "raw_cutout_only":
        return execute_flow_a_without_person(
            image_rgb=image_rgb,
            image_bytes=image_bytes,
            job_id=job_id,
            target_garment=target_garment,
            auto_straighten=auto_straighten,
            include_base64=include_base64,
            t_start=t_start,
        )

    # Master Unified 3D Studio Flow (ChatGPT Vision + Qwen GPU + BiRefNet Lite 1024x1024)
    return execute_flow_b_with_person(
        image_rgb=image_rgb,
        image_bytes=image_bytes,
        job_id=job_id,
        target_garment=target_garment,
        auto_repair=auto_repair,
        auto_straighten=auto_straighten,
        include_base64=include_base64,
        t_start=t_start,
    )
