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
from app.core.garment_extraction import (
    analyze_semantic_map,
    classify_outfit_structure,
    detect_multi_garment_layout,
    extract_garment_cutout_guided,
)
from app.core.image_processing import (
    detect_rotation_angle,
    image_to_base64_data_uri,
    load_image_bytes,
    normalize_to_1024,
    straighten_garment,
)
from app.core.inpainting import (
    compute_arm_occlusion_mask,
    repair_garment_cutout,
)
from app.core.person_parsing import run_segformer_inference
from app.core.pipeline.pipeline_config import OUTPUT_DIR
from app.core.pipeline.pipeline_router import resolve_execution_flow
from app.experimental.qwen_generation import get_qwen_garment_engine

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
) -> dict[str, Any]:
    """Execute Flow B: Person parsing & fabric inpainting for model-worn garments.

    Workflow:
    1. Isolate person foreground via BiRefNet Lite to strip background clutter.
    2. Execute SegFormer B2 Clothes parsing on isolated person foreground.
    3. Classify outfit structure (independent garments vs traditional ensembles).
    4. Extract confidence-guided garment cutout with boundary snapping.
    5. Inpaint occluded fabric (crossed arms, hands, accessories) using LaMa FFC ONNX.
    6. Auto-deskew angle correction & 1024x1024 transparent canvas normalization.
    """
    if t_start is None:
        t_start = time.perf_counter()

    orig_w, orig_h = image_rgb.size

    # Step 1: Background Removal with BiRefNet Lite to isolate clean person foreground
    t_fg_start = time.perf_counter()
    fg_cutout = extract_isolated_flow_a(image_rgb)
    fg_alpha = np.array(fg_cutout.split()[3])
    fg_time_ms = round((time.perf_counter() - t_fg_start) * 1000, 2)

    # Isolate clean person foreground onto white canvas to avoid background edge noise
    person_rgb = Image.new("RGB", image_rgb.size, (255, 255, 255))
    person_rgb.paste(image_rgb, mask=Image.fromarray(fg_alpha))

    # Step 2: Person Parsing (SegFormer B2 Clothes) on extracted person foreground
    t_seg_start = time.perf_counter()
    pred_map, probs_tensor = run_segformer_inference(person_rgb, fg_alpha=fg_alpha)
    scene_info = analyze_semantic_map(pred_map)
    seg_time_ms = round((time.perf_counter() - t_seg_start) * 1000, 2)

    # Automatic safety fallback: If no human body (face, arms, hair, skin) is present,
    # the image is an isolated product, flat-lay, or mannequin mistakenly sent as Flow B.
    # Route gracefully to Flow A (BiRefNet Lite) rather than hallucinating garments on background noise.
    has_human_detected = scene_info.get("has_human", True)
    if not has_human_detected:
        logger.info(
            "Flow B invoked on image with no human detected (scene_type=%s, human_pixels=%s). "
            "Gracefully delegating to Flow A (isolated background removal).",
            scene_info.get("scene_type"),
            scene_info.get("human_pixel_count", 0),
        )
        return execute_flow_a_without_person(
            image_rgb=image_rgb,
            image_bytes=image_bytes,
            job_id=job_id,
            target_garment=target_garment,
            auto_straighten=auto_straighten,
            include_base64=include_base64,
            t_start=t_start,
        )

    detected_garments = scene_info["detected_garments"]

    # Step 3: Outfit Structure Classification
    classification = classify_outfit_structure(
        pred_map=pred_map,
        probs_tensor=probs_tensor,
        detected_garments=detected_garments,
        scene_type="person",
        target_filter=target_garment,
    )
    outfit_mode = classification["outfit_mode"]
    ensemble_type = classification["ensemble_type"]
    planned_items = classification["planned_items"]

    processed_items: list[dict[str, Any]] = []

    # Step 4: Extract and repair each planned garment item
    if planned_items:
        for idx, item in enumerate(planned_items):
            cat_key = item["key"]
            cat_label = item["label"]
            target_cids = item["target_class_ids"]
            is_fused = item["is_fused"]
            coverage_pct = item.get("coverage_percentage", 100.0)
            priority = item.get("priority", "first_priority")

            # 4A. Confidence-Aware Mask Fusion on person foreground with boundary snapping
            cutout_rgba = extract_garment_cutout_guided(
                image_rgb=person_rgb,
                probs_tensor=probs_tensor,
                target_class_ids=target_cids,
                foreground_cutout=fg_cutout,
                solid_threshold=0.38 if is_fused else 0.45,
                edge_threshold=0.15,
                min_component_area=150,
            )

            # 4B. Occlusion Inspection & Qwen-Image-2.1 Fabric Reconstruction
            repaired = False
            repair_method = None
            if auto_repair:
                alpha_arr = np.array(cutout_rgba.split()[3])
                garment_pixel_area = (alpha_arr > 20).sum()
                notch_mask = compute_arm_occlusion_mask(
                    pred_map, garment_class_id=target_cids, cutout_alpha=alpha_arr
                )
                notch_px = (notch_mask > 0).sum()
                if 60 < notch_px < int(0.35 * garment_pixel_area):
                    qwen_engine = get_qwen_garment_engine()
                    is_avail, _ = qwen_engine.is_available()
                    if is_avail:
                        # Primary: Qwen-Image-2.1 reference-guided reconstruction on GPU
                        q_res = qwen_engine.reconstruct_garment(
                            original_image=person_rgb,
                            garment_cutout=cutout_rgba,
                            repair_mask=notch_mask,
                        )
                        if q_res.status == "success":
                            cutout_rgba = q_res.image
                            repaired = True
                            repair_method = "qwen_image_2_1"
                    else:
                        # Graceful dev environment fallback without locking CPU
                        try:
                            cutout_rgba, _ = repair_garment_cutout(cutout_rgba, notch_mask, original_rgb=person_rgb)
                            repaired = True
                            repair_method = "qwen_image_2_1_pending_gpu (dev_fallback: cpu_safe)"
                        except Exception:
                            repaired = False
                            repair_method = "unrepaired (pending_gpu)"

            # 4C. Auto-Deskewing
            angle_deg = 0.0
            if auto_straighten:
                angle_deg = detect_rotation_angle(cutout_rgba)
                if abs(angle_deg) > 0.5:
                    cutout_rgba = straighten_garment(cutout_rgba, angle_deg)

            # 4D. 1024x1024 Transparent Canvas Normalization
            normalized_1024 = normalize_to_1024(cutout_rgba, target_size=1024, fill_ratio=0.82)

            # 4E. Background Quality Validation & Edge Inspection
            val_res = validate_garment_background(normalized_1024, original_rgb=image_rgb)
            final_cutout = val_res.get("refined_cutout", normalized_1024)

            out_filename = f"{job_id}_{cat_key}.png"
            out_path = OUTPUT_DIR / out_filename
            final_cutout.save(out_path, format="PNG", optimize=True)

            b64_data = image_to_base64_data_uri(final_cutout) if include_base64 else None
            conf_score = round(float(sum(probs_tensor[cid].mean() for cid in target_cids) / len(target_cids)), 3)

            processed_items.append({
                "garment_id": f"{cat_key}_{idx}",
                "category": cat_key,
                "recognized_label": cat_label,
                "image": out_filename,
                "confidence": conf_score,
                "coverage_percentage": coverage_pct,
                "priority": priority,
                "is_fused": is_fused,
                "rotation_angle_deg": angle_deg,
                "repaired": repaired,
                "repair_method": repair_method,
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

    else:
        # Fallback if no specific garment class met solid threshold: use full foreground
        cutout_rgba = fg_cutout
        cat_key = "upper_clothes"
        cat_label = "Garment"
        angle_deg = 0.0
        if auto_straighten:
            angle_deg = detect_rotation_angle(cutout_rgba)
            if abs(angle_deg) > 0.5:
                cutout_rgba = straighten_garment(cutout_rgba, angle_deg)

        normalized_1024 = normalize_to_1024(cutout_rgba, target_size=1024, fill_ratio=0.82)
        val_res = validate_garment_background(normalized_1024, original_rgb=image_rgb)
        final_cutout = val_res.get("refined_cutout", normalized_1024)

        out_filename = f"{job_id}_{cat_key}.png"
        out_path = OUTPUT_DIR / out_filename
        final_cutout.save(out_path, format="PNG", optimize=True)

        b64_data = image_to_base64_data_uri(final_cutout) if include_base64 else None
        processed_items.append({
            "garment_id": f"{cat_key}_0",
            "category": cat_key,
            "recognized_label": cat_label,
            "image": out_filename,
            "confidence": 0.95,
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

    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
    return {
        "success": True,
        "job_id": job_id,
        "execution_time_ms": total_time_ms,
        "total_garments": len(processed_items),
        "garments": processed_items,
        "status": "success",
        "mode": "with_person",
        "has_human": True,
        "scene_type": "person",
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
            "scene_type": "person",
            "outfit_mode": outfit_mode,
            "ensemble_type": ensemble_type,
            "segmentation_time_ms": seg_time_ms,
            "foreground_time_ms": fg_time_ms,
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
    """Execute the Arangtik Garment AI pipeline with explicit flow routing.

    Routes execution based on whether the image contains a person:
    - Flow A (without_person): Direct background removal via BiRefNet Lite with zero
      human-parsing overhead, normalized to clean 1024x1024 transparent canvas.
    - Flow B (with_person): Background removal -> SegFormer B2 clothes parsing ->
      garment extraction -> LaMa inpainting fabric repair -> 1024x1024 normalization.

    Args:
        image_bytes: Raw binary image payload.
        person_mode: Explicit mode: 'with_person' (Flow B) or 'without_person' (Flow A).
        has_human: Optional explicit boolean flag from caller.
        mode_hint: Optional legacy hint ('worn', 'isolated', 'auto').
        target_garment: Garment filter: 'all', 'upper_clothes', 'pants', 'dress', 'skirt'.
        auto_repair: Enable LaMa fabric inpainting for occlusions in Flow B.
        auto_straighten: Enable deskewing of tilted garments.
        include_base64: Whether to include Base64 data URI in response.

    Returns:
        Structured dictionary matching ProcessGarmentResponse schema.
    """
    t_start = time.perf_counter()
    job_id = uuid.uuid4().hex[:10]

    # Ingest and validate image
    image_rgb = load_image_bytes(image_bytes)

    # Resolve execution flow (Flow A without person vs Flow B with person)
    selected_flow = resolve_execution_flow(
        person_mode=person_mode,
        has_human=has_human,
        mode_hint=mode_hint,
    )

    if selected_flow == "without_person":
        return execute_flow_a_without_person(
            image_rgb=image_rgb,
            image_bytes=image_bytes,
            job_id=job_id,
            target_garment=target_garment,
            auto_straighten=auto_straighten,
            include_base64=include_base64,
            t_start=t_start,
        )

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
