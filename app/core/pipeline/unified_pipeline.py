"""Master Pipeline Orchestrator: Coordinates detection, segmentation, inpainting, and normalization."""
from __future__ import annotations

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


def process_garment_pipeline(
    image_bytes: bytes,
    target_garment: str = "all",
    auto_repair: bool = True,
    auto_straighten: bool = True,
    include_base64: bool = False,
    mode_hint: str = "auto",
) -> dict[str, Any]:
    """Execute the end-to-end Arangtik Garment AI pipeline.

    1. Load image and check orientation.
    2. SegFormer semantic analysis and BiRefNet Lite foreground boundary extraction.
    3. Intelligent Outfit Classification (Mode A: Independent, Mode B: Single-Piece, Mode C: Traditional Saree/Lehenga).
    4. Mannequin vs Human model differentiation.
    5. Mask fusion, boundary snapping, and defect inpainting.
    6. Rotation angle deskew & 1024x1024 transparent canvas normalization.
    """
    t_start = time.perf_counter()
    job_id = uuid.uuid4().hex[:10]

    # Step 1: Ingest and validate image
    image_rgb = load_image_bytes(image_bytes)
    orig_w, orig_h = image_rgb.size

    # Step 2: Background Removal with existing BiRefNet Lite model (Flow A engine)
    t_fg_start = time.perf_counter()
    fg_cutout = extract_isolated_flow_a(image_rgb)
    fg_alpha = np.array(fg_cutout.split()[3])
    fg_time_ms = round((time.perf_counter() - t_fg_start) * 1000, 2)

    # -------------------------------------------------------------
    # FLOW A: Pure Salient Background Removal (Flat-lay / Hanger)
    # When mode_hint == "isolated", skip Person Parsing & Human Analysis completely.
    # -------------------------------------------------------------
    if mode_hint == "isolated":
        # Check for multiple garments layout (e.g. 12 dresses catalog grid)
        is_multi, multi_comps = detect_multi_garment_layout(image_rgb, fg_alpha=fg_alpha)
        processed_items: list[dict[str, Any]] = []

        if is_multi and len(multi_comps) >= 2:
            # Multi-Garment Extraction Flow
            arr_rgb = np.array(image_rgb)
            total_multi_area = max(1, sum(c["area"] for c in multi_comps))

            for idx, comp in enumerate(multi_comps):
                g_id = f"garment_{idx+1:03d}"
                c_mask = comp["comp_mask"]
                c_soft = cv2.GaussianBlur(c_mask.astype(np.float32) * 255.0, (3, 3), 0.5)
                g_alpha = np.clip(c_soft, 0, 255).astype(np.uint8)

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
                    "output_png_url": f"/api/v1/outputs/{out_filename}",
                    "output_png_base64": b64_data,
                    "background_removal_status": val_res["status"],
                    "background_removal_confidence": val_res["confidence"],
                    "garment_preservation_status": val_res["preservation_status"],
                    "validation_errors": val_res["validation_errors"],
                })

            mode = "multi_garment"
            scene_type = "multi_garment"
            outfit_mode = "mode_a_independent"
            ensemble_type = "multi_garment_collection"

        else:
            # Single Product Flat-lay / Hanger background removal (BiRefNet Lite pure flow)
            cutout_rgba = fg_cutout
            angle_deg = 0.0
            if auto_straighten:
                angle_deg = detect_rotation_angle(cutout_rgba)
                if abs(angle_deg) > 0.5:
                    cutout_rgba = straighten_garment(cutout_rgba, angle_deg)

            normalized_1024 = normalize_to_1024(cutout_rgba, target_size=1024, fill_ratio=0.82)
            val_res = validate_garment_background(normalized_1024, original_rgb=image_rgb)
            final_cutout = val_res.get("refined_cutout", normalized_1024)

            w, h = cutout_rgba.size
            if h > w * 1.35:
                cat_key = "dress"
                cat_label = "Dress / Gown / Full Outfit"
            elif h > w * 1.15:
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
                "output_png_url": f"/api/v1/outputs/{out_filename}",
                "output_png_base64": b64_data,
                "background_removal_status": val_res["status"],
                "background_removal_confidence": val_res["confidence"],
                "garment_preservation_status": val_res["preservation_status"],
                "validation_errors": val_res["validation_errors"],
            })

            mode = "isolated_product"
            scene_type = "flat_lay"
            outfit_mode = "mode_b_single_piece"
            ensemble_type = "single_product"

        total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return {
            "success": True,
            "status": "success",
            "job_id": job_id,
            "mode": mode,
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

    # Step 3: Extract Person / Foreground Part
    # Isolate the clean person/garment foreground so background clutter cannot contaminate downstream parsing
    person_rgb = Image.new("RGB", image_rgb.size, (255, 255, 255))
    person_rgb.paste(image_rgb, mask=Image.fromarray(fg_alpha))

    # Step 4: Human Analysis & Person Parsing (Flow B SegFormer) on extracted person foreground
    t_seg_start = time.perf_counter()
    pred_map, probs_tensor = run_segformer_inference(person_rgb, fg_alpha=fg_alpha)
    scene_info = analyze_semantic_map(pred_map)
    seg_time_ms = round((time.perf_counter() - t_seg_start) * 1000, 2)

    has_human = scene_info["has_human"]
    scene_type = scene_info.get("scene_type", "person" if has_human else "flat_lay")
    detected_garments = scene_info["detected_garments"]

    if mode_hint == "worn":
        has_human = True
        scene_type = "person"

    # Step 4: Check for Multiple Independent Garments in flat-lay / catalog images ONLY (never on human models)
    is_multi = False
    multi_comps: list[dict[str, Any]] = []
    if not has_human and scene_type in {"flat_lay", "mannequin"}:
        is_multi, multi_comps = detect_multi_garment_layout(image_rgb, fg_alpha=fg_alpha)

    processed_items: list[dict[str, Any]] = []

    if is_multi and len(multi_comps) >= 2:
        # Multi-Garment Extraction Flow (Flat-lay / Catalog Grid only)
        mode = "multi_garment"
        scene_type = "multi_garment"
        outfit_mode = "mode_a_independent"
        ensemble_type = "multi_garment_collection"
        has_human = False

        arr_rgb = np.array(image_rgb)
        total_multi_area = max(1, sum(c["area"] for c in multi_comps))

        for idx, comp in enumerate(multi_comps):
            g_id = f"garment_{idx+1:03d}"
            c_mask = comp["comp_mask"]
            # Smooth component mask boundary with gentle subpixel anti-aliasing
            c_soft = cv2.GaussianBlur(c_mask.astype(np.float32) * 255.0, (3, 3), 0.5)
            g_alpha = np.clip(c_soft, 0, 255).astype(np.uint8)

            # Crop around component bounding box with safety padding
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

            # Validate background removal quality & check for residual halo
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
                "output_png_url": f"/api/v1/outputs/{out_filename}",
                "output_png_base64": b64_data,
                "background_removal_status": val_res["status"],
                "background_removal_confidence": val_res["confidence"],
                "garment_preservation_status": val_res["preservation_status"],
                "validation_errors": val_res["validation_errors"],
            })

    else:
        # Step 5: Intelligent Outfit Structure Classification
        classification = classify_outfit_structure(
            pred_map=pred_map,
            probs_tensor=probs_tensor,
            detected_garments=detected_garments,
            scene_type=scene_type,
            target_filter=target_garment,
        )
        outfit_mode = classification["outfit_mode"]
        ensemble_type = classification["ensemble_type"]
        planned_items = classification["planned_items"]

        # For isolated product / flat-lay images without human, ensure single product is not split by edge noise
        if not has_human and len(planned_items) > 1 and scene_type in {"flat_lay", "mannequin"}:
            planned_items = [planned_items[0]]

        # Step 6: Execution based on outfit classification
        if planned_items:
            mode = "human_worn" if scene_type == "person" else ("mannequin" if scene_type == "mannequin" else "isolated_product")

            for idx, item in enumerate(planned_items):
                cat_key = item["key"]
                cat_label = item["label"]
                target_cids = item["target_class_ids"]
                is_fused = item["is_fused"]
                coverage_pct = item.get("coverage_percentage", 100.0)
                priority = item.get("priority", "first_priority")

                # 6A. Confidence-Aware Mask Fusion on extracted person foreground with BiRefNet Lite boundary guidance
                cutout_rgba = extract_garment_cutout_guided(
                    image_rgb=person_rgb,
                    probs_tensor=probs_tensor,
                    target_class_ids=target_cids,
                    foreground_cutout=fg_cutout,
                    solid_threshold=0.38 if is_fused else 0.45,
                    edge_threshold=0.15,
                    min_component_area=150,
                )

                # 6B. Occlusion Inspection & LaMa Repair
                # Strictly apply LaMa inpainting only to REAL PERSON photos where arms/hands overlap garment.
                repaired = False
                repair_method = None
                if auto_repair and scene_type == "person":
                    alpha_arr = np.array(cutout_rgba.split()[3])
                    garment_pixel_area = (alpha_arr > 20).sum()
                    notch_mask = compute_arm_occlusion_mask(
                        pred_map, garment_class_id=target_cids, cutout_alpha=alpha_arr
                    )
                    notch_px = (notch_mask > 0).sum()
                    if 60 < notch_px < int(0.35 * garment_pixel_area):
                        cutout_rgba, _ = repair_garment_cutout(cutout_rgba, notch_mask)
                        repaired = True
                        repair_method = "lama_ffc_onnx"

                # 6C. Auto-Straightening
                angle_deg = 0.0
                if auto_straighten:
                    angle_deg = detect_rotation_angle(cutout_rgba)
                    if abs(angle_deg) > 0.5:
                        cutout_rgba = straighten_garment(cutout_rgba, angle_deg)

                # 6D. 1024x1024 Transparent Canvas Normalization
                normalized_1024 = normalize_to_1024(cutout_rgba, target_size=1024, fill_ratio=0.82)

                # 6E. Background quality validation & halo inspection
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
                    "output_png_url": f"/api/v1/outputs/{out_filename}",
                    "output_png_base64": b64_data,
                    "background_removal_status": val_res["status"],
                    "background_removal_confidence": val_res["confidence"],
                    "garment_preservation_status": val_res["preservation_status"],
                    "validation_errors": val_res["validation_errors"],
                })

        else:
            # Fallback: Isolated Product Shot (Flow A BiRefNet Lite)
            mode = "isolated_product"
            cutout_rgba = fg_cutout

            cat_key = "product_garment"
            cat_label = "Garment"
            w, h = cutout_rgba.size
            if h > w * 1.3:
                cat_key = "pants"
                cat_label = "Pants / Trousers"
            else:
                cat_key = "upper_clothes"
                cat_label = "Shirt / Top"

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
                "confidence": 0.98,
                "coverage_percentage": 100.0,
                "priority": "first_priority",
                "is_fused": False,
                "rotation_angle_deg": angle_deg,
                "repaired": False,
                "repair_method": None,
                "width": 1024,
                "height": 1024,
                "image_dimensions": [1024, 1024],
                "output_png_url": f"/api/v1/outputs/{out_filename}",
                "output_png_base64": b64_data,
                "background_removal_status": val_res["status"],
                "background_removal_confidence": val_res["confidence"],
                "garment_preservation_status": val_res["preservation_status"],
                "validation_errors": val_res["validation_errors"],
            })

    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

    total_detected = len(multi_comps) if is_multi else 1
    total_extracted = len(processed_items)
    total_bg_removed = sum(1 for g in processed_items if g.get("background_removal_status") == "success")
    total_partial = sum(1 for g in processed_items if g.get("background_removal_status") == "partial")
    total_failed = sum(1 for g in processed_items if g.get("background_removal_status") == "failed")

    return {
        "success": True,
        "status": "success",
        "job_id": job_id,
        "mode": mode,
        "has_human": has_human,
        "scene_type": scene_type,
        "outfit_mode": outfit_mode,
        "execution_time_ms": total_time_ms,
        "total_detected": total_detected,
        "total_extracted": total_extracted,
        "total_background_removed": total_bg_removed,
        "total_partial": total_partial,
        "total_failed": total_failed,
        "garments_found": total_extracted,
        "garments": processed_items,
        "diagnostics": {
            "original_width": orig_w,
            "original_height": orig_h,
            "scene_type": scene_type,
            "outfit_mode": outfit_mode,
            "ensemble_type": ensemble_type,
            "segmentation_time_ms": seg_time_ms,
            "foreground_time_ms": fg_time_ms,
        },
    }

