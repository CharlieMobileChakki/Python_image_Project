"""Master Pipeline Orchestrator: Coordinates detection, segmentation, inpainting, and normalization."""
from __future__ import annotations

import base64
from io import BytesIO
import os
from pathlib import Path
import time
from typing import Any
import uuid

import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps

from app.core.detector import analyze_semantic_map, classify_outfit_structure, detect_multi_garment_layout
from app.core.inpainter import repair_garment_cutout
from app.core.normalizer import detect_rotation_angle, normalize_to_1024, straighten_garment
from app.core.validator import validate_garment_background
from app.core.segmenter import (
    extract_garment_cutout_flow_b,
    extract_garment_cutout_guided,
    extract_isolated_flow_a,
    run_segformer_inference,
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def image_to_base64_data_uri(img_rgba: Image.Image) -> str:
    """Convert PIL RGBA image to Base64 PNG data URI."""
    buf = BytesIO()
    img_rgba.save(buf, format="PNG", optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


def compute_arm_occlusion_mask(
    pred_map: np.ndarray,
    garment_class_id: int | list[int] | tuple[int, ...],
    cutout_alpha: np.ndarray | None = None,
    arm_classes: tuple[int, ...] = (14, 15),
    dilation_radius: int = 17,
) -> np.ndarray:
    """Detect regions where model's arms cut into or cross over garment fabric.

    Strictly triggers ONLY for genuine human occlusions:
    - Reconstructs fabric hidden by hands/arms crossing over the torso/garment body.
    - Excludes arms resting outside on walls or hanging down in the background.
    - Excludes background regions (Class 0), natural necklines, and bottom hemlines.
    """
    if isinstance(garment_class_id, (list, tuple)):
        garment_mask = np.isin(pred_map, garment_class_id)
    else:
        garment_mask = (pred_map == garment_class_id)

    arms_mask = np.isin(pred_map, arm_classes)
    bg_mask = (pred_map == 0)

    if garment_mask.sum() == 0 or (arms_mask.sum() < 100 and cutout_alpha is None):
        return np.zeros_like(garment_mask, dtype=np.uint8)

    H, W = pred_map.shape
    garment_rows = np.any(garment_mask, axis=1)
    if not np.any(garment_rows):
        return np.zeros_like(garment_mask, dtype=np.uint8)

    y_indices = np.where(garment_rows)[0]
    y_min, y_max = y_indices[0], y_indices[-1]
    garment_height = max(1, y_max - y_min)

    # Protect natural neckline (top 10%) and natural hemline (bottom 5%)
    y_top_limit = y_min + int(0.10 * garment_height)
    y_bot_limit = y_max - int(0.05 * garment_height)

    # 1. Arm regions genuinely crossing the garment torso/body
    crossed_arms = np.zeros_like(garment_mask, dtype=bool)
    if arms_mask.sum() > 0:
        for y in range(y_top_limit, y_bot_limit):
            row_g = np.where(garment_mask[y])[0]
            row_a = np.where(arms_mask[y])[0]
            if len(row_g) < 25 or len(row_a) == 0:
                continue
            g_left, g_right = row_g[0], row_g[-1]
            # Arm pixel must be embedded horizontally inside the garment silhouette
            in_torso = (row_a > (g_left + 15)) & (row_a < (g_right - 15))
            valid_x = row_a[in_torso]
            if len(valid_x) > 0:
                crossed_arms[y, valid_x] = True

        # Filter out arm components that touch external background over majority of perimeter
        if crossed_arms.any():
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(crossed_arms.astype(np.uint8))
            kernel = np.ones((7, 7), np.uint8)
            filtered_arms = np.zeros_like(crossed_arms, dtype=bool)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < 40:
                    continue
                comp = (labels == i)
                dilated = cv2.dilate(comp.astype(np.uint8), kernel) > 0
                ring = dilated & ~comp
                bg_touch = (ring & bg_mask).sum()
                total_ring = max(1, ring.sum())
                # If arm component touches external background over >= 25%, it is an external limb
                if (bg_touch / total_ring) < 0.25:
                    filtered_arms |= comp
            crossed_arms = filtered_arms

    # If there are no confirmed human arm/hand pixels crossing the garment, do not repair!
    # Prevents artificial holes being carved into plackets, necklines, or embroidery on flat-lays or mannequins.
    if not crossed_arms.any():
        return np.zeros_like(garment_mask, dtype=np.uint8)

    # 2. Internal transparent holes enclosed within the garment silhouette that touch crossed arms
    internal_holes = np.zeros_like(garment_mask, dtype=bool)
    if cutout_alpha is not None:
        contours, _ = cv2.findContours(
            (cutout_alpha > 20).astype(np.uint8) * 255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        filled_silhouette = np.zeros_like(cutout_alpha)
        for c in contours:
            cv2.drawContours(filled_silhouette, [c], -1, 255, -1)
        # Internal void: strictly enclosed, NOT background, NOT in neckline or hemline
        candidate_holes = (filled_silhouette == 255) & (cutout_alpha < 30) & (~bg_mask)
        candidate_holes[:y_top_limit, :] = False
        candidate_holes[y_bot_limit:, :] = False
        # Must touch or be near crossed arm pixels
        dilated_arms = cv2.dilate(crossed_arms.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        internal_holes = candidate_holes & dilated_arms

    # Union of genuine crossed arms and enclosed arm holes
    repair_mask = crossed_arms | internal_holes
    if repair_mask.sum() < 60:
        return np.zeros_like(garment_mask, dtype=np.uint8)

    # Dilate slightly so LaMa seamlessly blends the boundary into surrounding intact fabric
    mask_im = Image.fromarray((repair_mask.astype(np.uint8) * 255), mode="L")
    dilated_mask = np.array(mask_im.filter(ImageFilter.MaxFilter(dilation_radius)))
    return dilated_mask


def process_garment_pipeline(
    image_bytes: bytes,
    target_garment: str = "all",
    auto_repair: bool = True,
    auto_straighten: bool = True,
    include_base64: bool = False,
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
    try:
        raw_image = Image.open(BytesIO(image_bytes))
        image_rgb = ImageOps.exif_transpose(raw_image).convert("RGB")
    except Exception as exc:
        raise ValueError("Invalid or corrupted image format. Please upload a valid JPG, PNG, or WEBP.") from exc

    orig_w, orig_h = image_rgb.size

    # Step 2: Extract high-resolution foreground boundary with BiRefNet Lite
    t_fg_start = time.perf_counter()
    fg_cutout = extract_isolated_flow_a(image_rgb)
    fg_alpha = np.array(fg_cutout.split()[3])
    fg_time_ms = round((time.perf_counter() - t_fg_start) * 1000, 2)

    # Step 3: Check for Multiple Independent Garments in a single product/catalog image
    is_multi, multi_comps = detect_multi_garment_layout(image_rgb, fg_alpha=fg_alpha)

    processed_items: list[dict[str, Any]] = []

    if is_multi and len(multi_comps) >= 2:
        # Multi-Garment Extraction Flow
        mode = "multi_garment"
        scene_type = "multi_garment"
        outfit_mode = "mode_a_independent"
        ensemble_type = "multi_garment_collection"
        has_human = False
        seg_time_ms = 0.0

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
        # Step 4: Semantic scene analysis with SegFormer (for single garment or worn outfit)
        t_seg_start = time.perf_counter()
        pred_map, probs_tensor = run_segformer_inference(image_rgb)
        scene_info = analyze_semantic_map(pred_map)
        seg_time_ms = round((time.perf_counter() - t_seg_start) * 1000, 2)

        has_human = scene_info["has_human"]
        scene_type = scene_info.get("scene_type", "person" if has_human else "flat_lay")
        detected_garments = scene_info["detected_garments"]

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

                # 6A. Confidence-Aware Mask Fusion with BiRefNet Lite boundary guidance
                cutout_rgba = extract_garment_cutout_guided(
                    image_rgb=image_rgb,
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

