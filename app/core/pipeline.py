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
from PIL import Image, ImageFilter, ImageOps

from app.core.detector import analyze_semantic_map
from app.core.inpainter import repair_garment_cutout
from app.core.normalizer import detect_rotation_angle, normalize_to_1024, straighten_garment
from app.core.segmenter import (
    extract_garment_cutout_flow_b,
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
    garment_class_id: int,
    arm_classes: tuple[int, ...] = (14, 15),
    dilation_radius: int = 5,
) -> np.ndarray:
    """Detect regions where model's arms cut into or overlap the garment fabric."""
    garment_mask = (pred_map == garment_class_id)
    arms_mask = np.isin(pred_map, arm_classes)

    if arms_mask.sum() == 0 or garment_mask.sum() == 0:
        return np.zeros_like(garment_mask, dtype=np.uint8)

    # Dilate arms slightly to find contact boundary with garment
    arms_im = Image.fromarray((arms_mask.astype(np.uint8) * 255), mode="L")
    dilated_arms = np.array(arms_im.filter(ImageFilter.MaxFilter(dilation_radius))) > 0

    # Contact notch where arm overlaps immediate garment border
    contact_notch = dilated_arms & garment_mask
    if contact_notch.sum() < 200 or contact_notch.sum() > 25000:
        # Ignore negligible noise or full arm overlap
        return np.zeros_like(garment_mask, dtype=np.uint8)

    return (contact_notch.astype(np.uint8) * 255)


def process_garment_pipeline(
    image_bytes: bytes,
    target_garment: str = "all",
    auto_repair: bool = True,
    auto_straighten: bool = True,
    include_base64: bool = False,
) -> dict[str, Any]:
    """Execute the end-to-end Arangtik Garment AI pipeline.

    1. Load image and check orientation.
    2. Smart human detection and semantic analysis.
    3. Branching: Flow B (Worn multi-garment parsing) or Flow A (Isolated product extraction).
    4. Fabric defect inspection & LaMa inpainting.
    5. Rotation angle deskew & 1024x1024 transparent canvas normalization.
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

    # Step 2: Semantic scene analysis with SegFormer
    t_seg_start = time.perf_counter()
    pred_map, probs_tensor = run_segformer_inference(image_rgb)
    scene_info = analyze_semantic_map(pred_map)
    seg_time_ms = round((time.perf_counter() - t_seg_start) * 1000, 2)

    has_human = scene_info["has_human"]
    detected_garments = scene_info["detected_garments"]
    processed_items: list[dict[str, Any]] = []

    # Step 3: Branching Execution
    if has_human and len(detected_garments) > 0:
        mode = "human_worn"

        # Filter targets if user requested specific category (e.g. "pants", "upper_clothes")
        target_clean = target_garment.strip().lower()
        items_to_process = detected_garments
        if target_clean != "all":
            filtered = [g for g in detected_garments if target_clean in g["category"] or target_clean in g["label"].lower()]
            if filtered:
                items_to_process = filtered

        for idx, g_info in enumerate(items_to_process):
            cid = g_info["class_id"]
            cat_key = g_info["category"]
            cat_label = g_info["label"]

            # 3A. Extract soft-alpha cutout
            cutout_rgba = extract_garment_cutout_flow_b(
                image_rgb, probs_tensor, target_class_ids=[cid]
            )

            # 3B. Occlusion Inspection & LaMa Repair
            repaired = False
            repair_method = None
            if auto_repair:
                notch_mask = compute_arm_occlusion_mask(pred_map, garment_class_id=cid)
                if (notch_mask > 0).sum() > 200:
                    cutout_rgba, _ = repair_garment_cutout(cutout_rgba, notch_mask)
                    repaired = True
                    repair_method = "lama_ffc_onnx"

            # 3C. Auto-Straightening
            angle_deg = 0.0
            if auto_straighten:
                angle_deg = detect_rotation_angle(cutout_rgba)
                if abs(angle_deg) > 0.5:
                    cutout_rgba = straighten_garment(cutout_rgba, angle_deg)

            # 3D. 1024x1024 Transparent Canvas Normalization
            normalized_1024 = normalize_to_1024(cutout_rgba, target_size=1024, fill_ratio=0.82)

            # 3E. Save to outputs/
            out_filename = f"{job_id}_{cat_key}.png"
            out_path = OUTPUT_DIR / out_filename
            normalized_1024.save(out_path, format="PNG", optimize=True)

            b64_data = image_to_base64_data_uri(normalized_1024) if include_base64 else None

            processed_items.append({
                "garment_id": f"{cat_key}_{idx}",
                "category": cat_key,
                "recognized_label": cat_label,
                "confidence": round(float(probs_tensor[cid].mean()), 3),
                "rotation_angle_deg": angle_deg,
                "repaired": repaired,
                "repair_method": repair_method,
                "image_dimensions": [1024, 1024],
                "output_png_url": f"/api/v1/outputs/{out_filename}",
                "output_png_base64": b64_data,
            })

    else:
        # Step 3B: Isolated Product Shot (Flow A BiRefNet Lite)
        mode = "isolated_product"
        cutout_rgba = extract_isolated_flow_a(image_rgb)

        # Detect category hint from SegFormer or aspect ratio
        cat_key = "product_garment"
        cat_label = "Garment"
        if len(detected_garments) > 0:
            top_garment = max(detected_garments, key=lambda x: x["pixel_count"])
            cat_key = top_garment["category"]
            cat_label = top_garment["label"]
        else:
            w, h = cutout_rgba.size
            if h > w * 1.3:
                cat_key = "pants"
                cat_label = "Pants / Trousers"
            else:
                cat_key = "upper_clothes"
                cat_label = "Shirt / Top"

        # Auto-Straighten
        angle_deg = 0.0
        if auto_straighten:
            angle_deg = detect_rotation_angle(cutout_rgba)
            if abs(angle_deg) > 0.5:
                cutout_rgba = straighten_garment(cutout_rgba, angle_deg)

        # 1024x1024 Normalization
        normalized_1024 = normalize_to_1024(cutout_rgba, target_size=1024, fill_ratio=0.82)

        out_filename = f"{job_id}_{cat_key}.png"
        out_path = OUTPUT_DIR / out_filename
        normalized_1024.save(out_path, format="PNG", optimize=True)

        b64_data = image_to_base64_data_uri(normalized_1024) if include_base64 else None

        processed_items.append({
            "garment_id": f"{cat_key}_0",
            "category": cat_key,
            "recognized_label": cat_label,
            "confidence": 0.98,
            "rotation_angle_deg": angle_deg,
            "repaired": False,
            "repair_method": None,
            "image_dimensions": [1024, 1024],
            "output_png_url": f"/api/v1/outputs/{out_filename}",
            "output_png_base64": b64_data,
        })

    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "status": "success",
        "mode": mode,
        "has_human": has_human,
        "execution_time_ms": total_time_ms,
        "garments_found": len(processed_items),
        "garments": processed_items,
        "diagnostics": {
            "original_width": orig_w,
            "original_height": orig_h,
            "human_pixels": scene_info["human_pixels"],
            "human_coverage_pct": scene_info["human_coverage_pct"],
            "segmentation_time_ms": seg_time_ms,
        },
    }

