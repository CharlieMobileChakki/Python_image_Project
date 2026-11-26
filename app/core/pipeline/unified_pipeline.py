"""Master Pipeline Orchestrator: Direct 1024x1024 Studio Generation (Apparel & Jewelry)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import os
import time
from typing import Any
import uuid

import cv2
import numpy as np
from PIL import Image, ImageFilter

from app.core.flow_b import (
    analyze_garment_and_build_prompts,
    analyze_jewelry_and_build_prompts,
    create_multi_angle_composite,
    generate_garment_via_gpu,
    get_worker_capacity,
    is_jewelry_domain,
)
from app.core.image_processing import (
    calibrate_product_colors,
    image_to_base64_data_uri,
    load_image_bytes,
    validate_product_output,
)
from app.core.pipeline.pipeline_config import OUTPUT_DIR

logger = logging.getLogger(__name__)

_JEWELRY_STAND_MAP = {
    "earrings": "beige_velvet_brass_t_bar_stand",
    "jhumkas": "beige_velvet_brass_t_bar_stand",
    "ring": "beige_velvet_brass_ring_block",
    "bangles": "beige_velvet_brass_bangle_bar",
    "kada": "beige_velvet_brass_bangle_bar",
}


def enhance_jewelry_micro_relief(image: Image.Image) -> Image.Image:
    """Enhance micro-relief carvings, leaf veins, and bead borders for jewelry conditioning via CLAHE."""
    try:
        rgb = np.array(image.convert("RGB"))
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        enhanced_lab = cv2.merge((clahe.apply(l_ch), a_ch, b_ch))
        return Image.fromarray(cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB))
    except Exception as exc:
        logger.warning("CLAHE micro-relief enhancement failed (%s); using original image.", exc)
        return image


def _fit_square_white_canvas(crop_img: Image.Image, category: str, target_square: int = 1400) -> Image.Image:
    """Scale image onto a centered square white canvas and apply CLAHE if jewelry."""
    cw, ch = crop_img.size
    scale = (target_square * 0.88) / float(max(cw, ch))
    nw = max(1, int(cw * scale))
    nh = max(1, int(ch * scale))
    resized = crop_img.resize((nw, nh), Image.Resampling.LANCZOS)

    square_canvas = Image.new("RGB", (target_square, target_square), (255, 255, 255))
    square_canvas.paste(resized, ((target_square - nw) // 2, (target_square - nh) // 2))
    return enhance_jewelry_micro_relief(square_canvas) if is_jewelry_domain(category) else square_canvas


def get_optimal_item_reference_crop(
    image: Image.Image,
    category: str,
    box_2d: list[int | float] | tuple[int | float, ...] | None = None,
    total_items: int = 1,
) -> Image.Image:
    """Produce a square-conditioned reference image centered for Qwen visual conditioning."""
    try:
        w, h = image.size
        target_square = 1400

        if category in ("earrings", "jhumkas") and total_items > 1:
            use_heuristic = True
            crop_l, crop_t, crop_r, crop_b = 0, 0, w, h
            if box_2d and isinstance(box_2d, (list, tuple)) and len(box_2d) == 4:
                ymin, xmin, ymax, xmax = [float(v) for v in box_2d]
                is_norm_1 = all(0.0 <= v <= 1.0 for v in (ymin, xmin, ymax, xmax))
                sx = w if is_norm_1 else (w / 1000.0)
                sy = h if is_norm_1 else (h / 1000.0)
                left, top, right, bottom = int(xmin * sx), int(ymin * sy), int(xmax * sx), int(ymax * sy)

                if (bottom - top) > 40 and top < h * 0.6 and bottom < h * 0.75:
                    box_w = right - left
                    box_h = bottom - top
                    if box_w < box_h * 0.65:
                        left = max(0, min(left, w - right))
                        right = min(w, max(right, w - left))
                        box_w = right - left

                    pad_x, pad_y = int(box_w * 0.12), int(box_h * 0.12)
                    crop_l, crop_t = max(0, left - pad_x), max(0, top - pad_y)
                    crop_r, crop_b = min(w, right + pad_x), min(h, bottom + pad_y)
                    use_heuristic = False

            if use_heuristic:
                crop_l, crop_t = max(0, int(w * 0.28)), max(0, int(h * 0.04))
                crop_r, crop_b = min(w, int(w * 0.72)), min(h, int(h * 0.56))

            return _fit_square_white_canvas(image.crop((crop_l, crop_t, crop_r, crop_b)), category, target_square)

        if box_2d and isinstance(box_2d, (list, tuple)) and len(box_2d) == 4:
            ymin, xmin, ymax, xmax = [float(v) for v in box_2d]
            is_norm_1 = all(0.0 <= v <= 1.0 for v in (ymin, xmin, ymax, xmax))
            sx = w if is_norm_1 else (w / 1000.0)
            sy = h if is_norm_1 else (h / 1000.0)
            left, top, right, bottom = int(xmin * sx), int(ymin * sy), int(xmax * sx), int(ymax * sy)
            box_w, box_h = right - left, bottom - top
            if box_w >= 30 and box_h >= 30 and (box_w < w * 0.95 or box_h < h * 0.95):
                pad_x, pad_y = int(box_w * 0.10), int(box_h * 0.10)
                cropped = image.crop((max(0, left - pad_x), max(0, top - pad_y), min(w, right + pad_x), min(h, bottom + pad_y)))
                return _fit_square_white_canvas(cropped, category, target_square)

        if abs(w - h) > 20 or max(w, h) < 1000:
            return _fit_square_white_canvas(image, category, target_square)

        return image
    except Exception as exc:
        logger.warning("Failed to condition item reference (%s); using full image.", exc)
        return image


def _save_jewelry_output(
    item_ref: Image.Image,
    generated_raw: Image.Image,
    out_filename: str,
) -> Image.Image:
    """Calibrate, sharpen, upscale to 2048x2048 RGB, and write PNG for jewelry outputs."""
    calibrated = calibrate_product_colors(item_ref, generated_raw)
    sharpened = calibrated.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.5, percent=135, threshold=1))
    hd_output = sharpened.resize((2048, 2048), Image.Resampling.LANCZOS)
    hd_output.save(OUTPUT_DIR / out_filename, format="PNG", compress_level=1)
    return hd_output


def _process_single_planned_garment(
    idx: int,
    plan: dict[str, Any],
    total_planned: int,
    image_rgb: Image.Image,
    job_id: str,
    include_base64: bool,
    item_count_override: int | None,
) -> tuple[dict[str, Any], float, float]:
    """Generate a single planned apparel or jewelry item directly on remote GPU."""
    cat_key = plan["category"]
    cat_label = plan["recognized_label"]
    qwen_prompt = plan["prompt"]
    g_id = f"garment_{idx + 1:03d}"
    is_jewelry = is_jewelry_domain(cat_key)
    expected_cnt = item_count_override or plan.get("item_count", 1)

    item_ref_image = get_optimal_item_reference_crop(
        image=image_rgb,
        category=cat_key,
        box_2d=plan.get("box_2d"),
        total_items=total_planned,
    )

    gpu_steps = 26 if is_jewelry else 8
    gpu_guidance = 2.5 if is_jewelry else 1.0
    apparel_size = max(512, int(os.getenv("QWEN_APPAREL_SIZE", "1024")))

    t_gpu_start = time.perf_counter()
    generated_raw, gpu_meta = generate_garment_via_gpu(
        reference_image=item_ref_image,
        prompt=qwen_prompt,
        steps=gpu_steps,
        guidance_scale=gpu_guidance,
        width=1024 if is_jewelry else apparel_size,
        height=1024 if is_jewelry else apparel_size,
    )
    gpu_ms = round((time.perf_counter() - t_gpu_start) * 1000, 2)

    t_post_start = time.perf_counter()
    if is_jewelry:
        out_filename = f"{job_id}_{g_id}_jewelry_{cat_key}.png"
        final_output = _save_jewelry_output(item_ref_image, generated_raw, out_filename)
        stand = _JEWELRY_STAND_MAP.get(cat_key, "beige_velvet_brass_neck_bust")
        val_res = validate_product_output(item_ref_image, final_output, expected_count=expected_cnt)
    else:
        direct_rgb = generated_raw.convert("RGB")
        if direct_rgb.size != (1024, 1024):
            direct_rgb = direct_rgb.resize((1024, 1024), Image.Resampling.LANCZOS)
        final_output = direct_rgb
        out_filename = f"{job_id}_{g_id}.png"
        final_output.save(OUTPUT_DIR / out_filename, format="PNG", compress_level=1)
        stand = None
        val_res = validate_product_output(item_ref_image, final_output, expected_count=1)
    postprocess_ms = round((time.perf_counter() - t_post_start) * 1000, 2)

    val_conf = val_res.get("mask_quality", 0.98)
    b64_data = image_to_base64_data_uri(final_output) if include_base64 else None

    item_payload = {
        "garment_id": g_id,
        "category": cat_key,
        "recognized_label": cat_label,
        "title": cat_label,
        "stand_type": stand,
        "presentation_mode": "mode_c_display_stand" if is_jewelry else "ghost_mannequin",
        "detected_item_count": val_res.get("detected_count", expected_cnt),
        "validation_metrics": val_res,
        "image": out_filename,
        "url": f"/api/outputs/{out_filename}",
        "output_png_url": f"/api/outputs/{out_filename}",
        "confidence": val_conf,
        "coverage_percentage": round(100.0 / max(1, total_planned), 1),
        "priority": "first_priority",
        "is_fused": False,
        "repaired": True,
        "repair_method": f"chatgpt_vision + {gpu_meta.get('model', 'qwen_gpu')}",
        "engine": f"{gpu_meta.get('gpu_name', 'NVIDIA GPU')} Turbo (bfloat16)",
        "gpu_name": gpu_meta.get("gpu_name", "NVIDIA GPU"),
        "width": 2048 if is_jewelry else 1024,
        "height": 2048 if is_jewelry else 1024,
        "image_dimensions": [2048, 2048] if is_jewelry else [1024, 1024],
        "output_png_base64": b64_data,
        "garment_preservation_status": "preserved" if val_res.get("verified", True) else "modified",
        "validation_errors": [] if val_res.get("verified", True) else [val_res.get("count_status", "Check count")],
    }
    return item_payload, gpu_ms, postprocess_ms


def execute_studio_generation(
    image_rgb: Image.Image,
    job_id: str,
    target_garment: str = "all",
    include_base64: bool = False,
    t_start: float | None = None,
    custom_prompt: str | None = None,
    item_count_override: int | None = None,
) -> dict[str, Any]:
    """Execute Direct Studio Generation: ChatGPT Vision -> Remote GPU Qwen 2.1 -> 1024x1024 RGB PNG."""
    if t_start is None:
        t_start = time.perf_counter()

    orig_w, orig_h = image_rgb.size

    t_prompt_start = time.perf_counter()
    planned_garments = analyze_garment_and_build_prompts(
        image_rgb=image_rgb,
        target_garment=target_garment,
        custom_prompt=custom_prompt,
    )
    prompt_time_ms = round((time.perf_counter() - t_prompt_start) * 1000, 2)

    total_planned = len(planned_garments)
    worker_capacity = get_worker_capacity()

    if total_planned > 1 and worker_capacity > 1:
        max_parallel = min(total_planned, worker_capacity, 4)
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = [
                pool.submit(
                    _process_single_planned_garment,
                    idx,
                    plan,
                    total_planned,
                    image_rgb,
                    job_id,
                    include_base64,
                    item_count_override,
                )
                for idx, plan in enumerate(planned_garments)
            ]
            item_results = [f.result() for f in futures]
    else:
        item_results = [
            _process_single_planned_garment(
                idx,
                plan,
                total_planned,
                image_rgb,
                job_id,
                include_base64,
                item_count_override,
            )
            for idx, plan in enumerate(planned_garments)
        ]

    processed_items = [r[0] for r in item_results]
    gpu_time_ms = round(sum(r[1] for r in item_results), 2)
    postprocess_time_ms = round(sum(r[2] for r in item_results), 2)
    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "success": True,
        "status": "success",
        "message": "Studio image generated successfully",
        "job_id": job_id,
        "mode": "studio_generation",
        "scene_type": "studio",
        "execution_time_ms": total_time_ms,
        "total_garments": len(processed_items),
        "total_detected": len(processed_items),
        "total_extracted": len(processed_items),
        "garments_found": len(processed_items),
        "garments": processed_items,
        "diagnostics": {
            "original_width": orig_w,
            "original_height": orig_h,
            "prompt_time_ms": prompt_time_ms,
            "gpu_generation_time_ms": gpu_time_ms,
            "postprocess_time_ms": postprocess_time_ms,
        },
    }


def execute_jewelry_studio_pipeline(
    images_rgb: list[Image.Image] | Image.Image,
    job_id: str,
    jewelry_type: str = "auto",
    custom_prompt: str | None = None,
    include_base64: bool = False,
    t_start: float | None = None,
    item_count_override: int | None = None,
) -> dict[str, Any]:
    """Execute 3D Luxury Jewelry Studio Pipeline via direct Qwen 2.1 GPU generation."""
    if t_start is None:
        t_start = time.perf_counter()

    img_list = [images_rgb] if isinstance(images_rgb, Image.Image) else (list(images_rgb) if images_rgb else [])

    t_prompt_start = time.perf_counter()
    jewelry_plan = analyze_jewelry_and_build_prompts(
        images=img_list,
        jewelry_type=jewelry_type,
        custom_prompt=custom_prompt,
    )
    prompt_time_ms = round((time.perf_counter() - t_prompt_start) * 1000, 2)

    cat = jewelry_plan.get("category", "necklace")
    label = jewelry_plan.get("recognized_label", "Fine Gold Jewelry")
    prompt = jewelry_plan.get("prompt", "")
    expected_cnt = item_count_override or jewelry_plan.get("item_count", 1)
    stand = jewelry_plan.get("stand_type", _JEWELRY_STAND_MAP.get(cat, "beige_velvet_brass_neck_bust"))

    composite_ref = enhance_jewelry_micro_relief(create_multi_angle_composite(img_list, target_size=1024))
    t_gpu_start = time.perf_counter()
    generated_raw, gpu_meta = generate_garment_via_gpu(
        reference_image=composite_ref,
        prompt=prompt,
        steps=26,
        guidance_scale=2.5,
        width=1024,
        height=1024,
    )
    gpu_time_ms = round((time.perf_counter() - t_gpu_start) * 1000, 2)

    t_post_start = time.perf_counter()
    out_filename = f"{job_id}_jewelry_{cat}.png"
    hd_output = _save_jewelry_output(composite_ref, generated_raw, out_filename)
    val_metrics = validate_product_output(img_list[0], hd_output, expected_count=expected_cnt)
    postprocess_ms = round((time.perf_counter() - t_post_start) * 1000, 2)

    b64_data = image_to_base64_data_uri(hd_output) if include_base64 else None
    total_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
    total_time_sec = round(total_time_ms / 1000.0, 2)

    item_payload = {
        "garment_id": "jewelry_001",
        "category": cat,
        "recognized_label": label,
        "title": label,
        "image": out_filename,
        "url": f"/api/outputs/{out_filename}",
        "output_png_url": f"/api/outputs/{out_filename}",
        "resolution": "2048x2048",
        "stand_type": stand,
        "presentation_mode": "mode_c_display_stand",
        "detected_item_count": val_metrics.get("detected_count", expected_cnt),
        "validation_metrics": val_metrics,
        "engine": f"{gpu_meta.get('gpu_name', 'NVIDIA GPU')} Turbo (bfloat16)",
        "gpu_name": gpu_meta.get("gpu_name", "NVIDIA GPU"),
        "views_analyzed": len(img_list),
        "output_png_base64": b64_data,
        "confidence": val_metrics.get("mask_quality", 0.98),
        "garment_preservation_status": "preserved" if val_metrics.get("verified", True) else "modified",
        "validation_errors": [] if val_metrics.get("verified", True) else [val_metrics.get("count_status", "Check count")],
    }

    return {
        "status": "success",
        "success": True,
        "message": "Jewelry presentation generated successfully",
        "job_id": job_id,
        "mode": "jewelry_studio",
        "scene_type": "jewelry",
        "total_garments": 1,
        "garments": [item_payload],
        "processing_time_sec": total_time_sec,
        "execution_time_ms": total_time_ms,
        "total_detected": 1,
        "total_extracted": 1,
        "garments_found": 1,
        "diagnostics": {
            "category": cat,
            "stand_type": stand,
            "detected_item_count": val_metrics.get("detected_count", expected_cnt),
            "expected_count": expected_cnt,
            "views_analyzed": len(img_list),
            "prompt_time_ms": prompt_time_ms,
            "gpu_generation_time_ms": gpu_time_ms,
            "postprocess_time_ms": postprocess_ms,
            "validation": val_metrics,
        },
    }


def process_garment_pipeline(
    image_bytes: bytes | list[bytes],
    target_garment: str = "all",
    include_base64: bool = False,
    custom_prompt: str | None = None,
    item_count_override: int | None = None,
    force_jewelry: bool = False,
) -> dict[str, Any]:
    """Execute the Direct 1024x1024 Studio Generation Pipeline (Apparel & Jewelry)."""
    t_start = time.perf_counter()
    job_id = uuid.uuid4().hex[:10]

    if isinstance(image_bytes, list):
        images_rgb = [load_image_bytes(b) for b in image_bytes if b]
    else:
        images_rgb = [load_image_bytes(image_bytes)]

    if not images_rgb:
        raise ValueError("No valid image data provided to pipeline.")

    if len(images_rgb) > 1 or force_jewelry or is_jewelry_domain(target_garment):
        return execute_jewelry_studio_pipeline(
            images_rgb=images_rgb,
            job_id=job_id,
            jewelry_type=target_garment if is_jewelry_domain(target_garment) else "auto",
            custom_prompt=custom_prompt,
            include_base64=include_base64,
            t_start=t_start,
            item_count_override=item_count_override,
        )

    return execute_studio_generation(
        image_rgb=images_rgb[0],
        job_id=job_id,
        target_garment=target_garment,
        include_base64=include_base64,
        t_start=t_start,
        custom_prompt=custom_prompt,
        item_count_override=item_count_override,
    )
