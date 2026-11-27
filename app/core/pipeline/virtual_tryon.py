"""End-to-end Virtual Try-On & AI Outfit Stylist pipeline orchestration.

Combines ChatGPT Vision garment/accessory pattern analysis & combination curation
with multi-image Qwen-Image-2.1-Turbo synthesis on the remote GPU worker.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
import time
from typing import Any
import uuid

import numpy as np
from PIL import Image, ImageEnhance, ImageStat

logger = logging.getLogger(__name__)

from app.core.flow_b import (
    generate_garment_via_gpu,
    get_worker_capacity,
    is_gpu_worker_configured,
)
from app.core.flow_b.tryon_stylist import (
    _dynamic_missing_garment,
    analyze_wardrobe_and_plan_tryon,
    build_qwen_tryon_prompt,
)
from app.core.image_processing import image_to_base64_data_uri, load_image_bytes
from app.core.image_processing.face_analyzer import detect_face_and_landmarks
from app.core.pipeline.pipeline_config import OUTPUT_DIR
from app.core.pipeline.tryon_validator import enforce_quality_and_refine

TRYON_OUTPUT_WIDTH = 832
TRYON_OUTPUT_HEIGHT = 1152


def _detect_face_box_cv(avatar_rgb: Image.Image) -> tuple[int, int, int, int] | None:
    """Detect primary face bbox ``(x1, y1, x2, y2)`` via YuNet deep landmark detector."""
    try:
        info = detect_face_and_landmarks(avatar_rgb)
        if info and "bbox" in info:
            x, y, fw, fh = info["bbox"]
            w, h = avatar_rgb.size
            return (
                max(0, int(x - 0.20 * fw)),
                max(0, int(y - 0.25 * fh)),
                min(w, int(x + 1.20 * fw)),
                min(h, int(y + 1.15 * fh)),
            )
    except Exception as exc:
        logger.debug("Deep face detection skipped: %s", exc)
    return None


def prepare_face_identity_ref(
    avatar_rgb: Image.Image,
    face_box_2d: list[int] | None = None,
    canvas_size: int = 576,
) -> Image.Image:
    """Create a borderless, soft-feathered head & upper-neck portrait fading smoothly into studio ivory."""
    w, h = avatar_rgb.size
    if w < 128 or h < 128:
        return avatar_rgb

    crop_coords: tuple[int, int, int, int] | None = None
    if face_box_2d and len(face_box_2d) == 4:
        ymin, xmin, ymax, xmax = face_box_2d
        bw = max(1.0, (xmax - xmin) * w / 1000.0)
        bh = max(1.0, (ymax - ymin) * h / 1000.0)
        x1 = max(0, int(xmin * w / 1000.0 - 0.28 * bw))
        y1 = max(0, int(ymin * h / 1000.0 - 0.24 * bh))
        x2 = min(w, int(xmax * w / 1000.0 + 0.28 * bw))
        y2 = min(h, int(ymax * h / 1000.0 + 0.26 * bh))
        if x2 - x1 >= 48 and y2 - y1 >= 48:
            crop_coords = (x1, y1, x2, y2)

    if crop_coords is None:
        cv_box = _detect_face_box_cv(avatar_rgb)
        if cv_box is not None:
            cx1, cy1, cx2, cy2 = cv_box
            bw = max(1.0, float(cx2 - cx1))
            bh = max(1.0, float(cy2 - cy1))
            crop_coords = (
                max(0, int(cx1 - 0.16 * bw)),
                max(0, int(cy1 - 0.14 * bh)),
                min(w, int(cx2 + 0.16 * bw)),
                min(h, int(cy2 + 0.18 * bh)),
            )

    if crop_coords is None:
        return avatar_rgb

    head_crop = avatar_rgb.crop(crop_coords)
    # Apply only gentle shadow lift on very dark night photos so skin tone/contrast stay natural and faithful
    mean_luma = float(ImageStat.Stat(head_crop.convert("L")).mean[0])
    if 15.0 < mean_luma < 85.0:
        lift = min(1.18, 96.0 / mean_luma)
        head_crop = ImageEnhance.Brightness(head_crop).enhance(lift)
    if min(head_crop.size) < 400:
        head_crop = ImageEnhance.Sharpness(head_crop).enhance(1.18)

    # Scale cleanly to fit canvas with high pixel density
    max_head_dim = int(canvas_size * 0.92)
    head_crop.thumbnail((max_head_dim, max_head_dim), Image.Resampling.LANCZOS)
    rw, rh = head_crop.size
    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)

    # Preserve 100% of face, hair volume, glasses, ears, temples, and chin crystal-clear.
    # Soft vertical fade strictly across the bottom 18% where the avatar's old clothing collar begins.
    y_norm = yy / max(1.0, float(rh))
    bottom_fade = np.clip((0.99 - y_norm) / 0.18, 0.0, 1.0)

    # Subtle perimeter soften (outermost 4% of crop edges so raw rectangular photo borders don't show)
    x_edge = np.minimum(xx, rw - 1 - xx) / max(1.0, rw * 0.04)
    y_edge_top = np.clip(yy / max(1.0, rh * 0.04), 0.0, 1.0)
    border_fade = np.clip(np.minimum(x_edge, y_edge_top), 0.0, 1.0)

    raw_alpha = bottom_fade * border_fade
    alpha = 0.5 - 0.5 * np.cos(np.pi * raw_alpha)

    bg_color = np.array([244.0, 241.0, 236.0], dtype=np.float32)
    fg = np.asarray(head_crop, dtype=np.float32)
    blended = np.clip(fg * alpha[..., None] + bg_color * (1.0 - alpha[..., None]), 0, 255).astype(np.uint8)
    feathered = Image.fromarray(blended, mode="RGB")

    canvas = Image.new("RGB", (canvas_size, canvas_size), (244, 241, 236))
    px = (canvas_size - rw) // 2
    py = (canvas_size - rh) // 2
    canvas.paste(feathered, (px, py))
    return canvas


def _decode_group_entries(
    raw_entries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Decode uploaded image bytes in each garment/accessory group entry to PIL RGB images."""
    decoded: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw_entries):
        img_rgb = load_image_bytes(entry["bytes"])
        decoded.append({
            "index": idx,
            "image": img_rgb,
            "role_hint": entry.get("role_hint", "auto"),
            "filename": entry.get("filename") or f"item_{idx + 1}.jpg",
        })
    return decoded


def process_virtual_tryon_pipeline(
    avatar_bytes: bytes,
    garment_entries: list[dict[str, Any]],
    accessory_entries: list[dict[str, Any]] | None = None,
    max_combinations: int = 2,
    occasion: str = "smart_casual",
    custom_prompt: str | None = None,
    base_url: str = "",
    include_base64: bool = False,
    seed: int | None = None,
    photography_style: str = "auto",
    pose_preference: str = "auto",
) -> dict[str, Any]:
    """Run full virtual try-on pipeline: analyze wardrobe -> plan combinations -> synthesize try-on looks."""
    if not garment_entries:
        raise ValueError("At least one garment image is required for virtual try-on.")

    t_start = time.perf_counter()
    job_id = uuid.uuid4().hex[:12]

    gpu_ok, gpu_msg = is_gpu_worker_configured()
    if not gpu_ok:
        raise RuntimeError(gpu_msg)

    avatar_rgb = load_image_bytes(avatar_bytes)
    garments = _decode_group_entries(garment_entries)
    accessories = _decode_group_entries(accessory_entries or [])

    t_vision = time.perf_counter()
    plan = analyze_wardrobe_and_plan_tryon(
        avatar_img=avatar_rgb,
        garments=garments,
        accessories=accessories,
        max_combinations=max_combinations,
        occasion=occasion,
    )
    vision_ms = round((time.perf_counter() - t_vision) * 1000, 1)

    garment_by_idx = {g["garment_index"]: g for g in plan["garments"]}
    acc_by_idx = {a["accessory_index"]: a for a in plan["accessories"]}
    combos = plan["combinations"]
    identity_avatar_img = prepare_face_identity_ref(avatar_rgb, plan.get("avatar_face_box_2d"))

    def _synthesize_combo(combo_idx: int, combo: dict[str, Any]) -> dict[str, Any]:
        g_indices = combo["selected_garment_indices"]
        a_indices = combo["selected_accessory_indices"]
        selected_g_meta = [garment_by_idx[i] for i in g_indices]
        selected_a_meta = [acc_by_idx[i] for i in a_indices]

        comp_top = str(combo.get("complement_top") or "").strip()
        comp_bottom = str(combo.get("complement_bottom") or "").strip()
        prompt_text, total_refs = build_qwen_tryon_prompt(
            avatar_summary=plan["avatar_summary"],
            selected_garments=selected_g_meta,
            selected_accessories=selected_a_meta,
            occasion=combo.get("occasion") or occasion,
            custom_prompt=custom_prompt,
            original_worn_clothes=plan.get("original_worn_clothes", ""),
            avatar_gender=plan.get("avatar_gender", "unknown"),
            complement_top=comp_top,
            complement_bottom=comp_bottom,
            forensic_face_details=plan.get("forensic_face_details", ""),
            photography_style=photography_style,
            pose_preference=pose_preference,
            combo_index=combo_idx,
        )
        condition_images: list[Image.Image] = (
            [identity_avatar_img]
            + [garments[i]["image"] for i in g_indices]
            + [accessories[i]["image"] for i in a_indices]
        )[:total_refs]

        combo_seed = (seed + combo_idx) if seed is not None else None
        out_img, gpu_meta = generate_garment_via_gpu(
            reference_image=condition_images,
            prompt=prompt_text,
            seed=combo_seed,
            width=TRYON_OUTPUT_WIDTH,
            height=TRYON_OUTPUT_HEIGHT,
        )

        # Automated Quality Validation & Biometric Face Harmonization
        validated_img, val_report = enforce_quality_and_refine(
            generated_img=out_img,
            avatar_img=avatar_rgb,
            selected_garments=selected_g_meta,
            selected_accessories=selected_a_meta,
            avatar_gender=plan.get("avatar_gender", "unknown"),
        )

        # Bounded Retry on Critical Defect if SLA Time Budget Permits (< 45s)
        time_elapsed = time.perf_counter() - t_start
        if not val_report.get("passed", True) and time_elapsed < 45.0:
            retry_seed = (combo_seed or 42) + 101
            retry_prompt = prompt_text
            coverage = val_report.get("clothing_coverage", {})
            if not coverage.get("torso_covered", True):
                retry_prompt += " CRITICAL CORRECTION: MANDATORY FULL TORSO COVERAGE. The model MUST be fully clothed with a complete opaque upper shirt covering chest and stomach, NEVER bare-chested."
            if not coverage.get("legs_covered", True):
                retry_prompt += " CRITICAL CORRECTION: MANDATORY FULL LOWER COVERAGE. The model MUST wear complete opaque trousers or pants covering waist to ankles."
            if val_report.get("face_similarity", 1.0) < 0.35:
                retry_prompt += " CRITICAL CORRECTION: IDENTICAL FACE BIOMETRIC LOCK. Reproduce the exact facial structure, eye shape, nose, and jawline from <image1>."

            logger.info("Quality validation failed (%s); attempting bounded retry with seed %s", val_report.get("defects"), retry_seed)
            try:
                retry_img, retry_meta = generate_garment_via_gpu(
                    reference_image=condition_images,
                    prompt=retry_prompt,
                    seed=retry_seed,
                    width=TRYON_OUTPUT_WIDTH,
                    height=TRYON_OUTPUT_HEIGHT,
                )
                retry_val_img, retry_report = enforce_quality_and_refine(
                    generated_img=retry_img,
                    avatar_img=avatar_rgb,
                    selected_garments=selected_g_meta,
                    selected_accessories=selected_a_meta,
                    avatar_gender=plan.get("avatar_gender", "unknown"),
                )
                if retry_report.get("passed", False) or retry_report.get("face_similarity", 0.0) >= val_report.get("face_similarity", 0.0):
                    validated_img = retry_val_img
                    val_report = retry_report
                    gpu_meta = retry_meta
                    val_report["retried"] = True
            except Exception as retry_exc:
                logger.warning("Bounded retry failed: %s", retry_exc)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_filename = f"{job_id}_look_{combo_idx + 1:03d}.png"
        out_path = OUTPUT_DIR / out_filename
        validated_img.save(out_path, format="PNG")

        clean_base = base_url.rstrip("/")
        rel_url = f"/api/outputs/{out_filename}"
        public_url = f"{clean_base}{rel_url}" if clean_base else rel_url

        res_str = f"{validated_img.width}x{validated_img.height}"
        b64_uri = image_to_base64_data_uri(validated_img) if include_base64 else None

        garment_details = [
            {
                "role": g["role"],
                "index": g["garment_index"],
                "label": g["title"],
                "details": f"{g['colors']}; {g['patterns_and_design']}",
                "body_placement": None,
                "is_complementary": False,
            }
            for g in selected_g_meta
        ]
        if not any(g["role"] in ("top", "outerwear", "full_outfit") for g in selected_g_meta):
            default_t_title, default_t_desc = _dynamic_missing_garment(
                "top", selected_g_meta, plan.get("avatar_gender", "unknown"), combo.get("occasion") or occasion
            )
            garment_details.insert(0, {
                "role": "top",
                "index": -1,
                "label": "Default Modest Complementary Top",
                "details": comp_top or f"{default_t_title}: {default_t_desc}",
                "body_placement": None,
                "is_complementary": True,
            })
        elif not any(g["role"] in ("bottom", "full_outfit") for g in selected_g_meta):
            default_b_title, default_b_desc = _dynamic_missing_garment(
                "bottom", selected_g_meta, plan.get("avatar_gender", "unknown"), combo.get("occasion") or occasion
            )
            garment_details.append({
                "role": "bottom",
                "index": -1,
                "label": default_b_title,
                "details": comp_bottom or default_b_desc,
                "body_placement": None,
                "is_complementary": True,
            })

        return {
            "combination_index": combo_idx + 1,
            "title": combo["title"],
            "occasion": combo.get("occasion") or occasion,
            "style_reason": combo["style_reason"],
            "url": public_url,
            "resolution": res_str,
            "output_png_url": public_url,
            "output_png_base64": b64_uri,
            "selected_garment_indices": g_indices,
            "selected_accessory_indices": a_indices,
            "garment_details": garment_details,
            "accessory_details": [
                {
                    "role": "accessory",
                    "index": a["accessory_index"],
                    "label": a["title"],
                    "details": a["design_details"],
                    "body_placement": a["body_placement"],
                }
                for a in selected_a_meta
            ],
            "avatar_description": plan["avatar_summary"],
            "prompt_used": prompt_text,
            "engine": gpu_meta.get("model", "Qwen/Qwen-Image-2.1-Turbo"),
            "gpu_name": gpu_meta.get("gpu_name", "Remote GPU"),
            "photography_style": photography_style,
            "pose_preference": pose_preference,
            "validation_metrics": val_report,
        }

    workers = max(1, min(len(combos), get_worker_capacity()))
    if len(combos) > 1 and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_synthesize_combo, idx, c) for idx, c in enumerate(combos)]
            generated_looks = [fut.result() for fut in futures]
    else:
        generated_looks = [_synthesize_combo(idx, c) for idx, c in enumerate(combos)]

    total_ms = round((time.perf_counter() - t_start) * 1000, 1)
    sims = [
        look.get("validation_metrics", {}).get("face_similarity", 0.0)
        for look in generated_looks
        if look.get("validation_metrics")
    ]
    avg_sim = round(float(np.mean(sims)), 3) if sims else 0.0

    return {
        "status": "success",
        "success": True,
        "job_id": job_id,
        "total_combinations": len(generated_looks),
        "combinations": generated_looks,
        "garment_catalog": plan["garments"],
        "accessory_catalog": plan["accessories"],
        "processing_time_sec": round(total_ms / 1000.0, 2),
        "execution_time_ms": total_ms,
        "mode": "virtual_tryon_stylist",
        "diagnostics": {
            "prompt_source": plan["prompt_source"],
            "avatar_gender": plan.get("avatar_gender", "unknown"),
            "photography_style": photography_style,
            "pose_preference": pose_preference,
            "excluded_accessories": [a["title"] for a in plan.get("excluded_accessories", [])],
            "excluded_garments": [g["title"] for g in plan.get("excluded_garments", [])],
            "vision_analysis_ms": vision_ms,
            "garments_uploaded": len(garments),
            "accessories_uploaded": len(accessories),
            "combinations_requested": max_combinations,
            "validation_summary": {
                "all_looks_passed": all(
                    look.get("validation_metrics", {}).get("passed", True) for look in generated_looks
                ),
                "grades": [
                    look.get("validation_metrics", {}).get("validation_grade", "A") for look in generated_looks
                ],
                "average_face_similarity": avg_sim,
            },
            "combinations_generated": len(generated_looks),
            "output_resolution": f"{TRYON_OUTPUT_WIDTH}x{TRYON_OUTPUT_HEIGHT}",
        },
    }
