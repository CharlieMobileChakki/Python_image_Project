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
from PIL import Image, ImageEnhance, ImageFilter, ImageStat

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
from app.core.pipeline.pipeline_config import OUTPUT_DIR
import os

TRYON_OUTPUT_WIDTH = int(os.getenv("TRYON_OUTPUT_WIDTH", "832"))
TRYON_OUTPUT_HEIGHT = int(os.getenv("TRYON_OUTPUT_HEIGHT", "1152"))
TRYON_DEFAULT_STEPS = int(os.getenv("TRYON_INFERENCE_STEPS", "14"))
TRYON_DEFAULT_GUIDANCE = float(os.getenv("TRYON_GUIDANCE_SCALE", "1.4"))

QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "standard": {"width": 832, "height": 1152, "steps": 12, "guidance_scale": 1.3},
    "high": {"width": 896, "height": 1280, "steps": 14, "guidance_scale": 1.5},
    "ultra": {"width": 1024, "height": 1344, "steps": 18, "guidance_scale": 1.8},
}


def clean_studio_backdrop(img: Image.Image) -> Image.Image:
    """Eliminate vertical wall seams, vertical line artifacts, and dirty gradient streaks from seamless studio cyclorama."""
    try:
        arr = np.array(img, dtype=np.float32)
        h, w, _ = arr.shape
        if h < 200 or w < 200:
            return img

        # Sample studio background tone from top corners (strictly away from head/subject)
        corner_w = max(20, int(w * 0.15))
        corner_h = max(20, int(h * 0.10))
        top_left = arr[:corner_h, :corner_w]
        top_right = arr[:corner_h, -corner_w:]
        bg_target = (top_left.mean(axis=(0, 1)) + top_right.mean(axis=(0, 1))) / 2.0

        # Distance in RGB space from true background
        color_dist = np.sqrt(np.sum((arr - bg_target) ** 2, axis=-1))
        luma = arr.mean(axis=-1)

        # Grounding shadow and subject protection:
        # Only target bright background pixels (luma > 208, color_dist < 34)
        bg_affinity = np.clip((34.0 - color_dist) / 16.0, 0.0, 1.0) * np.clip((luma - 208.0) / 25.0, 0.0, 1.0)

        # Ground contact shadow protection: lower 18% floor area where feet contact the ground
        y_coords = np.arange(h).astype(np.float32) / max(1.0, float(h))
        floor_mask = np.clip((y_coords - 0.88) / 0.10, 0.0, 1.0)[..., None]
        bg_affinity = bg_affinity * (1.0 - floor_mask * np.clip((230.0 - luma) / 30.0, 0.0, 1.0))

        if bg_affinity.max() > 0.05:
            # Softly blend background towards clean target to eliminate vertical seams
            homogenized = bg_target.reshape(1, 1, 3)
            weight = (bg_affinity[..., None] * 0.82)
            cleaned = arr * (1.0 - weight) + homogenized * weight
            return Image.fromarray(np.clip(cleaned, 0, 255).astype(np.uint8))
        return img
    except Exception as exc:
        logger.debug("Studio backdrop cleaning skipped: %s", exc)
        return img


def refine_facial_micro_clarity(img: Image.Image) -> Image.Image:
    """Refine facial micro-clarity: removes compression pixelation/tearing while boosting iris, lash, and hair crispness."""
    try:
        w, h = img.size
        if w < 256 or h < 256:
            return img

        # Target head region: centered horizontally, top 1% to 24% of vertical height
        hx1 = max(0, int(w * 0.32))
        hy1 = max(0, int(h * 0.01))
        hx2 = min(w, int(w * 0.68))
        hy2 = min(h, int(h * 0.24))

        hw = hx2 - hx1
        hh = hy2 - hy1
        if hw < 32 or hh < 32:
            return img

        head_crop = img.crop((hx1, hy1, hx2, hy2))
        head_np = np.array(head_crop)

        # Edge-preserving bilateral filter to eliminate pixelation and noise tearing on flat skin
        try:
            import cv2
            smooth = cv2.bilateralFilter(head_np, 5, 26, 26)
            detail = head_np.astype(np.float32) - smooth.astype(np.float32)
            refined = np.clip(smooth.astype(np.float32) + detail * 1.30, 0, 255)
        except Exception:
            smooth_pil = head_crop.filter(ImageFilter.GaussianBlur(radius=1.2))
            smooth = np.array(smooth_pil, dtype=np.float32)
            detail = head_np.astype(np.float32) - smooth
            refined = np.clip(smooth + detail * 1.30, 0, 255)

        # 2D Cosine window feathering for 100% seamless, invisible integration
        yy, xx = np.mgrid[0:hh, 0:hw].astype(np.float32)
        margin_x = max(1.0, float(hw) * 0.12)
        margin_y = max(1.0, float(hh) * 0.12)
        fade_x = np.clip(np.minimum(xx, hw - 1 - xx) / margin_x, 0.0, 1.0)
        fade_y = np.clip(np.minimum(yy, hh - 1 - yy) / margin_y, 0.0, 1.0)
        alpha = (0.5 - 0.5 * np.cos(np.pi * np.minimum(fade_x, fade_y)))[..., None]

        blended = (head_np.astype(np.float32) * (1.0 - alpha) + refined * alpha).astype(np.uint8)

        full_arr = np.array(img)
        full_arr[hy1:hy2, hx1:hx2] = blended
        return Image.fromarray(full_arr)
    except Exception as exc:
        logger.debug("Facial micro-clarity refinement skipped: %s", exc)
        return img


def _detect_face_box_cv(avatar_rgb: Image.Image) -> tuple[int, int, int, int] | None:
    """Detect primary frontal face ``(x1, y1, x2, y2)`` via OpenCV Haar cascade when available."""
    try:
        import cv2
        import numpy as np

        gray = cv2.cvtColor(np.array(avatar_rgb), cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if cascade.empty():
            return None
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        if len(faces) == 0:
            return None
        x, y, fw, fh = max(faces, key=lambda b: int(b[2]) * int(b[3]))
        w, h = avatar_rgb.size
        return (
            max(0, int(x - 0.35 * fw)),
            max(0, int(y - 0.45 * fh)),
            min(w, int(x + 1.35 * fw)),
            min(h, int(y + 1.18 * fh)),
        )
    except Exception as exc:
        logger.debug("OpenCV face detection skipped: %s", exc)
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
    quality_preset: str = "auto",
) -> dict[str, Any]:
    """Run full virtual try-on pipeline: analyze wardrobe -> plan combinations -> synthesize try-on looks."""
    if not garment_entries:
        raise ValueError("At least one garment image is required for virtual try-on.")

    t_start = time.perf_counter()
    job_id = uuid.uuid4().hex[:12]

    gpu_ok, gpu_msg = is_gpu_worker_configured()
    if not gpu_ok:
        raise RuntimeError(gpu_msg)

    # Resolve output resolution, diffusion steps, and guidance based on quality preset
    q_key = (quality_preset or "auto").strip().lower()
    if q_key in QUALITY_PRESETS:
        spec = QUALITY_PRESETS[q_key]
        out_w = spec["width"]
        out_h = spec["height"]
        tryon_steps = spec["steps"]
        tryon_guidance = spec["guidance_scale"]
    else:
        out_w = TRYON_OUTPUT_WIDTH
        out_h = TRYON_OUTPUT_HEIGHT
        tryon_steps = TRYON_DEFAULT_STEPS
        tryon_guidance = TRYON_DEFAULT_GUIDANCE

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
            steps=tryon_steps,
            guidance_scale=tryon_guidance,
            seed=combo_seed,
            width=out_w,
            height=out_h,
        )

        # 1. Clean studio backdrop: eliminates vertical wall seams, panel stripes, and background smudges
        out_img = clean_studio_backdrop(out_img)

        # 2. Targeted facial micro-clarity: removes compression pixelation/tearing ("fatna") and sharpens iris, lashes, and hair
        out_img = refine_facial_micro_clarity(out_img)

        # 3. Gentle micro-contrast polish
        try:
            enhancer = ImageEnhance.Sharpness(out_img)
            out_img = enhancer.enhance(1.05)
        except Exception:
            pass

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_filename = f"{job_id}_look_{combo_idx + 1:03d}.png"
        out_path = OUTPUT_DIR / out_filename
        out_img.save(out_path, format="PNG")

        clean_base = base_url.rstrip("/")
        rel_url = f"/api/outputs/{out_filename}"
        public_url = f"{clean_base}{rel_url}" if clean_base else rel_url

        res_str = f"{out_img.width}x{out_img.height}"
        b64_uri = image_to_base64_data_uri(out_img) if include_base64 else None

        def _format_garment_item_details(g: dict[str, Any]) -> str:
            parts = [f"{g['colors']}; {g['patterns_and_design']}"]
            tex = g.get("fabric_texture") or g.get("fabric_type")
            if tex and tex.lower() not in g.get("patterns_and_design", "").lower():
                parts.append(f"Texture: {tex}")
            if g.get("has_logo") and g.get("logo_and_graphics") and "strictly none" not in g.get("logo_and_graphics", "").lower():
                parts.append(f"Logo/Marks: {g['logo_and_graphics']}")
            elif g.get("has_logo") is False:
                parts.append("Logo/Marks: None (Solid Plain Unbranded)")
            return " | ".join(parts)

        garment_details = [
            {
                "role": g["role"],
                "index": g["garment_index"],
                "label": g["title"],
                "details": _format_garment_item_details(g),
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
        }

    workers = max(1, min(len(combos), get_worker_capacity()))
    if len(combos) > 1 and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_synthesize_combo, idx, c) for idx, c in enumerate(combos)]
            generated_looks = [fut.result() for fut in futures]
    else:
        generated_looks = [_synthesize_combo(idx, c) for idx, c in enumerate(combos)]

    total_ms = round((time.perf_counter() - t_start) * 1000, 1)
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
            "combinations_generated": len(generated_looks),
            "output_resolution": f"{TRYON_OUTPUT_WIDTH}x{TRYON_OUTPUT_HEIGHT}",
        },
    }
