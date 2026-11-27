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

from PIL import Image

logger = logging.getLogger(__name__)

from app.core.flow_b import (
    generate_garment_via_gpu,
    get_worker_capacity,
    is_gpu_worker_configured,
)
from app.core.flow_b.tryon_stylist import (
    analyze_wardrobe_and_plan_tryon,
    build_qwen_tryon_prompt,
)
from app.core.image_processing import image_to_base64_data_uri, load_image_bytes
from app.core.pipeline.pipeline_config import OUTPUT_DIR

TRYON_OUTPUT_WIDTH = 768
TRYON_OUTPUT_HEIGHT = 1024


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
    canvas_size: int = 512,
) -> Image.Image:
    """Extract a clean neutral-padded head/face identity reference excluding old torso clothes and background pose."""
    w, h = avatar_rgb.size
    if w < 128 or h < 128:
        return avatar_rgb

    crop_coords: tuple[int, int, int, int] | None = None
    if face_box_2d and len(face_box_2d) == 4:
        ymin, xmin, ymax, xmax = face_box_2d
        bw = max(1.0, (xmax - xmin) * w / 1000.0)
        bh = max(1.0, (ymax - ymin) * h / 1000.0)
        x1 = max(0, int(xmin * w / 1000.0 - 0.14 * bw))
        y1 = max(0, int(ymin * h / 1000.0 - 0.12 * bh))
        x2 = min(w, int(xmax * w / 1000.0 + 0.14 * bw))
        y2 = min(h, int(ymax * h / 1000.0 + 0.04 * bh))
        if x2 - x1 >= 48 and y2 - y1 >= 48:
            crop_coords = (x1, y1, x2, y2)

    if crop_coords is None:
        crop_coords = _detect_face_box_cv(avatar_rgb)

    if crop_coords is None:
        return avatar_rgb

    head_crop = avatar_rgb.crop(crop_coords)
    # Tightly fill 86% of a clean neutral studio card (~440px face) for maximum facial pixel fidelity & zero collar leak
    canvas = Image.new("RGB", (canvas_size, canvas_size), (244, 241, 236))
    max_head_dim = int(canvas_size * 0.86)
    head_crop.thumbnail((max_head_dim, max_head_dim), Image.Resampling.LANCZOS)
    px = (canvas_size - head_crop.width) // 2
    py = (canvas_size - head_crop.height) // 2
    canvas.paste(head_crop, (px, py))
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

        prompt_text, total_refs = build_qwen_tryon_prompt(
            avatar_summary=plan["avatar_summary"],
            selected_garments=selected_g_meta,
            selected_accessories=selected_a_meta,
            occasion=combo.get("occasion") or occasion,
            custom_prompt=custom_prompt,
            original_worn_clothes=plan.get("original_worn_clothes", ""),
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

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_filename = f"{job_id}_look_{combo_idx + 1:03d}.png"
        out_path = OUTPUT_DIR / out_filename
        out_img.save(out_path, format="PNG")

        clean_base = base_url.rstrip("/")
        rel_url = f"/api/outputs/{out_filename}"
        public_url = f"{clean_base}{rel_url}" if clean_base else rel_url

        res_str = f"{out_img.width}x{out_img.height}"
        b64_uri = image_to_base64_data_uri(out_img) if include_base64 else None

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
            "garment_details": [
                {
                    "role": g["role"],
                    "index": g["garment_index"],
                    "label": g["title"],
                    "details": f"{g['colors']}; {g['patterns_and_design']}",
                    "body_placement": None,
                }
                for g in selected_g_meta
            ],
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
            "vision_analysis_ms": vision_ms,
            "garments_uploaded": len(garments),
            "accessories_uploaded": len(accessories),
            "combinations_requested": max_combinations,
            "combinations_generated": len(generated_looks),
            "output_resolution": f"{TRYON_OUTPUT_WIDTH}x{TRYON_OUTPUT_HEIGHT}",
        },
    }
