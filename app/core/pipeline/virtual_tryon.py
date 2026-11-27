"""End-to-end Virtual Try-On & AI Outfit Stylist pipeline orchestration.

Combines ChatGPT Vision garment/accessory pattern analysis & combination curation
with multi-image Qwen-Image-2.1-Turbo synthesis on the remote GPU worker.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any
import uuid

from PIL import Image

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
        )
        condition_images: list[Image.Image] = (
            [avatar_rgb]
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
