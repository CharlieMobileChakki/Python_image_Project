"""Benchmark suite for Qwen-Image-2.1 Garment Generation on Arangtik test cases.

Evaluates:
1. Magenta Suit Salwar (unnamed.jpg)
2. White Kerala Saree (pexels-abhijith-ts-33843905-18216607.jpg)
3. Trouser Product (trouser reference matching the successful HF demo)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time

from PIL import Image

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.experimental.qwen_generation import (
    DEFAULT_GARMENT_TRANSPARENT_PROMPT,
    get_qwen_garment_engine,
)

BENCHMARK_DIR = Path("outputs/benchmarks/qwen_results")
BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

TEST_CASES = [
    {
        "id": "case_1_magenta_suit",
        "name": "Magenta Suit Salwar",
        "category": "suit_salwar",
        "input_path": Path("input_images/unnamed.jpg"),
        "custom_prompt": (
            "This is an RGBA image with transparency. "
            "A clean studio product photograph of the complete standalone magenta ethnic suit salwar. "
            "Reconstruct the continuous waist fabric where the hands were crossed. "
            "Preserve the vibrant magenta color, golden sleeve trim, and matching dupatta. "
            "The image has an alpha channel and the background is transparent."
        ),
    },
    {
        "id": "case_2_kerala_saree",
        "name": "White Kerala Saree with Red Blouse",
        "category": "saree",
        "input_path": Path("input_images/pexels-abhijith-ts-33843905-18216607.jpg"),
        "custom_prompt": (
            "This is an RGBA image with transparency. "
            "A clean studio product photograph of the complete standalone Kerala kasavu saree and red blouse. "
            "Reconstruct the pleated skirt drape and waistline. Preserve the gold border and red blouse. "
            "Remove the model completely. The image has an alpha channel and the background is transparent."
        ),
    },
    {
        "id": "case_3_trouser_product",
        "name": "Standalone Trouser Product",
        "category": "pants",
        "input_path": Path("outputs/f9e86c8b5c_pants.png"),
        "custom_prompt": (
            "This is an RGBA image with transparency. "
            "A clean studio product photograph of the complete standalone trousers laid flat. "
            "Reconstruct all folds, seams, and waistband. Center the product. "
            "The image has an alpha channel and the background is transparent."
        ),
    },
]


def run_benchmark():
    print("=====================================================================")
    print("=== Arangtik Garment AI: Qwen-Image-2.1 Benchmark Evaluation Suite ===")
    print("=====================================================================")
    print(f"Results Directory: {BENCHMARK_DIR.resolve()}\n")

    engine = get_qwen_garment_engine()
    is_avail, avail_reason = engine.is_available()

    print(f"Model ID: {engine.model_id}")
    print(f"Configured Device: {engine.configured_device}")
    print(f"Availability Status: {is_avail}")
    print(f"Environment Detail: {avail_reason}\n")

    summary_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": engine.model_id,
        "configured_device": engine.configured_device,
        "is_available": is_avail,
        "availability_reason": avail_reason,
        "cases": [],
    }

    for case in TEST_CASES:
        case_id = case["id"]
        case_name = case["name"]
        input_path = case["input_path"]

        print(f"---------------------------------------------------------------------")
        print(f"Testing {case_id}: {case_name}")
        print(f"Input: {input_path}")

        if not input_path.exists():
            print(f"  WARNING: Input file {input_path} not found. Skipping.")
            continue

        # Save reference copy
        case_dir = BENCHMARK_DIR / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        dest_input = case_dir / f"input_{input_path.name}"
        shutil.copy(input_path, dest_input)

        raw_img = Image.open(input_path).convert("RGB")
        print(f"  Image Size: {raw_img.size}")

        t0 = time.perf_counter()
        result = engine.generate(
            reference_image=raw_img,
            prompt=case["custom_prompt"],
            resolution=1024,
            inference_steps=28,
            seed=42,
            transparent_background=True,
        )
        elapsed = time.perf_counter() - t0

        print(f"  Status: {result.status.upper()}")
        print(f"  Inference Time: {result.inference_time_seconds}s (Total wall: {elapsed:.2f}s)")
        print(f"  Peak VRAM: {result.gpu_memory_peak_mb} MB")
        print(f"  Has Alpha Channel: {result.has_alpha_channel}")
        if result.error_detail:
            print(f"  Note: {result.error_detail}")

        # Save generated output
        out_file = case_dir / f"output_qwen_{case_id}.png"
        result.image.save(out_file, format="PNG")

        case_summary = {
            "case_id": case_id,
            "case_name": case_name,
            "category": case["category"],
            "input_resolution": list(raw_img.size),
            "output_resolution": [result.width, result.height],
            "has_alpha_channel": result.has_alpha_channel,
            "inference_time_seconds": result.inference_time_seconds,
            "peak_vram_mb": result.gpu_memory_peak_mb,
            "status": result.status,
            "error_detail": result.error_detail,
        }
        summary_results["cases"].append(case_summary)

    # Save complete JSON report
    report_file = BENCHMARK_DIR / "benchmark_summary.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n=====================================================================")
    print(f"Benchmark summary written to: {report_file}")
    print(f"=====================================================================")
    return summary_results


if __name__ == "__main__":
    run_benchmark()

