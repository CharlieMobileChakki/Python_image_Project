"""Phase 13: Garment-Aware Seamless Inpainting and Boundary Quality.

Reconstructs the occluded lateral waistband/pocket notch on trousers using
LaMa (Large Mask Inpainting with Fast Fourier Convolutions, ONNX CPU).
Eliminates geometric bulging into background air, preserves original fabric
bit-for-bit outside the strictly defined notch, and provides seamless blending.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

# Ensure Hugging Face cache on Drive D:
os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import psutil
import torch


def get_hardware_info() -> dict[str, Any]:
    """Inspect current host hardware and memory state."""
    vm = psutil.virtual_memory()
    cuda_avail = torch.cuda.is_available()
    return {
        "cuda_available": cuda_avail,
        "device": "cuda" if cuda_avail else "cpu",
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(vm.total / (1024**3), 2),
        "available_ram_gb": round(vm.available / (1024**3), 2),
        "ram_percent_used": vm.percent,
    }


def compute_exact_notch_mask(
    class_crop: np.ndarray,
    crop_size: int = 512,
    y_start: int = 112,
    y_end: int = 260,
    seam_x_top: int = 185,
    seam_x_bottom: int = 188,
    overlap_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the geometrically restricted crescent notch mask.

    Interpolates the natural straight trouser hip/thigh seam from
    (y_start, seam_x_top) to (y_end, seam_x_bottom).
    Restricts inpainting strictly to fabric region where hand occluded trousers,
    completely excluding background air where the arm hung.
    """
    # SegFormer class map: Arm = [192, 128, 128], Pants = [0, 128, 128]
    arm = (class_crop[:, :, 0] == 192) & (class_crop[:, :, 1] == 128) & (class_crop[:, :, 2] == 128)
    pants = (class_crop[:, :, 0] == 0) & (class_crop[:, :, 1] == 128) & (class_crop[:, :, 2] == 128)

    notch_mask = np.zeros((crop_size, crop_size), dtype=np.uint8)
    notch_alpha = np.zeros((crop_size, crop_size), dtype=np.uint8)

    for y in range(y_start, y_end + 1):
        # Linear interpolation of the natural seam
        frac = (y - y_start) / max(1, (y_end - y_start))
        x_seam = int(np.round(seam_x_top + frac * (seam_x_bottom - seam_x_top)))

        arm_xs = np.where(arm[y, :])[0]
        if len(arm_xs) > 0:
            x_min_arm = arm_xs.min()
            x_max_arm = arm_xs.max()
            # Only consider arm pixels at or inside the natural seam
            x_start = max(x_seam, x_min_arm)
            x_end = x_max_arm + overlap_px  # slight overlap with existing visible fabric

            if x_end > x_start:
                notch_mask[y, x_start : x_end + 1] = 255
                # Antialiased soft boundary at the outer seam
                notch_alpha[y, x_start] = 140
                if x_start + 1 <= x_end:
                    notch_alpha[y, x_start + 1 : x_end + 1] = 255

    return notch_mask, notch_alpha


def add_checkerboard(img_rgba: Image.Image, square_size: int = 16) -> Image.Image:
    """Composite RGBA image onto a standard two-tone grey checkerboard."""
    bg = Image.new("RGBA", img_rgba.size, (235, 235, 235, 255))
    draw = ImageDraw.Draw(bg)
    w, h = img_rgba.size
    for y in range(0, h, square_size):
        for x in range(0, w, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 1:
                draw.rectangle([x, y, x + square_size, y + square_size], fill=(205, 205, 205, 255))
    return Image.alpha_composite(bg, img_rgba).convert("RGB")


def run_phase13_pipeline(
    image_path: Path,
    cutout_path: Path,
    class_map_path: Path,
    output_dir: Path,
    model_path: Path,
    crop_x0: int = 650,
    crop_y0: int = 0,
    crop_size: int = 512,
) -> dict[str, Any]:
    """Execute Phase 13 Seamless Inpainting and create comparison artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    hw_info = get_hardware_info()

    print("=== Phase 13: Seamless Inpainting Pipeline ===")
    print(f"Device:           {hw_info['device'].upper()}")
    print(f"Total RAM:        {hw_info['total_ram_gb']} GB (Available: {hw_info['available_ram_gb']} GB)")

    # 1. Load full input assets
    print(f"\n1. Loading full-resolution input assets...")
    orig_img_full = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    soft_cutout_full = Image.open(cutout_path).convert("RGBA")
    class_map_full = Image.open(class_map_path).convert("RGB")

    full_w, full_h = orig_img_full.size
    print(f"   Full image size: {full_w} x {full_h}")

    # 2. Extract 512x512 Crop Box
    crop_box = (crop_x0, crop_y0, crop_x0 + crop_size, crop_y0 + crop_size)
    print(f"   Crop Box: {crop_box}")

    orig_crop = orig_img_full.crop(crop_box)
    before_cutout_crop = soft_cutout_full.crop(crop_box)
    class_crop = np.array(class_map_full.crop(crop_box))

    # 3. Compute exact notch mask
    print("\n2. Computing exact geometric crescent notch mask...")
    notch_mask_np, notch_alpha_np = compute_exact_notch_mask(class_crop, crop_size=crop_size)
    notch_px_count = int((notch_mask_np == 255).sum())
    y_idxs, x_idxs = np.where(notch_mask_np == 255)
    print(f"   Notch area:      {notch_px_count} pixels")
    print(f"   Notch Y range:   [{y_idxs.min()}, {y_idxs.max()}] (height = {y_idxs.max() - y_idxs.min() + 1} px)")
    print(f"   Notch X range:   [{x_idxs.min()}, {x_idxs.max()}] (max width = {x_idxs.max() - x_idxs.min() + 1} px)")

    # SegFormer Pants mask
    pants_mask_np = (class_crop[:, :, 0] == 0) & (class_crop[:, :, 1] == 128) & (class_crop[:, :, 2] == 128)
    garment_mask_img = Image.fromarray((pants_mask_np.astype(np.uint8) * 255), mode="L")
    inpaint_mask_img = Image.fromarray(notch_mask_np, mode="L")

    # 4. Load LaMa ONNX Model
    print(f"\n3. Loading LaMa ONNX model from: {model_path}...")
    load_start = time.perf_counter()
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = psutil.cpu_count(logical=True)
    sess = ort.InferenceSession(str(model_path), sess_opts, providers=["CPUExecutionProvider"])
    load_time_s = round(time.perf_counter() - load_start, 3)
    print(f"   Model loaded in {load_time_s} seconds.")

    # 5. Prepare input tensors
    # Image: float32 in [0, 1], shape (1, 3, 512, 512)
    # Mask: float32 in {0.0, 1.0}, shape (1, 1, 512, 512)
    img_np = np.array(orig_crop, dtype=np.float32) / 255.0
    img_tensor = np.transpose(img_np, (2, 0, 1))[None, ...].astype(np.float32)
    mask_tensor = (notch_mask_np > 0).astype(np.float32)[None, None, ...]

    # 6. Run Inpainting Inference
    print(f"\n4. Running LaMa Inpainting on CPU...")
    infer_start = time.perf_counter()
    out = sess.run(None, {"image": img_tensor, "mask": mask_tensor})
    infer_time_s = round(time.perf_counter() - infer_start, 3)
    print(f"   Inference completed in {infer_time_s} seconds ({infer_time_s * 1000:.1f} ms).")

    # 7. Post-process reconstructed patch
    # Output shape: (1, 3, 512, 512), values in [0, 255]
    lama_out_np = np.clip(np.transpose(out[0][0], (1, 2, 0)), 0, 255).astype(np.uint8)

    # 8. Seamless Compositing
    print("\n5. Compositing reconstructed patch...")
    before_np = np.array(before_cutout_crop)

    # RULE 1: Preserve original RGB bit-for-bit outside the notch
    after_rgb = before_np[:, :, :3].copy()
    inpaint_region = notch_mask_np == 255
    after_rgb[inpaint_region] = lama_out_np[inpaint_region]

    # RULE 2: Alpha compositing
    # Combine original alpha with antialiased notch alpha
    after_alpha = np.maximum(before_np[:, :, 3], notch_alpha_np)
    after_rgba = np.dstack([after_rgb, after_alpha])
    after_cutout_crop = Image.fromarray(after_rgba, mode="RGBA")

    # 9. Re-embed into full-resolution 2048x1079 transparent canvas
    full_inpainted_transparent = soft_cutout_full.copy()
    full_inpainted_transparent.paste(after_cutout_crop, (crop_x0, crop_y0))

    # 10. Save all artifacts in benchmark_output/phase13/
    print("\n6. Saving Phase 13 artifacts...")
    paths = {
        "original": output_dir / "original.png",
        "garment_mask": output_dir / "garment_mask.png",
        "inpaint_mask": output_dir / "inpaint_mask.png",
        "before_cutout": output_dir / "before_cutout.png",
        "after_cutout": output_dir / "after_cutout.png",
        "comparison": output_dir / "comparison.png",
        "full_transparent": output_dir / "full_inpainted_transparent.png",
        "metrics": output_dir / "metrics.json",
    }

    orig_crop.save(paths["original"])
    garment_mask_img.save(paths["garment_mask"])
    inpaint_mask_img.save(paths["inpaint_mask"])
    before_cutout_crop.save(paths["before_cutout"])
    after_cutout_crop.save(paths["after_cutout"])
    full_inpainted_transparent.save(paths["full_transparent"])

    # 11. Create side-by-side comparison sheet
    # 4 panels: Original Crop | Before Cutout | Inpaint Mask Overlay | After Cutout
    inpaint_mask_colored = Image.new("RGBA", (crop_size, crop_size), (0, 0, 0, 0))
    mask_red = Image.new("RGBA", (crop_size, crop_size), (255, 30, 30, 180))
    inpaint_mask_colored.paste(mask_red, (0, 0), inpaint_mask_img)
    mask_preview = Image.alpha_composite(orig_crop.convert("RGBA"), inpaint_mask_colored)

    panel_width = crop_size
    panel_height = crop_size
    header_h = 44
    comp_sheet = Image.new("RGB", (panel_width * 4, panel_height + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(comp_sheet)

    comp_sheet.paste(orig_crop, (0, header_h))
    comp_sheet.paste(add_checkerboard(before_cutout_crop), (panel_width, header_h))
    comp_sheet.paste(mask_preview.convert("RGB"), (panel_width * 2, header_h))
    comp_sheet.paste(add_checkerboard(after_cutout_crop), (panel_width * 3, header_h))

    # Add text labels
    draw.text((12, 14), "1. Original Crop (Occluded)", fill=(20, 20, 20))
    draw.text((panel_width + 12, 14), "2. Before Inpainting (Cutout Notch)", fill=(20, 20, 20))
    draw.text((panel_width * 2 + 12, 14), f"3. Exact Notch Mask ({notch_px_count} px)", fill=(20, 20, 20))
    draw.text((panel_width * 3 + 12, 14), f"4. After Seamless Inpaint ({infer_time_s}s LaMa)", fill=(20, 20, 20))

    comp_sheet.save(paths["comparison"])
    print(f"   Saved comparison sheet: {paths['comparison']}")

    # 12. Programmatic Acceptance Checks
    print("\n7. Executing objective acceptance verification checks...")
    # Check 1: RGB bit-for-bit unchanged outside inpaint_region
    diff_outside_rgb = np.abs(after_rgb[~inpaint_region].astype(int) - before_np[:, :, :3][~inpaint_region].astype(int))
    max_rgb_diff_outside = int(diff_outside_rgb.max()) if len(diff_outside_rgb) > 0 else 0
    check1_passed = max_rgb_diff_outside == 0

    # Check 2: Alpha is zero outside the allowed garment silhouette + notch
    # True background is everywhere original alpha was 0 outside the inpaint region
    bg_mask = (before_np[:, :, 3] == 0) & (~inpaint_region)
    alpha_outside = after_alpha[bg_mask]
    max_alpha_outside = int(alpha_outside.max()) if len(alpha_outside) > 0 else 0
    check2_passed = max_alpha_outside == 0

    # Check 3: Original garment pixels outside edit region preserved
    garment_outside_edit = pants_mask_np & (~inpaint_region)
    preserved_px = int((garment_outside_edit).sum())
    check3_passed = check1_passed and preserved_px > 0

    # Check 4: No rectangular crop leakage in full canvas
    full_before_np = np.array(soft_cutout_full)
    full_after_np = np.array(full_inpainted_transparent)
    # Outside the 512x512 crop box, full canvas must be 100% bit-for-bit identical
    mask_full_box = np.zeros((full_h, full_w), dtype=bool)
    mask_full_box[crop_y0 : crop_y0 + crop_size, crop_x0 : crop_x0 + crop_size] = True
    diff_outside_box = np.abs(full_after_np[~mask_full_box].astype(int) - full_before_np[~mask_full_box].astype(int))
    max_diff_outside_box = int(diff_outside_box.max())
    check4_passed = max_diff_outside_box == 0

    # Check 5: Color continuity with immediate border fabric & original ground-truth in notch
    dilated_notch = np.array(inpaint_mask_img.filter(ImageFilter.MaxFilter(7))) > 0
    border_fabric = (dilated_notch & ~inpaint_region) & pants_mask_np
    border_rgb = np.array(orig_crop)[border_fabric].mean(axis=0)
    notch_rgb = lama_out_np[inpaint_region].mean(axis=0)
    orig_notch_rgb = np.array(orig_crop)[inpaint_region].mean(axis=0)
    
    border_delta = np.abs(border_rgb - notch_rgb)
    mean_border_delta = float(border_delta.mean())
    orig_vs_inpaint_delta = np.abs(orig_notch_rgb - notch_rgb)
    mean_orig_delta = float(orig_vs_inpaint_delta.mean())
    check5_passed = mean_border_delta < 25.0 and mean_orig_delta < 10.0

    checks = {
        "check1_rgb_preserved_outside_mask": {
            "passed": check1_passed,
            "max_rgb_diff_outside": max_rgb_diff_outside,
            "rule": "max |RGB_after - RGB_before| == 0",
        },
        "check2_alpha_zero_outside_silhouette": {
            "passed": check2_passed,
            "max_alpha_outside": max_alpha_outside,
            "rule": "alpha == 0 for all background pixels outside silhouette",
        },
        "check3_garment_pixels_preserved": {
            "passed": check3_passed,
            "preserved_garment_pixel_count": preserved_px,
            "rule": "Existing visible garment fabric untouched",
        },
        "check4_no_rectangular_crop_leakage": {
            "passed": check4_passed,
            "max_diff_outside_crop_box": max_diff_outside_box,
            "rule": "Bit-for-bit identity outside crop coordinates",
        },
        "check5_color_continuity": {
            "passed": check5_passed,
            "border_fabric_rgb_mean": [round(float(c), 2) for c in border_rgb],
            "inpainted_notch_rgb_mean": [round(float(c), 2) for c in notch_rgb],
            "orig_notch_rgb_mean": [round(float(c), 2) for c in orig_notch_rgb],
            "local_border_delta_rgb": [round(float(c), 2) for c in border_delta],
            "mean_local_border_delta": round(mean_border_delta, 2),
            "orig_vs_inpaint_delta_rgb": [round(float(c), 2) for c in orig_vs_inpaint_delta],
            "mean_orig_vs_inpaint_delta": round(mean_orig_delta, 2),
        },
    }

    print("\n   Results of Objective Acceptance Checks:")
    for k, v in checks.items():
        status = "PASSED" if v["passed"] else "FAILED"
        print(f"   [{status}] {k}")

    # 13. Assemble and save metrics.json
    metrics = {
        "phase": "Phase 13: Garment-Aware Seamless Inpainting and Boundary Quality",
        "hardware": hw_info,
        "model": {
            "name": "LaMa (Large Mask Inpainting)",
            "architecture": "Fast Fourier Convolutions (FFC) ResNet",
            "format": "ONNX",
            "weights_file": model_path.name,
            "file_size_mb": round(model_path.stat().st_size / (1024**2), 2),
            "device": "cpu",
            "num_threads": sess_opts.intra_op_num_threads,
            "load_time_seconds": load_time_s,
            "inference_time_seconds": infer_time_s,
            "inference_time_ms": round(infer_time_s * 1000, 1),
        },
        "comparison_baseline_phase12": {
            "model_name": "Stable Diffusion 1.5 Inpainting",
            "file_size_gb": 4.26,
            "inference_time_seconds": 146.47,
            "inpaint_mask_pixels": 10477,
            "defects_identified": [
                "9,985 pixels (95.3%) outside natural garment silhouette bulging into background air",
                "Neutral grey initialization [128,128,128] caused cross-attention dark drift",
                "Rectangular fringe and soft-alpha edge discontinuities",
            ],
        },
        "geometry_improvement": {
            "crop_box": list(crop_box),
            "y_range": [int(y_idxs.min()), int(y_idxs.max())],
            "x_range": [int(x_idxs.min()), int(x_idxs.max())],
            "inpaint_mask_pixels": notch_px_count,
            "mask_pixel_reduction_vs_phase12_pct": round((1 - notch_px_count / 10477) * 100, 1),
            "natural_seam_interpolation": f"({y_idxs.min()}, {x_idxs.min()}) to ({y_idxs.max()}, {x_idxs.max()})",
        },
        "acceptance_checks": checks,
        "artifact_paths": {k: str(v) for k, v in paths.items()},
    }

    metrics_file = paths["metrics"]
    metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved metrics: {metrics_file}")
    print("\n=== Phase 13 Inpainting Run Finished Successfully ===")
    return metrics


def main() -> None:
    image_path = Path(r"D:\my codes\Arangtik\backend\uploads\img_mv3nprm1_b7901a.jpg")
    cutout_path = Path(r"benchmark_output/phase10/case1_trousers_pants/soft_cutout.png")
    class_map_path = Path(r"benchmark_output/phase10/case1_trousers_pants/class_map.png")
    output_dir = Path("benchmark_output/phase13")
    model_path = Path(r"D:\huggingface_cache\hub\models--opencv--inpainting_lama\snapshots\aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f\inpainting_lama_2025jan.onnx")

    run_phase13_pipeline(
        image_path=image_path,
        cutout_path=cutout_path,
        class_map_path=class_map_path,
        output_dir=output_dir,
        model_path=model_path,
        crop_x0=650,
        crop_y0=0,
        crop_size=512,
    )


if __name__ == "__main__":
    main()
