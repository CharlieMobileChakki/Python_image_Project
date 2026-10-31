"""Phase 12: Garment Inpainting Proof-of-Concept.

Reconstructs an occluded garment region (hand covering trousers waistband/pocket)
using Stable Diffusion Inpainting via Hugging Face Diffusers on CPU.
Preserves original visible garment pixels bit-for-bit outside the inpainting mask.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

# Redirect Hugging Face cache to Drive D: to preserve Drive C: space
os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import psutil
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionInpaintPipeline


def get_hardware_info() -> dict[str, Any]:
    """Inspect current host hardware, memory, and CUDA state."""
    vm = psutil.virtual_memory()
    cuda_avail = torch.cuda.is_available()
    return {
        "cuda_available": cuda_avail,
        "device": "cuda" if cuda_avail else "cpu",
        "pytorch_version": torch.__version__,
        "total_ram_gb": round(vm.total / (1024**3), 2),
        "available_ram_gb": round(vm.available / (1024**3), 2),
        "ram_percent_used": vm.percent,
    }


def prepare_inpainting_assets(
    image_path: Path,
    cutout_path: Path,
    class_map_path: Path,
    output_dir: Path,
    crop_x0: int = 650,
    crop_y0: int = 0,
    crop_size: int = 512,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Prepare cropped patches and explicit inpainting mask."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load full assets
    orig_img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    visible_cutout = Image.open(cutout_path).convert("RGBA")
    class_map_img = Image.open(class_map_path).convert("RGB")

    crop_box = (crop_x0, crop_y0, crop_x0 + crop_size, crop_y0 + crop_size)

    # Crop to 512x512
    orig_crop = orig_img.crop(crop_box)
    visible_crop = visible_cutout.crop(crop_box)
    class_crop = np.array(class_map_img.crop(crop_box))

    # Identify Right-arm (color: 192, 128, 128) and Pants (color: 0, 128, 128)
    arm_mask = (class_crop[:, :, 0] == 192) & (class_crop[:, :, 1] == 128) & (class_crop[:, :, 2] == 128)
    pants_mask = (class_crop[:, :, 0] == 0) & (class_crop[:, :, 1] == 128) & (class_crop[:, :, 2] == 128)

    # Explicit inpainting mask: the portion of the occluding hand that covers the trousers waistband and pocket
    # Restrict to Y: [73, 235], X: [135, 215] in crop coordinates
    y_indices = np.arange(crop_size)[:, None]
    x_indices = np.arange(crop_size)[None, :]
    hand_occlusion = arm_mask & (y_indices >= 73) & (y_indices <= 235) & (x_indices >= 135) & (x_indices <= 215)

    # Dilate by 3 pixels using PIL MaxFilter to ensure clean edge coverage of skin
    hand_img = Image.fromarray((hand_occlusion.astype(np.uint8) * 255), mode="L")
    dilated_mask = hand_img.filter(ImageFilter.MaxFilter(7))
    inpaint_mask_np = np.array(dilated_mask)

    # Save initial assets
    paths = {
        "original": output_dir / "original.png",
        "garment_mask": output_dir / "garment_mask.png",
        "visible_cutout": output_dir / "visible_cutout.png",
        "inpaint_mask": output_dir / "inpaint_mask.png",
    }

    orig_crop.save(paths["original"])
    Image.fromarray((pants_mask.astype(np.uint8) * 255), mode="L").save(paths["garment_mask"])
    visible_crop.save(paths["visible_cutout"])
    dilated_mask.save(paths["inpaint_mask"])

    meta = {
        "crop_box": list(crop_box),
        "inpaint_mask_pixels": int((inpaint_mask_np == 255).sum()),
        "pants_crop_pixels": int(pants_mask.sum()),
        "original_dimensions": list(orig_img.size),
        "crop_dimensions": [crop_size, crop_size],
    }

    return paths, meta


def run_garment_inpaint(
    paths: dict[str, Path],
    meta: dict[str, Any],
    output_dir: Path,
    model_id: str = "stable-diffusion-v1-5/stable-diffusion-inpainting",
    num_steps: int = 15,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> dict[str, Any]:
    """Run Stable Diffusion Inpainting pipeline on CPU with DPM-Solver++."""
    print(f"\n--- Loading inpainting model '{model_id}' on CPU ---")
    start_load = time.perf_counter()

    # Load pipeline with low CPU memory usage in float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    # Fast, high-quality convergence with DPM-Solver++ 2M Karras in 12-15 steps
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        use_karras_sigmas=True,
    )
    pipe.enable_attention_slicing()
    # Disable generic safety checker to prevent false-positive blackouts on waistband/pocket crops
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    load_time_s = round(time.perf_counter() - start_load, 2)
    print(f"Model loaded in {load_time_s} seconds.")

    # Prepare working RGB image:
    # 1. Visible garment cutout
    visible_cutout = Image.open(paths["visible_cutout"]).convert("RGBA")
    visible_np = np.array(visible_cutout)
    inpaint_mask = Image.open(paths["inpaint_mask"]).convert("L")
    inpaint_np = np.array(inpaint_mask)

    # Calculate average color of surrounding pants fabric for natural latent initialization
    pants_pixels = visible_np[visible_np[:, :, 3] > 200][:, :3]
    avg_color = pants_pixels.mean(axis=0).astype(np.uint8) if len(pants_pixels) > 0 else np.array([180, 165, 140], dtype=np.uint8)
    print(f"Computed average garment RGB: {list(avg_color)}")

    # Composite working RGB:
    # Neutral gray background where transparent; average garment color in inpaint mask; real fabric elsewhere
    working_rgb = np.full((512, 512, 3), 128, dtype=np.uint8)
    # Fill visible fabric
    alpha = visible_np[:, :, 3]
    vis_mask = alpha > 50
    working_rgb[vis_mask] = visible_np[vis_mask, :3]
    # Fill inpainting region with average garment color to avoid skin-tone bias in latents
    working_rgb[inpaint_np == 255] = avg_color

    working_img = Image.fromarray(working_rgb, mode="RGB")

    # Garment specific prompt
    prompt = (
        "photo of clean beige cotton chino trousers fabric texture, "
        "flat waistband seam, smooth cloth material, matching fabric weave, seamless"
    )
    negative_prompt = (
        "hand, fingers, skin, arm, flesh, human, wristwatch, nail, thumb, "
        "blur, distorted, dark splotches, plastic, holes, background"
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"\n--- Running Inpainting ({num_steps} steps on CPU with DPM-Solver++) ---")
    start_infer = time.perf_counter()
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=working_img,
            mask_image=inpaint_mask,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
    infer_time_s = round(time.perf_counter() - start_infer, 2)
    print(f"Inference completed in {infer_time_s} seconds.")

    generated_rgb = result.images[0]
    gen_np = np.array(generated_rgb)

    # 1. Save raw inpainted crop
    inpainted_crop_path = output_dir / "inpainted_crop.png"
    generated_rgb.save(inpainted_crop_path)

    # 2. Construct final transparent crop:
    # RULE: Keep 100% original garment pixels where inpaint_mask is 0!
    final_rgb = visible_np[:, :, :3].copy()
    # Replace ONLY pixels where inpaint_mask == 255
    inpaint_region = inpaint_np == 255
    final_rgb[inpaint_region] = gen_np[inpaint_region]

    # Alpha channel: Original visible alpha + solid alpha for the reconstructed inpainting region
    final_alpha = np.maximum(visible_np[:, :, 3], inpaint_np)
    final_rgba = np.dstack([final_rgb, final_alpha])
    final_transparent_crop = Image.fromarray(final_rgba, mode="RGBA")

    final_crop_path = output_dir / "final_transparent_crop.png"
    final_transparent_crop.save(final_crop_path)

    # 3. Construct full-resolution transparent PNG with reconstructed patch inserted
    full_cutout = Image.open(r"benchmark_output/phase10/case1_trousers_pants/soft_cutout.png").convert("RGBA")
    full_transparent = full_cutout.copy()
    crop_x0, crop_y0 = meta["crop_box"][0], meta["crop_box"][1]
    full_transparent.paste(final_transparent_crop, (crop_x0, crop_y0))
    full_path = output_dir / "final_transparent.png"
    full_transparent.save(full_path)

    # 4. Create 4-panel side-by-side comparison sheet
    orig_crop = Image.open(paths["original"]).convert("RGBA")
    vis_crop = Image.open(paths["visible_cutout"]).convert("RGBA")
    inpaint_mask_colored = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    # Red transparent overlay on mask
    mask_red = Image.new("RGBA", (512, 512), (255, 50, 50, 160))
    inpaint_mask_colored.paste(mask_red, (0, 0), inpaint_mask)
    mask_preview = Image.alpha_composite(orig_crop, inpaint_mask_colored)

    # Checkered background for transparent previews
    def add_checker(img: Image.Image) -> Image.Image:
        bg = Image.new("RGBA", img.size, (230, 230, 230, 255))
        draw = ImageDraw.Draw(bg)
        sq = 16
        for y in range(0, img.height, sq):
            for x in range(0, img.width, sq):
                if ((x // sq) + (y // sq)) % 2 == 1:
                    draw.rectangle([x, y, x + sq, y + sq], fill=(200, 200, 200, 255))
        return Image.alpha_composite(bg, img).convert("RGB")

    comp_sheet = Image.new("RGB", (512 * 4, 512 + 40), (255, 255, 255))
    draw = ImageDraw.Draw(comp_sheet)

    comp_sheet.paste(orig_crop.convert("RGB"), (0, 40))
    comp_sheet.paste(add_checker(vis_crop), (512, 40))
    comp_sheet.paste(mask_preview.convert("RGB"), (512 * 2, 40))
    comp_sheet.paste(add_checker(final_transparent_crop), (512 * 3, 40))

    draw.text((10, 12), "1. Original Crop (Hand Occlusion)", fill=(0, 0, 0))
    draw.text((512 + 10, 12), "2. Visible Cutout (Excluded Hand)", fill=(0, 0, 0))
    draw.text((512 * 2 + 10, 12), "3. Inpainting Target Mask", fill=(0, 0, 0))
    draw.text((512 * 3 + 10, 12), "4. Final Inpainted Transparent Cutout", fill=(0, 0, 0))

    comp_path = output_dir / "comparison.png"
    comp_sheet.save(comp_path)

    return {
        "model_name": model_id,
        "scheduler": "DPMSolverMultistepScheduler (DPM-Solver++ 2M Karras)",
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "model_load_time_seconds": load_time_s,
        "inference_time_seconds": infer_time_s,
        "inpainted_crop_path": str(inpainted_crop_path),
        "final_transparent_crop_path": str(final_crop_path),
        "final_transparent_path": str(full_path),
        "comparison_path": str(comp_path),
    }


def main() -> None:
    hw_info = get_hardware_info()
    print("=== Hardware & Environment Assessment ===")
    print(f"Device:           {hw_info['device'].upper()}")
    print(f"CUDA Available:   {hw_info['cuda_available']}")
    print(f"PyTorch Version:  {hw_info['pytorch_version']}")
    print(f"Total RAM:        {hw_info['total_ram_gb']} GB")
    print(f"Available RAM:    {hw_info['available_ram_gb']} GB ({hw_info['ram_percent_used']}% used)")

    output_dir = Path("benchmark_output/phase12")
    image_path = Path(r"D:\my codes\Arangtik\backend\uploads\img_mv3nprm1_b7901a.jpg")
    cutout_path = Path(r"benchmark_output/phase10/case1_trousers_pants/soft_cutout.png")
    class_map_path = Path(r"benchmark_output/phase10/case1_trousers_pants/class_map.png")

    print("\n=== Preparing Inpainting Assets ===")
    paths, meta = prepare_inpainting_assets(
        image_path=image_path,
        cutout_path=cutout_path,
        class_map_path=class_map_path,
        output_dir=output_dir,
    )
    print(f"Saved: {paths['original']}")
    print(f"Saved: {paths['garment_mask']}")
    print(f"Saved: {paths['visible_cutout']}")
    print(f"Saved: {paths['inpaint_mask']}")
    print(f"Inpainting Mask Pixels: {meta['inpaint_mask_pixels']} px")

    # Run inpainting
    results = run_garment_inpaint(
        paths=paths,
        meta=meta,
        output_dir=output_dir,
        model_id="stable-diffusion-v1-5/stable-diffusion-inpainting",
        num_steps=12,
        guidance_scale=7.5,
        seed=42,
    )

    # Save metrics JSON
    full_metrics = {
        "hardware": hw_info,
        "input": {
            "image": str(image_path),
            "cutout": str(cutout_path),
            "dimensions": meta["original_dimensions"],
            "crop_box": meta["crop_box"],
            "inpaint_mask_pixels": meta["inpaint_mask_pixels"],
        },
        "model": {
            "model_id": results["model_name"],
            "scheduler": results["scheduler"],
            "steps": results["num_steps"],
            "guidance_scale": results["guidance_scale"],
            "seed": results["seed"],
            "load_time_seconds": results["model_load_time_seconds"],
            "inference_time_seconds": results["inference_time_seconds"],
        },
        "output_files": {
            "original": str(paths["original"]),
            "garment_mask": str(paths["garment_mask"]),
            "visible_cutout": str(paths["visible_cutout"]),
            "inpaint_mask": str(paths["inpaint_mask"]),
            "inpainted_crop": results["inpainted_crop_path"],
            "final_transparent_crop": results["final_transparent_crop_path"],
            "final_transparent": results["final_transparent_path"],
            "comparison": results["comparison_path"],
        },
    }

    metrics_file = output_dir / "metrics.json"
    metrics_file.write_text(json.dumps(full_metrics, indent=2), encoding="utf-8")
    print(f"\nSaved metrics: {metrics_file}")
    print("\n=== Phase 12 Inpainting Run Finished Successfully ===")


if __name__ == "__main__":
    main()

