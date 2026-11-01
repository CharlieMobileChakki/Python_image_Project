"""Inpainter module: LaMa Fast Fourier Convolutions (FFC) ONNX engine for fabric repair."""
from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# Checkpoint path on Drive D:
LAMA_MODEL_PATH = Path(
    r"D:\huggingface_cache\hub\models--opencv--inpainting_lama\snapshots\aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f\inpainting_lama_2025jan.onnx"
)

_session: ort.InferenceSession | None = None


def load_inpainter_model() -> None:
    """Load LaMa ONNX session once into memory with optimized CPU threads."""
    global _session
    if _session is None:
        if not LAMA_MODEL_PATH.exists():
            raise FileNotFoundError(f"LaMa ONNX checkpoint not found at: {LAMA_MODEL_PATH}")
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = os.cpu_count() or 4
        _session = ort.InferenceSession(
            str(LAMA_MODEL_PATH), sess_opts, providers=["CPUExecutionProvider"]
        )


def is_inpainter_loaded() -> bool:
    """Return True if LaMa ONNX session is active."""
    return _session is not None


def inpaint_patch_512(img_crop_rgb: Image.Image, mask_crop_bin: np.ndarray) -> tuple[np.ndarray, float]:
    """Execute pure LaMa ONNX inpainting on an RGB crop.

    Enforces strictly that input tensors match LaMa's fixed (1, 3, 512, 512) and (1, 1, 512, 512) shapes.
    """
    global _session
    if _session is None:
        load_inpainter_model()

    orig_w, orig_h = img_crop_rgb.size
    # Ensure exact 512x512 resolution for LaMa input
    if (orig_w, orig_h) != (512, 512):
        img_crop_rgb = img_crop_rgb.resize((512, 512), Image.Resampling.BICUBIC)

    if mask_crop_bin.shape != (512, 512):
        mask_im = Image.fromarray(mask_crop_bin.astype(np.uint8), mode="L").resize((512, 512), Image.Resampling.NEAREST)
        mask_crop_bin = np.array(mask_im)

    # Preprocess tensors: float32 in [0, 1]
    img_np = np.array(img_crop_rgb, dtype=np.float32) / 255.0
    img_tensor = np.transpose(img_np, (2, 0, 1))[None, ...].astype(np.float32)
    mask_tensor = (mask_crop_bin > 0).astype(np.float32)[None, None, ...]

    t0 = time.perf_counter()
    out = _session.run(None, {"image": img_tensor, "mask": mask_tensor})
    pure_infer_ms = (time.perf_counter() - t0) * 1000.0

    # Postprocess output: (1, 3, 512, 512) -> (512, 512, 3) in [0, 255]
    out_np = np.clip(np.transpose(out[0][0], (1, 2, 0)), 0, 255).astype(np.uint8)

    # If original crop was not 512x512, restore back to original crop resolution
    if (orig_w, orig_h) != (512, 512):
        out_im = Image.fromarray(out_np, mode="RGB").resize((orig_w, orig_h), Image.Resampling.LANCZOS)
        out_np = np.array(out_im)

    return out_np, round(pure_infer_ms, 2)


def repair_garment_cutout(
    cutout_rgba: Image.Image,
    inpaint_mask: np.ndarray,
    crop_box: tuple[int, int, int, int] | None = None,
) -> tuple[Image.Image, float]:
    """Repair occluded or damaged garment fabric while strictly preserving intact fabric.

    Enforces:
    - Robust execution for any image resolution (even when width or height < 512 px).
    - Bit-for-bit identity outside edit mask (max |dRGB| == 0).
    - Zero background alpha leakage (alpha == 0 for all true background pixels).
    """
    full_w, full_h = cutout_rgba.size

    y_idxs, x_idxs = np.where(inpaint_mask > 0)
    if len(y_idxs) == 0:
        return cutout_rgba, 0.0

    # If image dimensions are smaller than 512 px, pad with transparent border so crop is always 512x512
    pad_w = max(512, full_w)
    pad_h = max(512, full_h)

    if pad_w > full_w or pad_h > full_h:
        padded_rgba = Image.new("RGBA", (pad_w, pad_h), (0, 0, 0, 0))
        padded_rgba.paste(cutout_rgba, (0, 0))

        padded_mask = np.zeros((pad_h, pad_w), dtype=np.uint8)
        padded_mask[:full_h, :full_w] = inpaint_mask
    else:
        padded_rgba = cutout_rgba
        padded_mask = inpaint_mask

    if crop_box is None:
        cx, cy = int(x_idxs.mean()), int(y_idxs.mean())
        x0 = max(0, min(pad_w - 512, cx - 256))
        y0 = max(0, min(pad_h - 512, cy - 256))
        x1 = x0 + 512
        y1 = y0 + 512
    else:
        x0, y0, x1, y1 = crop_box

    crop_rgba = padded_rgba.crop((x0, y0, x1, y1))
    mask_crop = padded_mask[y0:y1, x0:x1]

    if (mask_crop > 0).sum() == 0:
        return cutout_rgba, 0.0

    # Neutralize skin and unwanted object cues inside mask using surrounding fabric interpolation
    crop_rgb_np = np.array(crop_rgba.convert("RGB"))
    mask_bin = (mask_crop > 0).astype(np.uint8)
    prefilled_rgb = cv2.inpaint(crop_rgb_np, mask_bin, 5, cv2.INPAINT_TELEA)
    prefilled_im = Image.fromarray(prefilled_rgb, mode="RGB")

    # Execute LaMa on guaranteed 512x512 crop
    lama_out_np, infer_ms = inpaint_patch_512(prefilled_im, mask_crop)

    # 1. Bit-for-bit identity outside edit mask
    crop_rgba_np = np.array(crop_rgba)
    after_rgb = crop_rgba_np[:, :, :3].copy()
    inpaint_region = (mask_crop > 0)
    after_rgb[inpaint_region] = lama_out_np[inpaint_region]

    # 2. Alpha preservation & extension inside reconstructed fabric
    after_alpha = np.maximum(crop_rgba_np[:, :, 3], mask_crop)
    after_rgba = np.dstack([after_rgb, after_alpha])
    repaired_crop = Image.fromarray(after_rgba, mode="RGBA")

    # Re-embed into padded canvas
    padded_result = padded_rgba.copy()
    padded_result.paste(repaired_crop, (x0, y0))

    # Crop back to original dimensions
    repaired_full = padded_result.crop((0, 0, full_w, full_h))
    return repaired_full, infer_ms

