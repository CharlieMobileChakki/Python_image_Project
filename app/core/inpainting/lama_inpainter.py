"""LaMa Fast Fourier Convolutions (FFC) ONNX engine for fabric repair."""
from __future__ import annotations

import time
import cv2
import numpy as np
from PIL import Image

from app.core.inpainting.inpainting_utils import (
    ensure_mask_dimension,
    ensure_patch_dimension,
)
from app.core.model_manager.model_loader import get_inpainter_session


def inpaint_patch_512(img_crop_rgb: Image.Image, mask_crop_bin: np.ndarray) -> tuple[np.ndarray, float]:
    """Execute pure LaMa ONNX inpainting on an RGB crop.

    Enforces strictly that input tensors match LaMa's fixed (1, 3, 512, 512) and (1, 1, 512, 512) shapes.
    """
    session = get_inpainter_session()

    orig_w, orig_h = img_crop_rgb.size
    img_crop_rgb = ensure_patch_dimension(img_crop_rgb, 512)
    mask_crop_bin = ensure_mask_dimension(mask_crop_bin, 512)

    # Preprocess tensors: float32 in [0, 1]
    img_np = np.array(img_crop_rgb, dtype=np.float32) / 255.0
    img_tensor = np.transpose(img_np, (2, 0, 1))[None, ...].astype(np.float32)
    mask_tensor = (mask_crop_bin > 0).astype(np.float32)[None, None, ...]

    t0 = time.perf_counter()
    out = session.run(None, {"image": img_tensor, "mask": mask_tensor})
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
    inpaint_fn: Any | None = None,
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
    fn = inpaint_fn if inpaint_fn is not None else inpaint_patch_512
    lama_out_np, infer_ms = fn(prefilled_im, mask_crop)

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
