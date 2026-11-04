"""Garment reconstruction utilities: mask validation, reference preparation, and compositing.

Extracted and adapted from arangtik_garment_reconstruction standalone project.
Enforces:
- White means regenerate; black means preserve.
- Mask coverage validation (prevents over-painting intact fabric).
- High-fidelity safe masked compositing.
- Aspect-ratio preserving dimension alignment.
"""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image, ImageChops, ImageFilter, ImageOps

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


def load_pil_image(raw: bytes, name: str = "image") -> Image.Image:
    """Safely decode and normalize EXIF orientation for an uploaded image."""
    try:
        image = Image.open(io.BytesIO(raw))
        source_format = image.format
        image = ImageOps.exif_transpose(image)
        image.format = source_format
        image.load()
    except Exception as exc:
        raise ValueError(f"Could not decode image '{name}': {exc}") from exc

    if (image.format or "").upper() not in ALLOWED_FORMATS:
        raise ValueError(f"Unsupported image format for '{name}'. Use PNG, JPEG or WEBP.")
    return image


def resize_to_multiple_of_8(
    image: Image.Image, max_side: int = 1024
) -> Tuple[Image.Image, Tuple[int, int]]:
    """Resize while preserving aspect ratio; diffusion dimensions must be multiples of 8."""
    original_size = image.size
    width, height = image.size
    if width < 64 or height < 64:
        raise ValueError("Images must be at least 64x64 pixels.")

    scale = min(1.0, float(max_side) / float(max(width, height)))
    new_width = max(64, int(round((width * scale) / 8.0)) * 8)
    new_height = max(64, int(round((height * scale) / 8.0)) * 8)
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized, original_size


def prepare_mask(
    mask: Image.Image, target_size: Tuple[int, int], blur_radius: float = 1.5
) -> Image.Image:
    """Prepare binary repair mask: white (255) means regenerate, black (0) means preserve."""
    mask_l = mask.convert("L").resize(target_size, Image.Resampling.NEAREST)
    mask_thresh = mask_l.point(lambda value: 255 if value >= 128 else 0)
    if blur_radius > 0:
        mask_thresh = mask_thresh.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return mask_thresh


def validate_repair_mask(mask: Image.Image, max_coverage_ratio: float = 0.65) -> None:
    """Verify that repair mask is not empty and does not cover excessively large areas."""
    mask_l = mask.convert("L")
    histogram = mask_l.histogram()
    white_pixels = sum(histogram[128:])
    total_pixels = mask_l.width * mask_l.height
    ratio = white_pixels / max(1, total_pixels)

    if white_pixels == 0:
        raise ValueError("Repair mask is empty. Paint the missing garment region white.")
    if ratio > max_coverage_ratio:
        raise ValueError(
            f"Repair mask covers {ratio:.1%} of the canvas. This is too large. "
            "White should cover only missing/damaged garment fabric, not the whole background."
        )


def composite_garment(
    original_rgba: Image.Image,
    generated_rgb: Image.Image,
    repair_mask: Image.Image,
) -> Image.Image:
    """Use generated pixels only inside the repair mask, keeping untouched pixels bit-for-bit intact.

    Newly reconstructed garment pixels become opaque; all original intact alpha is retained.
    """
    original_rgba = original_rgba.convert("RGBA")
    original_rgb = original_rgba.convert("RGB")
    generated_rgb = generated_rgb.convert("RGB").resize(original_rgba.size, Image.Resampling.LANCZOS)
    repair_mask = repair_mask.convert("L").resize(original_rgba.size, Image.Resampling.LANCZOS)

    final_rgb = Image.composite(generated_rgb, original_rgb, repair_mask)
    final_alpha = ImageChops.lighter(original_rgba.getchannel("A"), repair_mask)

    final_rgba = final_rgb.convert("RGBA")
    final_rgba.putalpha(final_alpha)
    return final_rgba


def extract_garment_reference_crop(
    original_image: Image.Image,
    garment_mask: np.ndarray | None = None,
    padding: int = 24,
) -> Image.Image:
    """Extract a tight garment-only reference crop excluding human face and background.

    Used by IP-Adapter Plus to condition visual generation strictly on fabric texture.
    """
    orig_rgb = original_image.convert("RGB")
    if garment_mask is None or garment_mask.sum() == 0:
        w, h = orig_rgb.size
        return orig_rgb.crop((int(0.2 * w), int(0.25 * h), int(0.8 * w), int(0.85 * h)))

    y_idxs, x_idxs = np.where(garment_mask > 0)
    w, h = orig_rgb.size
    x0 = max(0, int(x_idxs.min()) - padding)
    y0 = max(0, int(y_idxs.min()) - padding)
    x1 = min(w, int(x_idxs.max()) + 1 + padding)
    y1 = min(h, int(y_idxs.max()) + 1 + padding)

    return orig_rgb.crop((x0, y0, x1, y1))

