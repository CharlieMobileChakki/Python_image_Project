from __future__ import annotations

import io
import math
from typing import Tuple

from PIL import Image, ImageChops, ImageFilter, ImageOps


ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


def load_pil_image(raw: bytes, name: str) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(raw))
        source_format = image.format
        image = ImageOps.exif_transpose(image)
        image.format = source_format
        image.load()
    except Exception as exc:
        raise ValueError(f"Could not decode image '{name}': {exc}") from exc

    if image.format not in ALLOWED_FORMATS:
        raise ValueError(f"Unsupported image format for '{name}'. Use PNG, JPEG or WEBP.")
    return image


def resize_to_sdxl(image: Image.Image, max_side: int) -> Tuple[Image.Image, Tuple[int, int]]:
    """Resize while preserving aspect ratio; SDXL dimensions must be multiples of 8."""
    original_size = image.size
    width, height = image.size
    if width < 64 or height < 64:
        raise ValueError("Images must be at least 64x64 pixels.")

    scale = min(1.0, float(max_side) / float(max(width, height)))
    new_width = max(64, int(round((width * scale) / 8.0)) * 8)
    new_height = max(64, int(round((height * scale) / 8.0)) * 8)
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized, original_size


def prepare_mask(mask: Image.Image, target_size: Tuple[int, int], blur_radius: float) -> Image.Image:
    """White means regenerate; black means preserve. Mask is always aligned to damaged image."""
    mask = mask.convert("L").resize(target_size, Image.Resampling.NEAREST)
    # Threshold first so anti-aliased upload artifacts do not turn the whole image into a mask.
    mask = mask.point(lambda value: 255 if value >= 128 else 0)
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return mask


def composite_garment(
    original_rgba: Image.Image,
    generated_rgb: Image.Image,
    repair_mask: Image.Image,
) -> Image.Image:
    """Use generated pixels only inside the repair mask and keep outside pixels/alpha intact."""
    original_rgba = original_rgba.convert("RGBA")
    original_rgb = original_rgba.convert("RGB")
    generated_rgb = generated_rgb.convert("RGB").resize(original_rgba.size, Image.Resampling.LANCZOS)
    repair_mask = repair_mask.convert("L").resize(original_rgba.size, Image.Resampling.LANCZOS)

    final_rgb = Image.composite(generated_rgb, original_rgb, repair_mask)

    # The repair mask must describe garment pixels only, never the transparent background.
    # Newly reconstructed garment pixels become opaque; all other original alpha is retained.
    final_alpha = ImageChops.lighter(original_rgba.getchannel("A"), repair_mask)
    final_rgba = final_rgb.convert("RGBA")
    final_rgba.putalpha(final_alpha)
    return final_rgba


def validate_repair_mask(mask: Image.Image) -> None:
    mask_l = mask.convert("L")
    histogram = mask_l.histogram()
    white_pixels = sum(histogram[128:])
    total_pixels = mask_l.width * mask_l.height
    ratio = white_pixels / max(1, total_pixels)
    if white_pixels == 0:
        raise ValueError("Repair mask is empty. Paint the missing garment region white.")
    if ratio > 0.65:
        raise ValueError(
            f"Repair mask covers {ratio:.1%} of the canvas. This is probably too large. "
            "White should cover only missing/damaged garment fabric, not the whole background."
        )
