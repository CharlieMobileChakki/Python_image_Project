"""Preprocessing helpers for background removal."""
from __future__ import annotations

from io import BytesIO
import os
from PIL import Image, ImageOps

MAX_INPUT_EDGE = int(os.getenv("MAX_INPUT_EDGE", "2400"))


def prepare_image_for_background_removal(image_bytes: bytes) -> tuple[Image.Image, tuple[int, int]]:
    """Validate, orient via EXIF, clamp to max input dimension, and return RGB PIL Image."""
    try:
        image = Image.open(BytesIO(image_bytes))
        image_format = (image.format or "").upper()
        if image_format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Please upload a JPG, PNG, or WEBP image.")
        image = ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a valid/safe image.") from exc

    original_size = image.size
    longest_edge = max(image.size)
    if longest_edge > MAX_INPUT_EDGE:
        scale = MAX_INPUT_EDGE / longest_edge
        resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(resized, Image.Resampling.LANCZOS)

    return image, original_size

