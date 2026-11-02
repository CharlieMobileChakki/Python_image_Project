"""Image loading and format validation utilities."""
from __future__ import annotations

from io import BytesIO
from PIL import Image, ImageOps


def load_image_bytes(image_bytes: bytes) -> Image.Image:
    """Load raw image bytes, apply EXIF orientation transpose, and convert to RGB.

    Raises:
        ValueError: If image bytes cannot be decoded or format is invalid.
    """
    try:
        raw_image = Image.open(BytesIO(image_bytes))
        image_rgb = ImageOps.exif_transpose(raw_image).convert("RGB")
        return image_rgb
    except Exception as exc:
        raise ValueError("Invalid or corrupted image format. Please upload a valid JPG, PNG, or WEBP.") from exc

