"""Image utility helpers (Base64 encoding, format conversion)."""
from __future__ import annotations

import base64
from io import BytesIO
from PIL import Image


def image_to_base64_data_uri(img_rgba: Image.Image) -> str:
    """Convert PIL RGBA image to Base64 PNG data URI."""
    buf = BytesIO()
    img_rgba.save(buf, format="PNG", optimize=True)
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

