"""Flow A: single-garment background removal and transparent PNG preparation."""
from __future__ import annotations

from io import BytesIO
import os
from typing import Any

from PIL import Image, ImageOps
from rembg import new_session, remove

from app.core.normalizer import detect_rotation_angle, normalize_to_1024, straighten_garment

MODEL_NAME = os.getenv("REMBG_MODEL", "birefnet-general-lite").strip()
OUTPUT_SIZE = int(os.getenv("OUTPUT_SIZE", "1024"))
MAX_INPUT_EDGE = int(os.getenv("MAX_INPUT_EDGE", "2400"))
CANVAS_FILL_RATIO = float(os.getenv("CANVAS_FILL_RATIO", "0.82"))

_session: Any | None = None


def load_model() -> None:
    """Load model once per API worker. The model weights download on first use."""
    global _session
    if _session is None:
        _session = new_session(MODEL_NAME)


def _prepare_input(image_bytes: bytes) -> tuple[bytes, tuple[int, int]]:
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

    output = BytesIO()
    image.save(output, format="PNG", optimize=True)
    return output.getvalue(), original_size


def _normalize_canvas(cutout_bytes: bytes) -> bytes:
    """Tight-crop the alpha subject, auto-straighten tilt in backend, and center on 1024x1024 canvas."""
    try:
        cutout = Image.open(BytesIO(cutout_bytes)).convert("RGBA")
    except OSError as exc:
        raise RuntimeError("The segmentation model returned an unreadable image.") from exc

    # Auto-deskew tilt in backend
    angle = detect_rotation_angle(cutout)
    if abs(angle) > 0.5:
        cutout = straighten_garment(cutout, angle)

    norm_1024 = normalize_to_1024(cutout, target_size=OUTPUT_SIZE, fill_ratio=CANVAS_FILL_RATIO)
    output = BytesIO()
    norm_1024.save(output, format="PNG", optimize=True)
    return output.getvalue()


def extract_garment(image_bytes: bytes) -> tuple[bytes, tuple[int, int]]:
    """Return transparent, centered PNG bytes and the original image dimensions."""
    if _session is None:
        load_model()

    prepared_bytes, original_size = _prepare_input(image_bytes)
    try:
        raw_cutout = remove(prepared_bytes, session=_session)
    except Exception as exc:  # expose a safe API error; details go to server logs
        raise RuntimeError(f"Model processing failed: {type(exc).__name__}") from exc
    return _normalize_canvas(raw_cutout), original_size
