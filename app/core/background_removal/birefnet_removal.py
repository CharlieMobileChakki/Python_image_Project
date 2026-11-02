"""BiRefNet Lite salient background removal (Flow A)."""
from __future__ import annotations

from PIL import Image
from rembg import remove

from app.core.model_manager.model_loader import get_flow_a_session


def extract_isolated_flow_a(image_rgb: Image.Image) -> Image.Image:
    """Flow A: Fast salient background removal for flat-lay, hanger, or mannequin photos."""
    session = get_flow_a_session()
    # rembg.remove returns transparent RGBA PIL Image
    return remove(image_rgb, session=session)

