"""Compatibility shim for inpainter module. Re-exports from inpainting and model_manager packages."""
from __future__ import annotations

import sys
from PIL import Image
import numpy as np

from app.core.inpainting import (
    compute_arm_occlusion_mask,
    ensure_mask_dimension,
    ensure_patch_dimension,
    inpaint_patch_512,
    repair_garment_cutout as _core_repair_garment_cutout,
)
from app.core.model_manager import (
    LAMA_MODEL_PATH,
    get_inpainter_session,
    is_inpainter_loaded,
    load_inpainter_model,
)


def repair_garment_cutout(
    cutout_rgba: Image.Image,
    inpaint_mask: np.ndarray,
    crop_box: tuple[int, int, int, int] | None = None,
    *args,
    **kwargs,
) -> tuple[Image.Image, float]:
    """Compatibility wrapper delegating to inpainting package while supporting mock patches on app.core.inpainter.inpaint_patch_512."""
    this_mod = sys.modules[__name__]
    patch_fn = getattr(this_mod, "inpaint_patch_512", inpaint_patch_512)
    return _core_repair_garment_cutout(
        cutout_rgba,
        inpaint_mask,
        crop_box=crop_box,
        inpaint_fn=patch_fn,
        *args,
        **kwargs,
    )


__all__ = [
    "LAMA_MODEL_PATH",
    "load_inpainter_model",
    "get_inpainter_session",
    "is_inpainter_loaded",
    "ensure_patch_dimension",
    "ensure_mask_dimension",
    "inpaint_patch_512",
    "compute_arm_occlusion_mask",
    "repair_garment_cutout",
]
