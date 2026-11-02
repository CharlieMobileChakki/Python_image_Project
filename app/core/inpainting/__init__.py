"""Inpainting subpackage: LaMa Fast Fourier Convolutions and occlusion mask repair."""
from app.core.inpainting.inpainting_utils import (
    ensure_mask_dimension,
    ensure_patch_dimension,
)
from app.core.inpainting.lama_inpainter import (
    inpaint_patch_512,
    repair_garment_cutout,
)
from app.core.inpainting.mask_repair import (
    compute_arm_occlusion_mask,
)

__all__ = [
    "compute_arm_occlusion_mask",
    "ensure_mask_dimension",
    "ensure_patch_dimension",
    "inpaint_patch_512",
    "repair_garment_cutout",
]

