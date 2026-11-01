"""Segmenter module: Wrappers for Flow A (BiRefNet Lite) and Flow B (SegFormer B2 Clothes)."""
from __future__ import annotations

import os
from typing import Any

os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
from PIL import Image, ImageFilter
from rembg import new_session, remove
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

FLOW_A_MODEL_NAME = os.getenv("REMBG_MODEL", "birefnet-general-lite").strip()
FLOW_B_MODEL_ID = "mattmdjaga/segformer_b2_clothes"

_flow_a_session: Any | None = None
_flow_b_processor: AutoImageProcessor | None = None
_flow_b_model: SegformerForSemanticSegmentation | None = None


def load_segmenter_models() -> None:
    """Preload both segmentation models into memory."""
    global _flow_a_session, _flow_b_processor, _flow_b_model
    if _flow_a_session is None:
        _flow_a_session = new_session(FLOW_A_MODEL_NAME)
    if _flow_b_model is None:
        _flow_b_processor = AutoImageProcessor.from_pretrained(FLOW_B_MODEL_ID)
        _flow_b_model = SegformerForSemanticSegmentation.from_pretrained(FLOW_B_MODEL_ID)
        _flow_b_model.eval()


def are_models_loaded() -> dict[str, bool]:
    """Check readiness status of segmentation models."""
    return {
        "flow_a_birefnet": _flow_a_session is not None,
        "flow_b_segformer": _flow_b_model is not None,
    }


def extract_isolated_flow_a(image_rgb: Image.Image) -> Image.Image:
    """Flow A: Fast salient background removal for flat-lay, hanger, or mannequin photos."""
    global _flow_a_session
    if _flow_a_session is None:
        _flow_a_session = new_session(FLOW_A_MODEL_NAME)
    # rembg.remove returns transparent RGBA PIL Image
    return remove(image_rgb, session=_flow_a_session)


def run_segformer_inference(image_rgb: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
    """Execute Flow B SegFormer forward pass.

    Returns:
        tuple(pred_map_2d_uint8, prob_tensor_18_classes_float32)
    """
    global _flow_b_processor, _flow_b_model
    if _flow_b_model is None or _flow_b_processor is None:
        load_segmenter_models()

    orig_w, orig_h = image_rgb.size
    inputs = _flow_b_processor(images=image_rgb, return_tensors="pt")

    with torch.no_grad():
        outputs = _flow_b_model(**inputs)
        logits = outputs.logits
        # Upsample logits to original image resolution
        upsampled = F.interpolate(logits, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
        probs = F.softmax(upsampled, dim=1).squeeze(0)
        pred_map = torch.argmax(probs, dim=0).cpu().numpy().astype(np.uint8)

    return pred_map, probs.cpu()


def extract_garment_cutout_flow_b(
    image_rgb: Image.Image,
    probs_tensor: torch.Tensor,
    target_class_ids: list[int],
    solid_threshold: float = 0.50,
    edge_threshold: float = 0.15,
) -> Image.Image:
    """Extract clean garment cutout from SegFormer probabilities with confidence-based soft alpha.

    Args:
        image_rgb: Original PIL image (RGB).
        probs_tensor: Probability tensor of shape (18, H, W).
        target_class_ids: List of class IDs to union (e.g. [4] for Upper, [6] for Pants).
        solid_threshold: Soft alpha solid threshold (1.0 alpha).
        edge_threshold: Soft alpha transition floor.

    Returns:
        Transparent RGBA PIL Image of extracted garment item.
    """
    # Sum probabilities of target classes
    target_probs = sum(probs_tensor[cid].numpy() for cid in target_class_ids)

    # Compute soft alpha mask
    alpha_float = np.zeros_like(target_probs, dtype=np.float32)
    # 1. Solid core
    solid_mask = target_probs >= solid_threshold
    alpha_float[solid_mask] = 1.0
    # 2. Smooth transition zone
    trans_mask = (target_probs >= edge_threshold) & ~solid_mask
    if trans_mask.any():
        denom = max(1e-5, solid_threshold - edge_threshold)
        alpha_float[trans_mask] = (target_probs[trans_mask] - edge_threshold) / denom

    alpha_uint8 = np.clip(alpha_float * 255.0, 0, 255).astype(np.uint8)

    # Apply 1px subtle gaussian smoothing on edges to prevent stair-stepping
    alpha_im = Image.fromarray(alpha_uint8, mode="L").filter(ImageFilter.GaussianBlur(radius=0.7))

    # Combine with original RGB
    rgba = image_rgb.convert("RGBA")
    rgba.putalpha(alpha_im)
    return rgba

