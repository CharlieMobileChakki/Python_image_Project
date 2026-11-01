"""Segmenter module: Wrappers for Flow A (BiRefNet Lite) and Flow B (SegFormer B2 Clothes)."""
from __future__ import annotations

import os
from typing import Any

os.environ["HF_HOME"] = r"D:\huggingface_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import cv2
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


def load_flow_a() -> None:
    """Preload Flow A (BiRefNet Lite) session."""
    global _flow_a_session
    if _flow_a_session is None:
        _flow_a_session = new_session(FLOW_A_MODEL_NAME)


def load_flow_b() -> None:
    """Preload Flow B (SegFormer B2) model."""
    global _flow_b_processor, _flow_b_model
    if _flow_b_model is None:
        try:
            _flow_b_processor = AutoImageProcessor.from_pretrained(FLOW_B_MODEL_ID, local_files_only=True)
            _flow_b_model = SegformerForSemanticSegmentation.from_pretrained(FLOW_B_MODEL_ID, local_files_only=True)
        except Exception:
            _flow_b_processor = AutoImageProcessor.from_pretrained(FLOW_B_MODEL_ID)
            _flow_b_model = SegformerForSemanticSegmentation.from_pretrained(FLOW_B_MODEL_ID)
        _flow_b_model.eval()


def load_segmenter_models() -> None:
    """Preload both segmentation models into memory."""
    load_flow_a()
    load_flow_b()


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
        load_flow_a()
    # rembg.remove returns transparent RGBA PIL Image
    return remove(image_rgb, session=_flow_a_session)


def run_segformer_inference(image_rgb: Image.Image) -> tuple[np.ndarray, torch.Tensor]:
    """Execute Flow B SegFormer forward pass.

    Returns:
        tuple(pred_map_2d_uint8, prob_tensor_18_classes_float32)
    """
    global _flow_b_processor, _flow_b_model
    if _flow_b_model is None or _flow_b_processor is None:
        load_flow_b()

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


def extract_garment_cutout_guided(
    image_rgb: Image.Image,
    probs_tensor: torch.Tensor,
    target_class_ids: list[int],
    foreground_cutout: Image.Image | None = None,
    solid_threshold: float = 0.40,
    edge_threshold: float = 0.15,
    min_component_area: int = 150,
) -> Image.Image:
    """Confidence-aware mask fusion with foreground boundary recovery.

    Preserves complete multi-region garments (such as Saree pallu + pleats) and utilizes
    BiRefNet Lite foreground boundaries for delicate edge retention (borders, pleats, drapes).

    Args:
        image_rgb: Original PIL image (RGB).
        probs_tensor: Probability tensor of shape (18, H, W).
        target_class_ids: List of class IDs to union (e.g. [4, 7] for Saree).
        foreground_cutout: Optional BiRefNet Lite RGBA cutout for edge snapping.
        solid_threshold: Soft alpha solid threshold.
        edge_threshold: Soft alpha transition floor.
        min_component_area: Minimum pixel area to keep connected components.

    Returns:
        Transparent RGBA PIL Image of extracted garment ensemble.
    """
    # 1. Sum probabilities of target classes across the ensemble
    target_probs = sum(probs_tensor[cid].numpy() for cid in target_class_ids)
    bg_prob = probs_tensor[0].numpy()

    # Competing non-garment classes (shoes, face/neck, bare legs, arms/hands, opposing wardrobe items)
    body_accessory_classes = [1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    opposing_classes = [c for c in body_accessory_classes if c not in target_class_ids]
    if 6 in target_class_ids and 4 not in target_class_ids:
        opposing_classes.append(4)
    elif 4 in target_class_ids and 6 not in target_class_ids:
        opposing_classes.append(6)

    # 2. Compute soft alpha mask
    alpha_float = np.zeros_like(target_probs, dtype=np.float32)
    solid_mask = target_probs >= solid_threshold
    alpha_float[solid_mask] = 1.0
    trans_mask = (target_probs >= edge_threshold) & ~solid_mask
    if trans_mask.any():
        denom = max(1e-5, solid_threshold - edge_threshold)
        alpha_float[trans_mask] = (target_probs[trans_mask] - edge_threshold) / denom

    alpha_uint8 = np.clip(alpha_float * 255.0, 0, 255).astype(np.uint8)

    # Suppress pixels where background confidence is dominant (eliminates background leak, grid lines, and white borders)
    alpha_uint8[bg_prob >= target_probs] = 0

    # Suppress pixels where a competing non-garment / body class has higher confidence
    if opposing_classes:
        opp_probs = np.maximum.reduce([probs_tensor[c].numpy() for c in opposing_classes])
        alpha_uint8[opp_probs > target_probs] = 0

    # 3. Foreground Boundary Recovery (using BiRefNet Lite when provided)
    if foreground_cutout is not None:
        fg_alpha = np.array(foreground_cutout.split()[3])
        # Check human presence
        is_human = (probs_tensor[11].numpy() > 0.25).any() or (probs_tensor[2].numpy() > 0.25).any()
        if not is_human:
            # On mannequins and isolated products, BiRefNet captures the full garment drape (neckline, embroidery, straps, saree)
            if (alpha_uint8 > 20).sum() > 200:
                fg_bin = (fg_alpha > 20).astype(np.uint8)
                n_labels, labels, _, _ = cv2.connectedComponentsWithStats(fg_bin)
                garment_touch = np.unique(labels[alpha_uint8 > 20])
                keep_fg = np.isin(labels, [lbl for lbl in garment_touch if lbl > 0])
                if keep_fg.any():
                    refined_alpha = np.where(keep_fg, fg_alpha, 0).astype(np.uint8)
                else:
                    refined_alpha = fg_alpha
            else:
                refined_alpha = fg_alpha
        else:
            fabric_threshold = min(edge_threshold, 0.15)
            refined_alpha = np.where(
                target_probs >= fabric_threshold,
                np.minimum(alpha_uint8, fg_alpha),
                0,
            ).astype(np.uint8)
        alpha_uint8 = refined_alpha

    # 4. Filter out disconnected noisy specks and non-garment props (tripod stands, plants, hangers)
    if min_component_area > 0 and (alpha_uint8 > 20).sum() > min_component_area:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (alpha_uint8 > 20).astype(np.uint8)
        )
        if num_labels > 2:
            largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            main_mask = (labels == largest_idx)
            dilated_main = cv2.dilate(main_mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
            clean_alpha = np.zeros_like(alpha_uint8)
            for i in range(1, num_labels):
                comp_mask = (labels == i)
                # Keep main garment body and directly connected/adjacent garment elements (sleeves, pleats)
                if (i == largest_idx) or (comp_mask[dilated_main].any() and stats[i, cv2.CC_STAT_AREA] >= min_component_area):
                    clean_alpha[comp_mask] = alpha_uint8[comp_mask]
            alpha_uint8 = clean_alpha
        elif num_labels == 2:
            alpha_uint8 = np.where(labels == 1, alpha_uint8, 0).astype(np.uint8)

    # 5. Clean alpha edge smoothing without aggressive erosion (preserves fine straps and embroidery)
    if is_human:
        opened = cv2.morphologyEx(alpha_uint8, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        alpha_im = Image.fromarray(opened, mode="L").filter(ImageFilter.GaussianBlur(radius=0.7))
        smoothed_np = np.array(alpha_im)
        final_alpha = np.where((opened > 0) & (smoothed_np >= 18), smoothed_np, 0).astype(np.uint8)
    else:
        # For product / flat-lay, preserve original fine subpixel anti-aliasing from BiRefNet Lite
        alpha_im = Image.fromarray(alpha_uint8, mode="L").filter(ImageFilter.GaussianBlur(radius=0.5))
        smoothed_np = np.array(alpha_im)
        final_alpha = np.where((alpha_uint8 > 0) & (smoothed_np >= 10), smoothed_np, 0).astype(np.uint8)

    # 6. Strict RGB decontamination: zero out RGB where alpha is 0
    rgba_np = np.array(image_rgb.convert("RGBA"))
    rgba_np[:, :, :3] = np.where(final_alpha[:, :, None] == 0, 0, rgba_np[:, :, :3])
    rgba_np[:, :, 3] = final_alpha
    return Image.fromarray(rgba_np, mode="RGBA")


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
    return extract_garment_cutout_guided(
        image_rgb=image_rgb,
        probs_tensor=probs_tensor,
        target_class_ids=target_class_ids,
        foreground_cutout=None,
        solid_threshold=solid_threshold,
        edge_threshold=edge_threshold,
        min_component_area=0,
    )

