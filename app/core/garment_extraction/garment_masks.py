"""Confidence-aware garment mask fusion and foreground boundary snapping."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageFilter
import torch


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

    # Competing non-garment classes: separate true human body from accessories/wardrobe
    human_body_classes = [2, 11, 12, 13, 14, 15]  # hair, face, legs, arms
    accessory_classes = [1, 3, 8, 9, 10, 16]      # hat, glove, sunglasses, belt, socks, bag
    
    opp_body_classes = [c for c in human_body_classes if c not in target_class_ids]
    opp_acc_classes = [c for c in accessory_classes if c not in target_class_ids]
    if 6 in target_class_ids and 4 not in target_class_ids:
        opp_acc_classes.append(4)
    elif 4 in target_class_ids and 6 not in target_class_ids:
        opp_acc_classes.append(6)

    # Check human presence
    is_human = (probs_tensor[11].numpy() > 0.25).any() or (probs_tensor[2].numpy() > 0.25).any()

    # 2. Compute soft alpha mask
    alpha_float = np.zeros_like(target_probs, dtype=np.float32)
    solid_mask = target_probs >= solid_threshold
    alpha_float[solid_mask] = 1.0
    trans_mask = (target_probs >= edge_threshold) & ~solid_mask
    if trans_mask.any():
        denom = max(1e-5, solid_threshold - edge_threshold)
        alpha_float[trans_mask] = (target_probs[trans_mask] - edge_threshold) / denom

    alpha_uint8 = np.clip(alpha_float * 255.0, 0, 255).astype(np.uint8)

    # Check foreground presence from BiRefNet Lite
    has_fg = foreground_cutout is not None
    fg_alpha = np.array(foreground_cutout.split()[3]) if has_fg else None
    is_fg = (fg_alpha > 50) if has_fg else np.ones_like(alpha_uint8, dtype=bool)

    # Suppress background:
    # If BiRefNet confirmed solid foreground, trust it unless SegFormer background is overwhelming
    if has_fg:
        alpha_uint8[(bg_prob >= target_probs) & ~is_fg] = 0
    else:
        alpha_uint8[bg_prob >= target_probs] = 0

    # Rule B: Remove actual human body pixels (face, hair, arms, hands, legs)
    if opp_body_classes:
        body_probs = np.maximum.reduce([probs_tensor[c].numpy() for c in opp_body_classes])
        # Body removal requires higher confidence than weak fabric boundary jitter
        is_body = (body_probs > target_probs) & (body_probs >= 0.32)
        alpha_uint8[is_body] = 0

    # Suppress distinct non-garment accessories (sunglasses, bags, shoes)
    if opp_acc_classes:
        acc_probs = np.maximum.reduce([probs_tensor[c].numpy() for c in opp_acc_classes])
        alpha_uint8[(acc_probs > target_probs) & (acc_probs >= 0.40)] = 0

    # 3. Foreground Boundary Recovery (using BiRefNet Lite when provided)
    if has_fg:
        if not is_human:
            # On mannequins and isolated products, BiRefNet captures the full garment drape
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

    # Rule E: Internal Void Healing for continuous garment fabric
    # Eliminate unintended transparent voids enclosed by garment regions that lie within solid foreground
    alpha_bin = (alpha_uint8 > 20).astype(np.uint8)
    contours, hierarchy = cv2.findContours(alpha_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None and len(contours) > 1:
        for i in range(len(contours)):
            if hierarchy[0][i][3] >= 0:  # Enclosed internal void
                area = cv2.contourArea(contours[i])
                if area < 10000:
                    hole_mask = np.zeros_like(alpha_uint8)
                    cv2.drawContours(hole_mask, [contours[i]], -1, 255, -1)
                    if opp_body_classes:
                        body_probs = np.maximum.reduce([probs_tensor[c].numpy() for c in opp_body_classes])
                        safe_to_heal = (hole_mask == 255) & is_fg & (body_probs < 0.35)
                    else:
                        safe_to_heal = (hole_mask == 255) & is_fg
                    if safe_to_heal.any():
                        heal_val = fg_alpha if has_fg else 255
                        alpha_uint8 = np.where(safe_to_heal, heal_val, alpha_uint8).astype(np.uint8)

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

