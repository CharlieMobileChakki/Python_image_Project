"""Garment semantic classification, physical continuity inspection, and outfit structuring."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np

from app.core.human_detection import (
    HUMAN_BODY_CLASSES,
    detect_human_presence,
)
from app.core.garment_extraction.coverage_rules import (
    GARMENT_PRIORITY_THRESHOLDS,
)
from app.core.garment_extraction.garment_aliases import (
    GARMENT_CLASSES,
)


def analyze_semantic_map(pred_map: np.ndarray) -> dict[str, Any]:
    """Analyze SegFormer prediction map to determine human presence, scene type, and garments.

    Args:
        pred_map: 2D numpy array of class IDs (0..17), matching image dimensions.

    Returns:
        dict containing has_human, scene_type ('person'|'mannequin'|'flat_lay'),
        human_pixel_count, detected_garments list.
    """
    total_pixels = pred_map.size
    unique_classes, counts = np.unique(pred_map, return_counts=True)
    class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))

    has_human, scene_type, metrics = detect_human_presence(pred_map)
    human_pixels = int(metrics["human_pixels"])
    human_pct = metrics["human_pct"]

    # Detect garments
    detected_garments = []
    for cid, info in GARMENT_CLASSES.items():
        cnt = class_counts.get(cid, 0)
        pct = (cnt / total_pixels) * 100.0
        # Valid garment if it covers at least 0.6% of image and > 500 pixels (or > 1500 px)
        if (cnt > 500 and pct > 0.6) or cnt > 1500:
            y_idxs, x_idxs = np.where(pred_map == cid)
            bbox = [int(x_idxs.min()), int(y_idxs.min()), int(x_idxs.max()), int(y_idxs.max())]
            detected_garments.append({
                "class_id": cid,
                "category": info["key"],
                "label": info["label"],
                "pixel_count": int(cnt),
                "coverage_pct": round(pct, 2),
                "bounding_box": bbox,
            })

    # Include scarf/dupatta if substantial
    scarf_cnt = class_counts.get(17, 0)
    if scarf_cnt > 1200:
        y_idxs, x_idxs = np.where(pred_map == 17)
        bbox = [int(x_idxs.min()), int(y_idxs.min()), int(x_idxs.max()), int(y_idxs.max())]
        detected_garments.append({
            "class_id": 17,
            "category": "scarf",
            "label": "Scarf / Dupatta / Pallu",
            "pixel_count": int(scarf_cnt),
            "coverage_pct": round((scarf_cnt / total_pixels) * 100.0, 2),
            "bounding_box": bbox,
        })

    return {
        "has_human": has_human,
        "scene_type": scene_type,
        "human_pixels": human_pixels,
        "human_coverage_pct": round(human_pct, 2),
        "detected_garments": detected_garments,
    }


def check_garment_physical_continuity(
    pred_map: np.ndarray,
    upper_cid: int = 4,
    lower_cids: tuple[int, ...] = (5, 7),
    bare_skin_cids: tuple[int, ...] = (11, 12, 13),
) -> dict[str, Any]:
    """Analyze physical connectivity, vertical overlap, waist boundary contact,
    and silhouette continuity between upper and lower garment components.

    Universal for both men and women: evaluates physical geometry and boundaries, not gender.
    """
    upper_mask = (pred_map == upper_cid)
    lower_mask = np.isin(pred_map, lower_cids)

    if upper_mask.sum() == 0 or lower_mask.sum() == 0:
        return {
            "is_continuous": False,
            "overlap_y": 0,
            "overlap_ratio": 0.0,
            "contact_pixels": 0,
            "has_bare_midriff": False,
            "total_span_ratio": 0.0,
        }

    H, W = pred_map.shape
    u_ys, u_xs = np.where(upper_mask)
    l_ys, l_xs = np.where(lower_mask)

    u_ymin, u_ymax = int(u_ys.min()), int(u_ys.max())
    l_ymin, l_ymax = int(l_ys.min()), int(l_ys.max())

    overlap_y = max(0, min(u_ymax, l_ymax) - max(u_ymin, l_ymin))
    min_h = max(1, min(u_ymax - u_ymin, l_ymax - l_ymin))
    overlap_ratio = overlap_y / min_h

    # Direct contact measurement using small 5x5 dilation
    kernel = np.ones((5, 5), np.uint8)
    dilated_u = cv2.dilate(upper_mask.astype(np.uint8), kernel) > 0
    contact_pixels = int((dilated_u & lower_mask).sum())

    # Total combined vertical extent
    combined_ymin = min(u_ymin, l_ymin)
    combined_ymax = max(u_ymax, l_ymax)
    total_span_ratio = (combined_ymax - combined_ymin) / max(1, H)

    # Bare midriff check: skin pixels between upper bottom and lower top
    has_bare_midriff = False
    if l_ymin > u_ymax:
        gap_mask = np.zeros_like(pred_map, dtype=bool)
        gap_mask[u_ymax:l_ymin, :] = True
        skin_in_gap = (np.isin(pred_map, bare_skin_cids) & gap_mask).sum()
        if skin_in_gap > 300:
            has_bare_midriff = True

    # Physical continuity criteria:
    # 1. Substantial vertical overlap (layered kurti, overlay bodice, saree drape): overlap_y > 40 px or overlap_ratio > 0.15
    # 2. Direct waist boundary contact: contact_pixels > 200 without bare midriff
    # 3. Full-length continuous silhouette: total span > 50% of image height
    is_continuous = False
    if not has_bare_midriff:
        if overlap_y > 40 or overlap_ratio > 0.15:
            is_continuous = True
        elif contact_pixels > 200 and total_span_ratio > 0.50:
            is_continuous = True

    return {
        "is_continuous": is_continuous,
        "overlap_y": overlap_y,
        "overlap_ratio": round(overlap_ratio, 3),
        "contact_pixels": contact_pixels,
        "has_bare_midriff": has_bare_midriff,
        "total_span_ratio": round(total_span_ratio, 3),
    }


def classify_outfit_structure(
    pred_map: np.ndarray,
    probs_tensor: Any | None,
    detected_garments: list[dict[str, Any]],
    scene_type: str = "person",
    target_filter: str = "all",
) -> dict[str, Any]:
    """Intelligently determine outfit structure: Independent garments vs Single-piece vs Coordinated traditional ensemble.

    Outfit splitting is strictly based on physical outfit structure, NOT gender.

    Modes:
    - Mode A (Multiple Independent Garments): e.g. T-Shirt + Jeans, Shirt + Pants.
    - Mode B (Single-Piece Garments): e.g. One-piece dress, Gown, Anarkali.
    - Mode C (Coordinated Traditional Outfits): e.g. Saree, Lehenga, Ethnic suit.
    """
    target_clean = target_filter.strip().lower()
    cids_present = {g["class_id"] for g in detected_garments}
    total_map_pixels = pred_map.size

    # Check 1: User explicitly requested Saree or Traditional Ensemble
    if target_clean in {"saree", "sari"}:
        fuse_classes = [cid for cid in [4, 7, 5, 17] if (pred_map == cid).sum() > 400]
        if not fuse_classes:
            fuse_classes = [4, 7]
        return {
            "outfit_mode": "mode_c_traditional_ensemble",
            "ensemble_type": "saree",
            "planned_items": [{
                "key": "saree",
                "label": "Saree / Traditional Ensemble",
                "target_class_ids": fuse_classes,
                "is_fused": True,
                "coverage_percentage": 100.0,
                "priority": "first_priority",
                "status": "active",
            }],
        }

    if target_clean in {"lehenga", "lehenga_choli"}:
        fuse_classes = [cid for cid in [4, 5, 17, 7] if (pred_map == cid).sum() > 400]
        if not fuse_classes:
            fuse_classes = [4, 5]
        return {
            "outfit_mode": "mode_c_traditional_ensemble",
            "ensemble_type": "lehenga",
            "planned_items": [{
                "key": "lehenga",
                "label": "Lehenga / Ethnic Ensemble",
                "target_class_ids": fuse_classes,
                "is_fused": True,
                "coverage_percentage": 100.0,
                "priority": "first_priority",
                "status": "active",
            }],
        }

    if target_clean in {"dress", "gown", "anarkali"}:
        fuse_classes = [7]
        if 4 in cids_present and (pred_map == 4).sum() > 800:
            fuse_classes.append(4)
        return {
            "outfit_mode": "mode_b_single_piece",
            "ensemble_type": "dress",
            "planned_items": [{
                "key": "dress",
                "label": "Dress / Gown / Anarkali",
                "target_class_ids": fuse_classes,
                "is_fused": len(fuse_classes) > 1,
                "coverage_percentage": 100.0,
                "priority": "first_priority",
                "status": "active",
            }],
        }

    # Presence flags across all detected classes
    has_upper = 4 in cids_present
    has_dress = 7 in cids_present
    has_skirt = 5 in cids_present
    has_pants = 6 in cids_present
    has_scarf = 17 in cids_present or (pred_map == 17).sum() > 500

    # Check 1B: Kurta + Pajama, Shirt + Pants, T-Shirt + Jeans (Mode A Independent Garments)
    # If pants/pajama are genuinely present alongside an upper garment/kurta, treat as independent pieces
    pants_px = int((pred_map == 6).sum())
    upper_px = int((pred_map == 4).sum())
    has_real_pants = has_pants and (pants_px > 0.15 * max(1, upper_px) or (scene_type == "person" and pants_px > 600))

    if has_real_pants and (has_upper or has_dress):
        upper_cids = [4]
        if has_dress and (pred_map == 7).sum() > 400:
            upper_cids.append(7)  # Long kurta / tunic spanning bodice and hem
        if has_scarf:
            upper_cids.append(17)

        total_combo_px = max(1, upper_px + pants_px)
        upper_cov = round((upper_px / total_combo_px) * 100.0, 2)
        pants_cov = round((pants_px / total_combo_px) * 100.0, 2)

        def _calc_prio(cov: float) -> str:
            if cov < GARMENT_PRIORITY_THRESHOLDS["low_priority"]:
                return "ignore_below"
            if cov < GARMENT_PRIORITY_THRESHOLDS["second_priority"]:
                return "low_priority"
            if cov < GARMENT_PRIORITY_THRESHOLDS["first_priority"]:
                return "second_priority"
            return "first_priority"

        planned = []
        if target_clean in {"all", "upper_clothes", "shirt", "top", "kurta"}:
            planned.append({
                "key": "upper_clothes",
                "label": "T-Shirt / Top / Shirt" if "t-shirt" in target_clean or "shirt" in target_clean else "Kurta / Shirt / Top",
                "target_class_ids": upper_cids,
                "is_fused": len(upper_cids) > 1,
                "coverage_percentage": upper_cov,
                "priority": _calc_prio(upper_cov),
                "status": "active",
            })
        if target_clean in {"all", "pants", "trousers", "pajama", "jeans"}:
            planned.append({
                "key": "pants",
                "label": "Pants / Pajama / Trousers",
                "target_class_ids": [6],
                "is_fused": False,
                "coverage_percentage": pants_cov,
                "priority": _calc_prio(pants_cov),
                "status": "active",
            })
        if not planned:
            planned = [{
                "key": "upper_clothes",
                "label": "Kurta / Shirt / Top",
                "target_class_ids": upper_cids,
                "is_fused": len(upper_cids) > 1,
                "coverage_percentage": upper_cov,
                "priority": _calc_prio(upper_cov),
                "status": "active",
            }]

        return {
            "outfit_mode": "mode_a_independent",
            "ensemble_type": "independent_wardrobe",
            "planned_items": planned,
        }

    # Check 2: Automatic Saree Detection (Mode C)
    # SegFormer splits a Saree into Upper-clothes (4, chest/pallu) and Dress (7, pleats/body).
    # In reality, they are ONE physical garment.

    # Determine whether pants are primary bifurcated trousers (e.g. Jeans/Pants starting at the waist)
    # vs secondary churidar/leggings peeking at the ankles underneath a full-length dress/kurta
    is_dominant_pants = False
    if has_pants:
        pants_g = next((g for g in detected_garments if g["class_id"] == 6), None)
        if pants_g:
            pants_bbox = pants_g.get("bounding_box")
            if not pants_bbox and (pred_map == 6).any():
                ys_p, xs_p = np.where(pred_map == 6)
                pants_bbox = [int(xs_p.min()), int(ys_p.min()), int(xs_p.max()), int(ys_p.max())]
            if pants_bbox:
                if pants_bbox[1] < 0.62 * pred_map.shape[0] and (pants_bbox[3] - pants_bbox[1]) > 0.25 * pred_map.shape[0]:
                    is_dominant_pants = True
            else:
                is_dominant_pants = True

    if has_upper and has_dress and not is_dominant_pants:
        # Saree or Full Continuous Dress/Anarkali
        fuse_classes = [4, 7]
        if has_skirt and (pred_map == 5).sum() > 800:
            fuse_classes.append(5)
        if has_scarf:
            fuse_classes.append(17)

        # Classify as Saree if traditional drape/scarf or mannequin presentation
        if has_scarf or scene_type == "mannequin":
            ensemble_type = "saree"
            label = "Saree / Traditional Ensemble"
            key = "saree"
            mode = "mode_c_traditional_ensemble"
        else:
            ensemble_type = "dress"
            label = "Dress / Gown / Anarkali"
            key = "dress"
            mode = "mode_b_single_piece"

        return {
            "outfit_mode": mode,
            "ensemble_type": ensemble_type,
            "planned_items": [{
                "key": key,
                "label": label,
                "target_class_ids": fuse_classes,
                "is_fused": True,
                "coverage_percentage": 100.0,
                "priority": "first_priority",
                "status": "active",
            }],
        }

    # Check 2B: Mannequin full-length drape or product display (e.g. Saree, Sherwani, Gown)
    if not is_dominant_pants and (scene_type in {"mannequin", "flat_lay"} or (has_upper and not has_pants and scene_type == "mannequin")):
        fuse_classes = [cid for cid in [4, 7, 5, 17] if (pred_map == cid).sum() > 80]
        if not fuse_classes:
            fuse_classes = [4]

        is_trad = (has_scarf and (pred_map == 17).sum() > 1200)
        has_dress_strong = (pred_map == 7).sum() > 1500
        has_upper_strong = (pred_map == 4).sum() > 1500

        scarf_pixels = int((pred_map == 17).sum())
        upper_pixels = int((pred_map == 4).sum())
        is_substantial_scarf = (scarf_pixels > 2500 and scarf_pixels > 0.25 * max(1, upper_pixels))

        if (is_trad and is_substantial_scarf) or (has_dress_strong and (has_upper_strong or is_substantial_scarf)):
            # Saree or Traditional Ensemble with long lower drape or large pallu
            if has_dress_strong and not is_substantial_scarf:
                mode = "mode_b_single_piece"
                ensemble_type = "dress"
                label = "Dress / Gown / Anarkali"
                key = "dress"
            else:
                mode = "mode_c_traditional_ensemble"
                ensemble_type = "saree"
                label = "Saree / Traditional Ensemble"
                key = "saree"
        elif has_dress_strong:
            mode = "mode_b_single_piece"
            ensemble_type = "dress"
            label = "Dress / Gown / Anarkali"
            key = "dress"
        else:
            mode = "mode_a_independent" if has_pants else "mode_b_single_piece"
            ensemble_type = "independent_wardrobe" if has_pants else "shirt"
            label = "Kurta / Shirt / Top"
            key = "upper_clothes"

        return {
            "outfit_mode": mode,
            "ensemble_type": ensemble_type,
            "planned_items": [{
                "key": key,
                "label": label,
                "target_class_ids": fuse_classes,
                "is_fused": len(fuse_classes) > 1,
                "coverage_percentage": 100.0,
                "priority": "first_priority",
                "status": "active",
            }],
        }

    # Check 3: Upper-clothes + Skirt Analysis (Mode B Single-Piece vs Mode C Traditional vs Mode A Independent)
    if has_upper and has_skirt and not is_dominant_pants:
        cont = check_garment_physical_continuity(pred_map, upper_cid=4, lower_cids=(5,))
        if cont["is_continuous"] or has_scarf or scene_type == "mannequin":
            fuse_classes = [4, 5]
            if has_scarf:
                fuse_classes.append(17)

            if has_scarf:
                ensemble_type = "lehenga"
                label = "Lehenga / Ethnic Ensemble"
                key = "lehenga"
                mode = "mode_c_traditional_ensemble"
            else:
                ensemble_type = "dress"
                label = "Dress / Two-Piece Gown"
                key = "dress"
                mode = "mode_b_single_piece"

            return {
                "outfit_mode": mode,
                "ensemble_type": ensemble_type,
                "planned_items": [{
                    "key": key,
                    "label": label,
                    "target_class_ids": fuse_classes,
                    "is_fused": True,
                    "coverage_percentage": 100.0,
                    "priority": "first_priority",
                    "status": "active",
                }],
            }

    # Check 4: Single-Piece Dress / Gown (Mode B)
    if has_dress and not has_pants:
        fuse_classes = [7]
        if has_upper:
            fuse_classes.append(4)
        return {
            "outfit_mode": "mode_b_single_piece",
            "ensemble_type": "dress",
            "planned_items": [{
                "key": "dress",
                "label": "Dress / Gown / Anarkali",
                "target_class_ids": fuse_classes,
                "is_fused": len(fuse_classes) > 1,
                "coverage_percentage": 100.0,
                "priority": "first_priority",
                "status": "active",
            }],
        }

    # Check 5: Multiple Independent Garments (Mode A)
    # E.g. Shirt + Pants, T-Shirt + Jeans, Top + Skirt
    # Calculate union of selected garment masks for percentage denominator
    active_garment_cids = [g["class_id"] for g in detected_garments if g["class_id"] in GARMENT_CLASSES]
    if not active_garment_cids:
        active_garment_cids = [4]

    union_mask = np.isin(pred_map, active_garment_cids)
    total_selected_pixels = max(1, int(union_mask.sum()))

    planned_items = []
    # If user selected a specific target (e.g. "pants"), filter to that
    items_to_evaluate = detected_garments
    if target_clean != "all":
        filtered = [g for g in detected_garments if target_clean in g["category"] or target_clean in g["label"].lower()]
        if filtered:
            items_to_evaluate = filtered

    for g in items_to_evaluate:
        cid = g["class_id"]
        # Skip scarf from independent list if clothes present
        if cid == 17 and len(items_to_evaluate) > 1:
            continue

        item_pixels = g["pixel_count"]
        # Exact formula: (individual_pixels / total_unique_selected_pixels) * 100
        coverage_pct = round((item_pixels / total_selected_pixels) * 100.0, 2)

        # Priority rules
        if coverage_pct < GARMENT_PRIORITY_THRESHOLDS["ignore_below"]:
            priority = "ignore_below"
            status = "ignored"
        elif coverage_pct < GARMENT_PRIORITY_THRESHOLDS["second_priority"]:
            priority = "low_priority"
            status = "active"
        elif coverage_pct < GARMENT_PRIORITY_THRESHOLDS["first_priority"]:
            priority = "second_priority"
            status = "active"
        else:
            priority = "first_priority"
            status = "active"

        planned_items.append({
            "key": g["category"],
            "label": g["label"],
            "target_class_ids": [cid],
            "is_fused": False,
            "coverage_percentage": coverage_pct,
            "priority": priority,
            "status": status,
        })

    return {
        "outfit_mode": "mode_a_independent",
        "ensemble_type": "independent_wardrobe",
        "planned_items": planned_items,
    }
