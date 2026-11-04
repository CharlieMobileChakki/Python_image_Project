"""Multi-garment layout detection and spatial component splitting for catalog grids."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np
from PIL import Image


def detect_multi_garment_layout(
    image_rgb: Image.Image,
    fg_alpha: np.ndarray | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Detect if an image contains multiple independently arranged garments on flat-lay or catalog grid.

    Relies strictly on true salient foreground alpha (BiRefNet Lite) rather than fragile
    outer border color metrics. Prevents textured carpets, complex studio walls, shadows,
    and mannequins from being erroneously classified as separate garments.

    Args:
        image_rgb: PIL RGB image.
        fg_alpha: Optional uint8 alpha mask from salient foreground extraction.

    Returns:
        (is_multi_garment, list of detected garment components with masks and bounding boxes)
    """
    arr = np.array(image_rgb)
    H, W, _ = arr.shape
    total_px = H * W

    # Use salient foreground alpha from BiRefNet Lite
    if fg_alpha is None:
        bin_mask = np.ones((H, W), dtype=np.uint8)
    else:
        # Solid foreground threshold
        bin_mask = (fg_alpha > 45).astype(np.uint8)

    # Bridge delicate embroidery, lace, folds, or internal pattern gaps
    kernel_size = max(5, int(min(W, H) * 0.02))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)

    # A legitimate independent catalog garment occupies significant canvas area
    # Minimum 4% of total canvas or 1000 pixels, with realistic physical span
    min_area = max(1000, int(0.04 * total_px))
    min_w = int(0.08 * W)
    min_h = int(0.08 * H)

    significant_comps: list[dict[str, Any]] = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        # Filter out entire canvas border
        if w > 0.98 * W and h > 0.98 * H:
            continue
        # Filter out 1-2px border edge artifacts
        if (x <= 1 and w <= 2) or (y <= 1 and h <= 2) or (x + w >= W - 1 and w <= 2) or (y + h >= H - 1 and h <= 2):
            continue

        # Must meet area and dimension thresholds
        if area >= min_area and (w >= min_w or h >= min_h):
            aspect = h / max(1, w)
            if aspect > 1.40:
                cat = "dress"
                label = "Dress / Gown / Full Outfit"
            elif aspect > 1.15 and y > H * 0.40:
                cat = "pants"
                label = "Pants / Lower Garment"
            elif aspect < 0.80:
                cat = "upper_clothes"
                label = "Top / Shirt / Upper Garment"
            else:
                cat = "upper_clothes"
                label = "Garment Cutout"

            # Clean component mask (labels == i)
            comp_mask = (labels == i)

            significant_comps.append({
                "label": i,
                "area": int(area),
                "bbox": [x, y, x + w, y + h],
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "category": cat,
                "label_name": label,
                "aspect_ratio": round(aspect, 2),
                "centroid": centroids[i].tolist(),
                "comp_mask": comp_mask,
            })

    total_fg_area = sum(c["area"] for c in significant_comps)

    # Multi-garment condition:
    # Requires at least 2 distinct significant garments where the largest item
    # does NOT dominate >= 85% of total foreground area.
    is_multi = False
    if len(significant_comps) >= 2 and total_fg_area > 0:
        largest_area = max(c["area"] for c in significant_comps)
        largest_share = largest_area / total_fg_area
        if largest_share < 0.85:
            is_multi = True
            row_height = max(30, int(H / 5))
            significant_comps.sort(key=lambda c: (c["y"] // row_height, c["x"]))

    return is_multi, significant_comps
