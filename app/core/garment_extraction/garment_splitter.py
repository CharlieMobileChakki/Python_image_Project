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
        alpha = np.ones((H, W), dtype=np.uint8) * 255
    else:
        alpha = fg_alpha.copy()

    # 1. Background color suppression for studio / catalog photos:
    # Sample corner pixels to detect uniform studio background (white, off-white, light gray)
    corners = np.array([
        arr[0, 0], arr[0, -1],
        arr[-1, 0], arr[-1, -1],
    ], dtype=np.float32)
    bg_color = np.median(corners, axis=0)
    bg_std = float(np.std(corners, axis=0).mean())

    cleaned_alpha = alpha.copy()

    # Only apply edge floodfill if the canvas corners represent a consistent studio background
    if bg_std < 30.0:
        color_diff = np.linalg.norm(arr.astype(np.float32) - bg_color, axis=2)
        # Cap max tolerance strictly to 32.0 so garment fabric is never matched
        is_bg_color = (color_diff < min(32.0, max(18.0, bg_std * 2.0)))

        flood_map = np.zeros((H + 2, W + 2), dtype=np.uint8)
        bg_seed_mask = is_bg_color.astype(np.uint8) * 255

        # Seed floodfill from outer canvas perimeter
        for y in [0, H - 1]:
            for x in range(0, W, max(1, W // 10)):
                if is_bg_color[y, x] and bg_seed_mask[y, x] == 255:
                    cv2.floodFill(bg_seed_mask, flood_map, (x, y), 128, flags=4 | (255 << 8))

        for x in [0, W - 1]:
            for y in range(0, H, max(1, H // 10)):
                if is_bg_color[y, x] and bg_seed_mask[y, x] == 255:
                    cv2.floodFill(bg_seed_mask, flood_map, (x, y), 128, flags=4 | (255 << 8))

        connected_bg = (bg_seed_mask == 128)
        cleaned_alpha[connected_bg] = 0

    # 2. Binary threshold
    bin_mask = (cleaned_alpha > 35).astype(np.uint8)

    # Gentle opening (kernel 3x3) to sever narrow noise bridges between distinct garments
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel_open)

    # Connected components analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    # Adaptive minimum area: 0.4% of total canvas or 200 pixels
    # Allows individual items in 8-12 item collection grids while discarding noise specks
    min_area = max(200, int(0.004 * total_px))
    min_w = max(12, int(0.02 * W))
    min_h = max(12, int(0.02 * H))

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
    # does NOT dominate >= 88% of total foreground area, or >= 3 garments.
    is_multi = False
    if len(significant_comps) >= 2 and total_fg_area > 0:
        largest_area = max(c["area"] for c in significant_comps)
        largest_share = largest_area / total_fg_area
        if len(significant_comps) >= 3 or largest_share < 0.88:
            is_multi = True
            row_height = max(30, int(H / 5))
            significant_comps.sort(key=lambda c: (c["y"] // row_height, c["x"]))

    return is_multi, significant_comps
