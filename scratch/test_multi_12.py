import cv2
from PIL import Image
import numpy as np
import sys
sys.path.insert(0, ".")
from app.core.detector import detect_multi_garment_layout
from app.core.validator import validate_garment_background
from app.core.normalizer import normalize_to_1024

img = Image.open("input images/istockphoto_850788940_612x612.png").convert("RGB")
arr_rgb = np.array(img)
H, W, _ = arr_rgb.shape

is_multi, multi_comps = detect_multi_garment_layout(img)
print(f"is_multi: {is_multi}, garments detected: {len(multi_comps)}")

success_count = 0
for idx, comp in enumerate(multi_comps):
    c_mask = comp["comp_mask"]
    c_soft = cv2.GaussianBlur(c_mask.astype(np.float32) * 255.0, (3, 3), 0.5)
    g_alpha = np.clip(c_soft, 0, 255).astype(np.uint8)

    x0, y0, x1, y1 = comp["bbox"]
    pad = 8
    cx0 = max(0, x0 - pad)
    cy0 = max(0, y0 - pad)
    cx1 = min(W, x1 + pad)
    cy1 = min(H, y1 + pad)

    sub_rgb = arr_rgb[cy0:cy1, cx0:cx1]
    sub_a = g_alpha[cy0:cy1, cx0:cx1]
    sub_rgba = Image.fromarray(np.dstack([sub_rgb, sub_a]), mode="RGBA")

    norm = normalize_to_1024(sub_rgba, target_size=1024, fill_ratio=0.82)
    val = validate_garment_background(norm, original_rgb=img)
    status = val["status"]
    conf = val["confidence"]
    errs = val["validation_errors"]
    if status == "success":
        success_count += 1
    print(f"Garment {idx+1:02d}: Cat={comp['category']}, Status={status}, Conf={conf}, Errors={errs}")

print(f"\nTotal Success: {success_count} / {len(multi_comps)}")

