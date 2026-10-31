import json
from pathlib import Path
from PIL import Image
import numpy as np

for sid in ["sample1_trousers_beige", "sample2_tshirt_white_plain", "sample3_tshirt_navy_dark", "sample4_anarkali_turquoise_embroidered"]:
    p = Path(f"benchmark_output/phase14/{sid}")
    m = json.load(open(p / "metrics.json"))
    print(f"\n=== {sid} ===")
    print("Border fabric mean RGB:", m["color_metrics"]["border_fabric_mean_rgb"])
    print("Inpaint patch mean RGB:", m["color_metrics"]["inpaint_patch_mean_rgb"])
    print("Mean border delta:", m["color_metrics"]["mean_border_delta_rgb"])
    
    # Inspect images
    orig = Image.open(p / "original.png")
    before = Image.open(p / "before_cutout.png")
    after = Image.open(p / "after_cutout.png")
    mask = Image.open(p / "inpaint_mask.png")
    mask_np = np.array(mask) == 255
    
    print("Orig RGB in mask:", np.array(orig)[mask_np].mean(axis=0))
    print("After RGB in mask:", np.array(after)[mask_np, :3].mean(axis=0))
    print("After alpha in mask:", np.array(after)[mask_np, 3].mean())
    print("After alpha outside mask (where before alpha == 0):", np.array(after)[(np.array(before)[:,:,3] == 0) & (~mask_np), 3].max())

