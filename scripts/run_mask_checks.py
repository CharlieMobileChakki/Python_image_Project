import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from PIL import Image
import numpy as np
from scripts.test_masks import get_sample1_mask, get_sample2_mask, get_sample3_mask, get_sample4_mask

# Test Sample 1
cmap1 = Image.open("benchmark_output/phase10/case1_trousers_pants/class_map.png")
class_crop1 = np.array(cmap1.crop((650, 0, 1162, 512)))
m1, a1 = get_sample1_mask(class_crop1)
print(f"Sample 1 mask pixels: {(m1 == 255).sum()}")

# Test Sample 2
c2 = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png").convert("RGBA")
crop2 = np.array(c2.crop((256, 50, 768, 562)))
m2, a2 = get_sample2_mask(crop2)
print(f"Sample 2 mask pixels: {(m2 == 255).sum()}")

# Test Sample 3
c3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/upper-clothes_cutout.png").convert("RGBA")
cmap3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/class_map.png").convert("RGB")
crop3 = np.array(c3.crop((250, 350, 762, 862)))
cmap_crop3 = np.array(cmap3.crop((250, 350, 762, 862)))
m3, a3 = get_sample3_mask(cmap_crop3, crop3)
print(f"Sample 3 mask pixels: {(m3 == 255).sum()}")

# Test Sample 4
c4 = Image.open("benchmark_output/img_mv3p042e_e20ae6_extracted.png").convert("RGBA")
crop4 = np.array(c4.crop((256, 50, 768, 562)))
m4, a4 = get_sample4_mask(crop4)
print(f"Sample 4 mask pixels: {(m4 == 255).sum()}")
