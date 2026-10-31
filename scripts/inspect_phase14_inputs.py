from pathlib import Path
from PIL import Image
import numpy as np

# Sample 1: Trousers
s1_orig = Path(r"D:\my codes\Arangtik\backend\uploads\img_mv3nprm1_b7901a.jpg")
s1_cutout = Path("benchmark_output/phase10/case1_trousers_pants/soft_cutout.png")
s1_cmap = Path("benchmark_output/phase10/case1_trousers_pants/class_map.png")
print("Sample 1:")
print(f"  Orig: {s1_orig.exists()}, Cutout: {s1_cutout.exists()}, CMap: {s1_cmap.exists()}")

# Sample 2: White T-Shirt
s2_orig = Path(r"D:\my codes\Arangtik\backend\uploads\img_mv50gr9c_87dbd3.jpg")
s2_cutout_a = Path("benchmark_output/img_mv50gr9c_87dbd3_extracted.png")
print("\nSample 2:")
print(f"  Orig: {s2_orig.exists()}, Cutout A: {s2_cutout_a.exists()}")
im2 = Image.open(s2_orig)
print(f"  Orig size: {im2.size}")
if s2_cutout_a.exists():
    im2_c = Image.open(s2_cutout_a)
    print(f"  Cutout size: {im2_c.size}")

# Sample 3: Navy T-shirt
s3_orig = Path(r"D:\my codes\Arangtik\backend\uploads\img_mv53zgfo_ffc202.jpg")
s3_dir = Path("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes")
s3_cutout = s3_dir / "upper-clothes_cutout.png"
s3_cmap = s3_dir / "class_map.png"
print("\nSample 3:")
print(f"  Orig: {s3_orig.exists()}, Cutout: {s3_cutout.exists()}, CMap: {s3_cmap.exists()}")

# Sample 4: Turquoise Anarkali
s4_orig = Path(r"D:\my codes\Arangtik\backend\uploads\img_mv3p042e_e20ae6.webp")
s4_cutout_a = Path("benchmark_output/img_mv3p042e_e20ae6_extracted.png")
print("\nSample 4:")
print(f"  Orig: {s4_orig.exists()}, Cutout A: {s4_cutout_a.exists()}")

