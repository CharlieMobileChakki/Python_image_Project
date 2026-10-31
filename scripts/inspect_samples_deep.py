from pathlib import Path
from PIL import Image
import numpy as np

# Sample 1: Trousers
print("=== Sample 1 ===")
s1_orig = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv3nprm1_b7901a.jpg")
s1_cutout = Image.open("benchmark_output/phase10/case1_trousers_pants/soft_cutout.png")
print(f"S1 orig: {s1_orig.size}, cutout: {s1_cutout.size}")

# Sample 2: White T-Shirt
print("\n=== Sample 2 ===")
s2_orig = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv50gr9c_87dbd3.jpg")
s2_cutout = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png")
print(f"S2 orig: {s2_orig.size}, cutout: {s2_cutout.size}")

# Sample 3: Navy T-Shirt
print("\n=== Sample 3 ===")
s3_orig = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv53zgfo_ffc202.jpg")
s3_cutout = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/upper-clothes_cutout.png")
s3_cmap = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/class_map.png")
print(f"S3 orig: {s3_orig.size}, cutout: {s3_cutout.size}, cmap: {s3_cmap.size}")

# Sample 4: Anarkali
print("\n=== Sample 4 ===")
s4_orig = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv3p042e_e20ae6.webp")
s4_cutout = Image.open("benchmark_output/img_mv3p042e_e20ae6_extracted.png")
print(f"S4 orig: {s4_orig.size}, cutout: {s4_cutout.size}")

