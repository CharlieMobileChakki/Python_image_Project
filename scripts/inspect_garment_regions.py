from PIL import Image
import numpy as np

# Sample 2: Plain White T-Shirt on hanger
# Original image: (375, 500)
# Let's inspect the wooden hanger at the neck opening
im2 = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv50gr9c_87dbd3.jpg").convert("RGB")
c2 = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png").convert("RGBA")

print(f"Sample 2 orig size: {im2.size}, cutout size: {c2.size}")

# Let's inspect Sample 3: Dark Navy T-shirt
im3 = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv53zgfo_ffc202.jpg").convert("RGB")
c3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/upper-clothes_cutout.png").convert("RGBA")
m3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/upper-clothes_mask.png").convert("L")
cmap3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/class_map.png").convert("RGB")

print(f"Sample 3 orig size: {im3.size}, cutout size: {c3.size}, cmap size: {cmap3.size}")

# Let's inspect Sample 4: Turquoise Embroidered Anarkali
im4 = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv3p042e_e20ae6.webp").convert("RGB")
c4 = Image.open("benchmark_output/img_mv3p042e_e20ae6_extracted.png").convert("RGBA")

print(f"Sample 4 orig size: {im4.size}, cutout size: {c4.size}")

