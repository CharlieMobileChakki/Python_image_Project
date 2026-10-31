from PIL import Image
import numpy as np

# Inspect Sample 2: White T-Shirt
print("=== Analyzing Sample 2 ===")
im2 = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv50gr9c_87dbd3.jpg").convert("RGB")
c2 = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png").convert("RGBA")
print("Original size:", im2.size)
print("Cutout size:", c2.size)

# Inspect Sample 3: Dark Navy T-Shirt
print("\n=== Analyzing Sample 3 ===")
im3 = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv53zgfo_ffc202.jpg").convert("RGB")
c3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/upper-clothes_cutout.png").convert("RGBA")
cmap3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/class_map.png").convert("RGB")
cmap3_arr = np.array(cmap3)
unique_colors = np.unique(cmap3_arr.reshape(-1, 3), axis=0)
print("Original size:", im3.size)
print("Unique colors in class map:", unique_colors.tolist())

# Inspect Sample 4: Turquoise Anarkali
print("\n=== Analyzing Sample 4 ===")
im4 = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv3p042e_e20ae6.webp").convert("RGB")
c4 = Image.open("benchmark_output/img_mv3p042e_e20ae6_extracted.png").convert("RGBA")
print("Original size:", im4.size)
print("Cutout size:", c4.size)

