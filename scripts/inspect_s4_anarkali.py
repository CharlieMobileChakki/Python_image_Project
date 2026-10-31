from PIL import Image
import numpy as np

im4 = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv3p042e_e20ae6.webp").convert("RGB")
c4 = Image.open("benchmark_output/img_mv3p042e_e20ae6_extracted.png").convert("RGBA")

print(f"im4 size: {im4.size}, c4 size: {c4.size}")

# Check what the turquoise garment colors look like:
c4_arr = np.array(c4)
alpha4 = c4_arr[:, :, 3]
y_idxs, x_idxs = np.where(alpha4 > 0)
print(f"c4 alpha box: y=[{y_idxs.min()}, {y_idxs.max()}], x=[{x_idxs.min()}, {x_idxs.max()}]")

# Sample some RGB values of the garment:
center_x = (x_idxs.min() + x_idxs.max()) // 2
for y in range(y_idxs.min() + 100, y_idxs.max() - 100, 80):
    px = c4_arr[y, center_x]
    print(f"y={y}, center_x={center_x}: RGBA={px.tolist()}")

# What non-garment elements are in this image?
# The manifest says: "Turquoise anarkali suit on metallic mannequin in retail shop"
# Observed defect in Flow A: "Mannequin stand / metallic arm / hanger / shop background"
# Let's inspect the neck / stand / arms
for y in range(y_idxs.min(), y_idxs.min() + 100, 15):
    px = c4_arr[y, center_x]
    print(f"Neck/Stand y={y}, center_x={center_x}: RGBA={px.tolist()}")

