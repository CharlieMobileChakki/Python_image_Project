from PIL import Image
import numpy as np

c2 = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png")
c2_np = np.array(c2)
# Brown wooden hanger detection: R > 140, G > 80, B < 120, R > B + 30
r, g, b, a = c2_np[:, :, 0], c2_np[:, :, 1], c2_np[:, :, 2], c2_np[:, :, 3]
is_wood = (a > 50) & (r > 130) & (g > 80) & (b < 120) & (r > b + 30)
y_w, x_w = np.where(is_wood)
print(f"Wooden hanger pixels detected: {len(y_w)}")
if len(y_w) > 0:
    print(f"Hanger Y range: [{y_w.min()}, {y_w.max()}], X range: [{x_w.min()}, {x_w.max()}]")

# Let's inspect what is in the original image as well
orig2 = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv50gr9c_87dbd3.jpg")
print("Original size:", orig2.size)

