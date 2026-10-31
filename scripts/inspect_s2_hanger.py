from PIL import Image
import numpy as np

c2 = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png")
c2_np = np.array(c2)
# Find where alpha > 0
alpha = c2_np[:, :, 3]
y_idxs, x_idxs = np.where(alpha > 0)
print(f"Sample 2 cutout bounding box: y=[{y_idxs.min()}, {y_idxs.max()}], x=[{x_idxs.min()}, {x_idxs.max()}]")

# Let's check the top region of the cutout (e.g. y from y_idxs.min() to y_idxs.min() + 300)
top_region = c2_np[y_idxs.min():y_idxs.min()+250, x_idxs.min():x_idxs.max()]
# Look for non-white/grey pixels in top_region
# White shirt is high RGB (e.g. > 180, 180, 180)
# Hanger hook or wood might be brown or metallic
print("Top region shape:", top_region.shape)
# Sample points near top center:
center_x = (x_idxs.min() + x_idxs.max()) // 2
for y in range(y_idxs.min(), y_idxs.min() + 200, 20):
    px = c2_np[y, center_x]
    print(f"y={y}, center_x={center_x}: RGBA={px.tolist()}")

