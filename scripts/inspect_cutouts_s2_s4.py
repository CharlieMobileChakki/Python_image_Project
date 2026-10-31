from PIL import Image
import numpy as np

# Let's inspect Sample 2 cutout:
c2 = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png")
c2_arr = np.array(c2)
print("Sample 2 cutout shape:", c2_arr.shape)
# Find alpha bounding box
alpha2 = c2_arr[:, :, 3]
y_idxs, x_idxs = np.where(alpha2 > 0)
print(f"Alpha > 0 bounding box: x=[{x_idxs.min()}, {x_idxs.max()}], y=[{y_idxs.min()}, {y_idxs.max()}]")
print(f"Non-zero alpha pixel count: {(alpha2 > 0).sum()}")

# Let's inspect Sample 4 cutout:
c4 = Image.open("benchmark_output/img_mv3p042e_e20ae6_extracted.png")
c4_arr = np.array(c4)
print("Sample 4 cutout shape:", c4_arr.shape)
alpha4 = c4_arr[:, :, 3]
y_idxs, x_idxs = np.where(alpha4 > 0)
print(f"Alpha > 0 bounding box: x=[{x_idxs.min()}, {x_idxs.max()}], y=[{y_idxs.min()}, {y_idxs.max()}]")
print(f"Non-zero alpha pixel count: {(alpha4 > 0).sum()}")

