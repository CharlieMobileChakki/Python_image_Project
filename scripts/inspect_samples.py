from pathlib import Path
from PIL import Image
import numpy as np

cut_w = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png")
# Let's inspect rows Y in [150, 350], X in [400, 624]
arr = np.array(cut_w)
for y in range(150, 320, 20):
    row_rgb = arr[y, 450:574, :3]
    row_a = arr[y, 450:574, 3]
    wood = (row_rgb[:, 0] > 140) & (row_rgb[:, 1] < 170) & (row_rgb[:, 2] < 140) & (row_a > 100)
    white = (row_rgb[:, 0] > 200) & (row_rgb[:, 1] > 200) & (row_rgb[:, 2] > 200) & (row_a > 100)
    print(f"Y={y}: wood_px={wood.sum()}, white_px={white.sum()}, trans_px={(row_a == 0).sum()}")
