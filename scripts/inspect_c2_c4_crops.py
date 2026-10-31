from PIL import Image
import numpy as np

c2 = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png")
crop_box = (256, 50, 768, 562)
crop_rgba = c2.crop(crop_box)
crop_rgb = crop_rgba.convert("RGB")

print("Crop size:", crop_rgb.size)
arr = np.array(crop_rgba)
print("RGB mean in non-zero alpha:", arr[arr[:, :, 3] > 0, :3].mean(axis=0))

c4 = Image.open("benchmark_output/img_mv3p042e_e20ae6_extracted.png")
crop_rgba4 = c4.crop(crop_box)
arr4 = np.array(crop_rgba4)
print("S4 RGB mean in non-zero alpha:", arr4[arr4[:, :, 3] > 0, :3].mean(axis=0))

