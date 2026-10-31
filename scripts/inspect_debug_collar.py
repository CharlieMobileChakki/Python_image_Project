from PIL import Image
import numpy as np

c2 = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png")
c2_arr = np.array(c2)
# Crop around the collar: x in [256, 768], y in [50, 562]
collar = c2.crop((256, 50, 768, 562))
collar.save("benchmark_output/phase14/debug_s2_collar.png")
print("Saved debug collar image to benchmark_output/phase14/debug_s2_collar.png")

# Let's inspect what is in collar:
# Where is the wooden hanger, and where is the neckline?
r, g, b, a = np.array(collar)[:, :, 0], np.array(collar)[:, :, 1], np.array(collar)[:, :, 2], np.array(collar)[:, :, 3]
is_wood = (a > 50) & (r > 130) & (g > 80) & (b < 120) & (r > b + 25)
is_fabric = (a > 50) & (r > 180) & (g > 180) & (b > 180)

print(f"Collar shape: {collar.size}")
print(f"Wood pixels: {is_wood.sum()}")
print(f"Fabric pixels: {is_fabric.sum()}")

# Print Y range of wood vs fabric
yw, xw = np.where(is_wood)
yf, xf = np.where(is_fabric)
print(f"Wood Y range: [{yw.min()}, {yw.max()}], X range: [{xw.min()}, {xw.max()}]")
print(f"Fabric Y range: [{yf.min()}, {yf.max()}], X range: [{xf.min()}, {xf.max()}]")

