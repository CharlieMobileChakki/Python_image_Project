from PIL import Image
import numpy as np

c3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/upper-clothes_cutout.png")
cmap3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/class_map.png")
orig3 = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv53zgfo_ffc202.jpg")

cmap_np = np.array(cmap3)
# 4: Upper-clothes = [0, 0, 128]
upper_mask = (cmap_np[:, :, 0] == 0) & (cmap_np[:, :, 1] == 0) & (cmap_np[:, :, 2] == 128)
# Arms: Left-arm [64, 128, 128], Right-arm [192, 128, 128]
arm_mask = ((cmap_np[:, :, 0] == 64) & (cmap_np[:, :, 1] == 128) & (cmap_np[:, :, 2] == 128)) | \
           ((cmap_np[:, :, 0] == 192) & (cmap_np[:, :, 1] == 128) & (cmap_np[:, :, 2] == 128))

y_u, x_u = np.where(upper_mask)
print(f"Upper clothes Y: [{y_u.min()}, {y_u.max()}], X: [{x_u.min()}, {x_u.max()}]")
print(f"Upper clothes total pixels: {upper_mask.sum()}")

# Let's inspect where arms touch or cut into upper clothes
# Look at model's arms around upper clothes
y_a, x_a = np.where(arm_mask)
print(f"Arms Y: [{y_a.min()}, {y_a.max()}], X: [{x_a.min()}, {x_a.max()}]")

# Let's check what the cutout alpha is:
c3_np = np.array(c3)
alpha = c3_np[:, :, 3]
y_c, x_c = np.where(alpha > 0)
print(f"Cutout alpha > 0 Y: [{y_c.min()}, {y_c.max()}], X: [{x_c.min()}, {x_c.max()}]")

# Sample RGB of navy shirt:
shirt_rgb = np.array(orig3)[upper_mask].mean(axis=0)
print(f"Mean navy shirt RGB: {shirt_rgb}")

