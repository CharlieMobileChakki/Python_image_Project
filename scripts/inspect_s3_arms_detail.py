from PIL import Image
import numpy as np

cmap3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/class_map.png")
crop_box = (250, 350, 762, 862)
crop_cmap = np.array(cmap3.crop(crop_box))

# SegFormer classes:
# Upper: [0, 0, 128]
# Right-arm: [192, 128, 128]
# Left-arm: [64, 128, 128]
upper = (crop_cmap[:, :, 0] == 0) & (crop_cmap[:, :, 1] == 0) & (crop_cmap[:, :, 2] == 128)
right_arm = (crop_cmap[:, :, 0] == 192) & (crop_cmap[:, :, 1] == 128) & (crop_cmap[:, :, 2] == 128)
left_arm = (crop_cmap[:, :, 0] == 64) & (crop_cmap[:, :, 1] == 128) & (crop_cmap[:, :, 2] == 128)

print("Upper pixels:", upper.sum())
print("Right arm pixels in crop:", right_arm.sum())
print("Left arm pixels in crop:", left_arm.sum())

# Where are the arms?
y_ra, x_ra = np.where(right_arm)
print(f"Right arm in crop: Y=[{y_ra.min()}, {y_ra.max()}], X=[{x_ra.min()}, {x_ra.max()}]")

y_la, x_la = np.where(left_arm)
print(f"Left arm in crop: Y=[{y_la.min()}, {y_la.max()}], X=[{x_la.min()}, {x_la.max()}]")

# Where is upper clothes in crop?
y_u, x_u = np.where(upper)
print(f"Upper in crop: Y=[{y_u.min()}, {y_u.max()}], X=[{x_u.min()}, {x_u.max()}]")

