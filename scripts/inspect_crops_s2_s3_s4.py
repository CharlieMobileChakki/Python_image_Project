from PIL import Image
import numpy as np

# Sample 2: Plain White T-Shirt
# c2 shape: (1024, 1024)
# Collar center is around x=512, y=180
# 512x512 crop box: x from 256 to 768, y from 0 to 512 (or 50 to 562)
c2 = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png").convert("RGBA")
crop_box_s2 = (256, 50, 768, 562)
crop_s2 = c2.crop(crop_box_s2)
crop_s2_np = np.array(crop_s2)

# Wooden hanger inside collar:
r, g, b, a = crop_s2_np[:, :, 0], crop_s2_np[:, :, 1], crop_s2_np[:, :, 2], crop_s2_np[:, :, 3]
is_wood_s2 = (a > 50) & (r > 130) & (g > 80) & (b < 120) & (r > b + 25)
print("Sample 2 wood pixels in crop:", is_wood_s2.sum())
y_w, x_w = np.where(is_wood_s2)
if len(y_w) > 0:
    print(f"S2 Wood Y in crop: [{y_w.min()}, {y_w.max()}], X: [{x_w.min()}, {x_w.max()}]")

# Sample 3: Dark Navy T-Shirt
# im3 size: (1200, 1436)
# Right arm touches upper clothes at y in [600, 740], x around 430-460
# Upper clothes center around x=580, y=500
# Let's place a 512x512 crop around the right arm / flank notch:
# y from 400 to 912, x from 250 to 762
c3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/upper-clothes_cutout.png").convert("RGBA")
cmap3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/class_map.png").convert("RGB")
crop_box_s3 = (250, 350, 762, 862)
crop_s3 = c3.crop(crop_box_s3)
crop_cmap3 = cmap3.crop(crop_box_s3)
cmap_np = np.array(crop_cmap3)
upper = (cmap_np[:, :, 0] == 0) & (cmap_np[:, :, 1] == 0) & (cmap_np[:, :, 2] == 128)
right_arm = (cmap_np[:, :, 0] == 192) & (cmap_np[:, :, 1] == 128) & (cmap_np[:, :, 2] == 128)
print("Sample 3 crop upper pixels:", upper.sum(), "right arm pixels:", right_arm.sum())

# Sample 4: Turquoise Embroidered Anarkali
# c4 shape: (1024, 1024)
# Neck/stand defect is around x=512, y=90..190
# 512x512 crop box: x from 256 to 768, y from 50 to 562
c4 = Image.open("benchmark_output/img_mv3p042e_e20ae6_extracted.png").convert("RGBA")
crop_box_s4 = (256, 50, 768, 562)
crop_s4 = c4.crop(crop_box_s4)
crop_s4_np = np.array(crop_s4)
r4, g4, b4, a4 = crop_s4_np[:, :, 0], crop_s4_np[:, :, 1], crop_s4_np[:, :, 2], crop_s4_np[:, :, 3]
# Mannequin stand is brownish/greyish at top of neck:
# y from 40 to 140 within crop:
stand_candidate = (a4 > 100) & (crop_s4_np.shape[0] > 0)
# Let's inspect where y < 140 has non-transparent pixels
top_neck = (a4 > 50) & (np.arange(512)[:, None] < 140)
print("Sample 4 top neck non-zero alpha pixels in crop:", top_neck.sum())

