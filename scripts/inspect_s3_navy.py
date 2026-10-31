from PIL import Image
import numpy as np

# Sample 3: Navy t-shirt
# Upper-clothes cutout and class map in benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes
c3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/upper-clothes_cutout.png")
cmap3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/class_map.png")
orig3 = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv53zgfo_ffc202.jpg")

cmap_np = np.array(cmap3)
# Look at SegFormer classes:
# Upper-clothes: [0, 128, 0]
# Pants: [0, 128, 128]
# Left Arm / Right Arm: [192, 128, 128]
# Left Leg / Right Leg: [0, 0, 128]
# Face/Neck: [192, 0, 0]
upper_mask = (cmap_np[:, :, 0] == 0) & (cmap_np[:, :, 1] == 128) & (cmap_np[:, :, 2] == 0)
arm_mask = (cmap_np[:, :, 0] == 192) & (cmap_np[:, :, 1] == 128) & (cmap_np[:, :, 2] == 128)
pants_mask = (cmap_np[:, :, 0] == 0) & (cmap_np[:, :, 1] == 128) & (cmap_np[:, :, 2] == 128)

y_u, x_u = np.where(upper_mask)
print(f"Upper clothes Y: [{y_u.min()}, {y_u.max()}], X: [{x_u.min()}, {x_u.max()}]")

# Where does arm touch or overlap upper_clothes?
from scipy.ndimage import binary_dilation
dilated_arm = binary_dilation(arm_mask, iterations=3)
overlap = dilated_arm & upper_mask
print(f"Overlap/contact pixels between arm and upper clothes: {overlap.sum()}")
y_ov, x_ov = np.where(overlap)
if len(y_ov) > 0:
    print(f"Contact Y: [{y_ov.min()}, {y_ov.max()}], X: [{x_ov.min()}, {x_ov.max()}]")

# Check where arm occludes or cuts into the upper clothes hem or sleeve
# For instance, the model's hand/arm near the hem or waist

