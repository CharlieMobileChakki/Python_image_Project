from PIL import Image
import numpy as np

cmap3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/class_map.png")
cmap_np = np.array(cmap3)
upper_mask = (cmap_np[:, :, 0] == 0) & (cmap_np[:, :, 1] == 0) & (cmap_np[:, :, 2] == 128)
left_arm = (cmap_np[:, :, 0] == 64) & (cmap_np[:, :, 1] == 128) & (cmap_np[:, :, 2] == 128)
right_arm = (cmap_np[:, :, 0] == 192) & (cmap_np[:, :, 1] == 128) & (cmap_np[:, :, 2] == 128)

print("Upper clothes pixels:", upper_mask.sum())
print("Left arm pixels:", left_arm.sum())
print("Right arm pixels:", right_arm.sum())

# Let's inspect where arms border the upper clothes:
# Let's check Y range 550 to 740 (waist / hem / wrist area):
for y in range(600, 740, 20):
    u_xs = np.where(upper_mask[y, :])[0]
    la_xs = np.where(left_arm[y, :])[0]
    ra_xs = np.where(right_arm[y, :])[0]
    u_str = f"[{u_xs.min()}, {u_xs.max()}]" if len(u_xs) else "none"
    la_str = f"[{la_xs.min()}, {la_xs.max()}]" if len(la_xs) else "none"
    ra_str = f"[{ra_xs.min()}, {ra_xs.max()}]" if len(ra_xs) else "none"
    print(f"y={y}: Upper={u_str}, LeftArm={la_str}, RightArm={ra_str}")

