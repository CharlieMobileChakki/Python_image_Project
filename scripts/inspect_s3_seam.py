from PIL import Image
import numpy as np

cmap3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/class_map.png").convert("RGB")
c3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/upper-clothes_cutout.png").convert("RGBA")
crop3_cmap = np.array(cmap3.crop((250, 350, 762, 862)))
crop3_c = np.array(c3.crop((250, 350, 762, 862)))

upper = (crop3_cmap[:, :, 0] == 0) & (crop3_cmap[:, :, 1] == 0) & (crop3_cmap[:, :, 2] == 128)
right_arm = (crop3_cmap[:, :, 0] == 192) & (crop3_cmap[:, :, 1] == 128) & (crop3_cmap[:, :, 2] == 128)

# Check upper clothes min and max X for y in 200..400
for y in range(220, 390, 20):
    u_xs = np.where(upper[y, :])[0]
    ra_xs = np.where(right_arm[y, :])[0]
    min_u = u_xs.min() if len(u_xs) else None
    max_ra = ra_xs.max() if len(ra_xs) else None
    print(f"y={y}: upper min_X={min_u}, arm max_X={max_ra}")

