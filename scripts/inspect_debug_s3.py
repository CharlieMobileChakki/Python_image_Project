from PIL import Image
import numpy as np

c3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/upper-clothes_cutout.png")
cmap3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/class_map.png")
orig3 = Image.open(r"D:\my codes\Arangtik\backend\uploads\img_mv53zgfo_ffc202.jpg")

crop_box = (250, 350, 762, 862)
crop_c3 = c3.crop(crop_box)
crop_orig3 = orig3.crop(crop_box)

crop_c3.save("benchmark_output/phase14/debug_s3_cutout.png")
crop_orig3.save("benchmark_output/phase14/debug_s3_orig.png")
print("Saved debug images for Sample 3 to benchmark_output/phase14/")

# Let's inspect what is in crop_c3 and crop_orig3
arr_c3 = np.array(crop_c3)
arr_orig3 = np.array(crop_orig3)
alpha = arr_c3[:, :, 3]
print("Cutout alpha > 0 pixels in crop:", (alpha > 0).sum())
print("Mean RGB of navy shirt in cutout:", arr_c3[alpha > 200, :3].mean(axis=0))

