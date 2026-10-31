from PIL import Image, ImageFilter
import numpy as np

# Sample 1: Trousers
def get_sample1_mask(class_crop):
    arm = (class_crop[:, :, 0] == 192) & (class_crop[:, :, 1] == 128) & (class_crop[:, :, 2] == 128)
    notch_mask = np.zeros((512, 512), dtype=np.uint8)
    notch_alpha = np.zeros((512, 512), dtype=np.uint8)
    y_start, y_end = 112, 260
    seam_x_top, seam_x_bottom = 185, 188
    for y in range(y_start, y_end + 1):
        frac = (y - y_start) / max(1, (y_end - y_start))
        x_seam = int(np.round(seam_x_top + frac * (seam_x_bottom - seam_x_top)))
        arm_xs = np.where(arm[y, :])[0]
        if len(arm_xs) > 0:
            x_min_arm = arm_xs.min()
            x_max_arm = arm_xs.max()
            x_start = max(x_seam, x_min_arm)
            x_end = x_max_arm + 3
            if x_end > x_start:
                notch_mask[y, x_start : x_end + 1] = 255
                notch_alpha[y, x_start] = 140
                if x_start + 1 <= x_end:
                    notch_alpha[y, x_start + 1 : x_end + 1] = 255
    return notch_mask, notch_alpha

# Sample 2: Plain White T-Shirt
def get_sample2_mask(crop_np):
    r, g, b, a = crop_np[:, :, 0], crop_np[:, :, 1], crop_np[:, :, 2], crop_np[:, :, 3]
    # Brown wood pixels inside crop
    is_wood = (a > 50) & (r > 130) & (g > 80) & (b < 120) & (r > b + 25)
    # Dilate slightly (3px) to ensure no brown fringe remains
    mask_im = Image.fromarray((is_wood.astype(np.uint8) * 255), mode="L")
    dilated_mask = np.array(mask_im.filter(ImageFilter.MaxFilter(5)))
    # Inpaint mask is where wood was
    inpaint_mask = (dilated_mask > 0).astype(np.uint8) * 255
    # For alpha: inner collar fabric should be opaque (255)
    inpaint_alpha = inpaint_mask.copy()
    return inpaint_mask, inpaint_alpha

# Sample 3: Dark Navy T-Shirt
def get_sample3_mask(cmap_crop, crop_np):
    # Upper-clothes [0, 0, 128], Right-arm [192, 128, 128]
    upper = (cmap_crop[:, :, 0] == 0) & (cmap_crop[:, :, 1] == 0) & (cmap_crop[:, :, 2] == 128)
    right_arm = (cmap_crop[:, :, 0] == 192) & (cmap_crop[:, :, 1] == 128) & (cmap_crop[:, :, 2] == 128)
    # Flank notch occurs along the seam where right arm occludes side seam
    # In crop (250, 350, 762, 862):
    # Right arm is around x in [140, 210], y in [240, 390] in crop coords
    notch_mask = np.zeros((512, 512), dtype=np.uint8)
    notch_alpha = np.zeros((512, 512), dtype=np.uint8)
    y_start, y_end = 240, 380
    # Natural vertical seam of navy shirt
    seam_x = 195
    for y in range(y_start, y_end + 1):
        arm_xs = np.where(right_arm[y, :])[0]
        if len(arm_xs) > 0:
            x_min_arm = arm_xs.min()
            x_max_arm = arm_xs.max()
            x_start = max(seam_x, x_min_arm)
            x_end = x_max_arm + 2
            if x_end > x_start:
                notch_mask[y, x_start : x_end + 1] = 255
                notch_alpha[y, x_start] = 140
                if x_start + 1 <= x_end:
                    notch_alpha[y, x_start + 1 : x_end + 1] = 255
    return notch_mask, notch_alpha

# Sample 4: Turquoise Embroidered Anarkali
def get_sample4_mask(crop_np):
    # Neck stand block: top center where y in [45, 140] and x in [210, 305]
    # Non-turquoise stand: r > 100, g < 150, or greyish
    r, g, b, a = crop_np[:, :, 0], crop_np[:, :, 1], crop_np[:, :, 2], crop_np[:, :, 3]
    is_stand = (a > 50) & (r > 70) & (g < 160) & (b < 160)
    # Restrict to neck box
    y_grid, x_grid = np.ogrid[:512, :512]
    stand_region = is_stand & (y_grid >= 45) & (y_grid <= 140) & (x_grid >= 210) & (x_grid <= 305)
    mask_im = Image.fromarray((stand_region.astype(np.uint8) * 255), mode="L")
    dilated_mask = np.array(mask_im.filter(ImageFilter.MaxFilter(5)))
    inpaint_mask = (dilated_mask > 0).astype(np.uint8) * 255
    inpaint_alpha = inpaint_mask.copy()
    return inpaint_mask, inpaint_alpha

print("All mask generators defined successfully.")

