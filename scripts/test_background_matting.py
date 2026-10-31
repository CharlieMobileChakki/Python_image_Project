from PIL import Image
import numpy as np

# Sample 2:
c2 = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png").convert("RGBA")
crop2 = c2.crop((256, 50, 768, 562))
arr2 = np.array(crop2)

# Notice arr2[:, :, :3] where alpha == 0
bg_pixels = arr2[:, :, 3] == 0
print("Sample 2 RGB in transparent pixels before convert:", arr2[bg_pixels, :3].mean(axis=0))

# Notice what PIL convert('RGB') does:
rgb_naive = crop2.convert("RGB")
print("Naive convert('RGB') in transparent pixels:", np.array(rgb_naive)[bg_pixels].mean(axis=0))

# Composite on white background:
bg_white = Image.new("RGBA", crop2.size, (255, 255, 255, 255))
rgb_white = Image.alpha_composite(bg_white, crop2).convert("RGB")
print("Composite on white in transparent pixels:", np.array(rgb_white)[bg_pixels].mean(axis=0))

# Or composite on fabric color (mean of garment):
garment_rgb = arr2[arr2[:, :, 3] > 200, :3].mean(axis=0).astype(int)
print("Garment mean RGB:", garment_rgb)
bg_fabric = Image.new("RGBA", crop2.size, tuple(garment_rgb) + (255,))
rgb_fabric = Image.alpha_composite(bg_fabric, crop2).convert("RGB")
print("Composite on fabric color:", np.array(rgb_fabric)[bg_pixels].mean(axis=0))

