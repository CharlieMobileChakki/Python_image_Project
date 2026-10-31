import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import onnxruntime as ort
import psutil

model_path = r"D:\huggingface_cache\hub\models--opencv--inpainting_lama\snapshots\aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f\inpainting_lama_2025jan.onnx"
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = psutil.cpu_count(logical=True)
sess = ort.InferenceSession(model_path, sess_opts, providers=["CPUExecutionProvider"])
print("LaMa ONNX session loaded.")

# Test Sample 2: White T-Shirt
# Crop box (256, 100, 768, 612) centered on chest / collar
c2 = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png").convert("RGBA")
crop2 = c2.crop((256, 100, 768, 612))
crop2_np = np.array(crop2)
# Create a realistic oval tag / occlusion mask on the front chest: center (256, 250), radius (35, 25)
mask2_im = Image.new("L", (512, 512), 0)
draw2 = ImageDraw.Draw(mask2_im)
draw2.ellipse([256 - 40, 250 - 30, 256 + 40, 250 + 30], fill=255)
mask2_np = np.array(mask2_im)

# Feed to LaMa
img2_np = np.array(crop2.convert("RGB"), dtype=np.float32) / 255.0
img2_tensor = np.transpose(img2_np, (2, 0, 1))[None, ...].astype(np.float32)
mask2_tensor = (mask2_np > 0).astype(np.float32)[None, None, ...]
out2 = sess.run(None, {"image": img2_tensor, "mask": mask2_tensor})
out2_np = np.clip(np.transpose(out2[0][0], (1, 2, 0)), 0, 255).astype(np.uint8)

inpaint_reg2 = (mask2_np == 255)
patch_rgb2 = out2_np[inpaint_reg2].mean(axis=0)
dilated2 = np.array(mask2_im.filter(ImageFilter.MaxFilter(7))) > 0
border_rgb2 = crop2_np[dilated2 & ~inpaint_reg2, :3].mean(axis=0)
delta2 = float(np.abs(border_rgb2 - patch_rgb2).mean())
print(f"Sample 2 (White T-Shirt): Border RGB={border_rgb2.round(1)}, Inpaint RGB={patch_rgb2.round(1)}, Delta={delta2:.2f}")

# Test Sample 3: Navy T-Shirt
# Crop box (250, 250, 762, 762) centered on navy chest / upper torso
c3 = Image.open("benchmark_output/flow_b_validation/img_mv53zgfo_ffc202_upper_clothes/upper-clothes_cutout.png").convert("RGBA")
crop3 = c3.crop((250, 250, 762, 762))
crop3_np = np.array(crop3)
mask3_im = Image.new("L", (512, 512), 0)
draw3 = ImageDraw.Draw(mask3_im)
draw3.ellipse([320 - 40, 250 - 30, 320 + 40, 250 + 30], fill=255)
mask3_np = np.array(mask3_im)

img3_np = np.array(crop3.convert("RGB"), dtype=np.float32) / 255.0
img3_tensor = np.transpose(img3_np, (2, 0, 1))[None, ...].astype(np.float32)
mask3_tensor = (mask3_np > 0).astype(np.float32)[None, None, ...]
out3 = sess.run(None, {"image": img3_tensor, "mask": mask3_tensor})
out3_np = np.clip(np.transpose(out3[0][0], (1, 2, 0)), 0, 255).astype(np.uint8)

inpaint_reg3 = (mask3_np == 255)
patch_rgb3 = out3_np[inpaint_reg3].mean(axis=0)
dilated3 = np.array(mask3_im.filter(ImageFilter.MaxFilter(7))) > 0
border_rgb3 = crop3_np[dilated3 & ~inpaint_reg3, :3].mean(axis=0)
delta3 = float(np.abs(border_rgb3 - patch_rgb3).mean())
print(f"Sample 3 (Navy T-Shirt): Border RGB={border_rgb3.round(1)}, Inpaint RGB={patch_rgb3.round(1)}, Delta={delta3:.2f}")

# Test Sample 4: Turquoise Embroidered Anarkali
# Crop box (256, 250, 768, 762) centered on embroidered bodice / chest panel
c4 = Image.open("benchmark_output/img_mv3p042e_e20ae6_extracted.png").convert("RGBA")
crop4 = c4.crop((256, 250, 768, 762))
crop4_np = np.array(crop4)
mask4_im = Image.new("L", (512, 512), 0)
draw4 = ImageDraw.Draw(mask4_im)
draw4.ellipse([256 - 40, 200 - 30, 256 + 40, 200 + 30], fill=255)
mask4_np = np.array(mask4_im)

img4_np = np.array(crop4.convert("RGB"), dtype=np.float32) / 255.0
img4_tensor = np.transpose(img4_np, (2, 0, 1))[None, ...].astype(np.float32)
mask4_tensor = (mask4_np > 0).astype(np.float32)[None, None, ...]
out4 = sess.run(None, {"image": img4_tensor, "mask": mask4_tensor})
out4_np = np.clip(np.transpose(out4[0][0], (1, 2, 0)), 0, 255).astype(np.uint8)

inpaint_reg4 = (mask4_np == 255)
patch_rgb4 = out4_np[inpaint_reg4].mean(axis=0)
dilated4 = np.array(mask4_im.filter(ImageFilter.MaxFilter(7))) > 0
border_rgb4 = crop4_np[dilated4 & ~inpaint_reg4, :3].mean(axis=0)
delta4 = float(np.abs(border_rgb4 - patch_rgb4).mean())
print(f"Sample 4 (Turquoise Anarkali): Border RGB={border_rgb4.round(1)}, Inpaint RGB={patch_rgb4.round(1)}, Delta={delta4:.2f}")

