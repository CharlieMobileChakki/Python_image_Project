import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

from PIL import Image
import numpy as np

# Let's inspect Sample 2 with white background compositing
c2 = Image.open("benchmark_output/img_mv50gr9c_87dbd3_extracted.png").convert("RGBA")
crop2 = c2.crop((256, 50, 768, 562))

# Composite crop2 on white background
bg_white = Image.new("RGBA", crop2.size, (255, 255, 255, 255))
crop2_white = Image.alpha_composite(bg_white, crop2).convert("RGB")

from scripts.run_phase14_multigarment_lama import generate_mask_sample2
m2, a2, g2 = generate_mask_sample2(np.array(crop2))

import onnxruntime as ort
import psutil

model_path = r"D:\huggingface_cache\hub\models--opencv--inpainting_lama\snapshots\aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f\inpainting_lama_2025jan.onnx"
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = psutil.cpu_count(logical=True)
sess = ort.InferenceSession(model_path, sess_opts, providers=["CPUExecutionProvider"])

img_np = np.array(crop2_white, dtype=np.float32) / 255.0
img_tensor = np.transpose(img_np, (2, 0, 1))[None, ...].astype(np.float32)
mask_tensor = (m2 > 0).astype(np.float32)[None, None, ...]

out = sess.run(None, {"image": img_tensor, "mask": mask_tensor})
out_np = np.clip(np.transpose(out[0][0], (1, 2, 0)), 0, 255).astype(np.uint8)

inpaint_region = (m2 == 255)
patch_mean = out_np[inpaint_region].mean(axis=0)
border_mean = np.array(crop2_white)[g2 & ~inpaint_region].mean(axis=0)

print("Sample 2 with white background:")
print("  Border fabric mean RGB:", border_mean)
print("  Inpainted patch mean RGB:", patch_mean)
print("  Mean border delta:", np.abs(border_mean - patch_mean).mean())
