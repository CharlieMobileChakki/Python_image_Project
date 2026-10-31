import time
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import onnxruntime as ort
import psutil

# Load ONNX model
model_path = r"D:\huggingface_cache\hub\models--opencv--inpainting_lama\snapshots\aee6d22f0a13e5e35af1c9a1c3afd62841fc6f3f\inpainting_lama_2025jan.onnx"
sess_opts = ort.SessionOptions()
sess_opts.intra_op_num_threads = psutil.cpu_count(logical=True)
sess = ort.InferenceSession(model_path, sess_opts, providers=["CPUExecutionProvider"])
print("LaMa ONNX session created.")

def run_lama(img_512_rgb: Image.Image, mask_512_bin: np.ndarray) -> tuple[np.ndarray, float]:
    img_np = np.array(img_512_rgb, dtype=np.float32) / 255.0
    img_tensor = np.transpose(img_np, (2, 0, 1))[None, ...].astype(np.float32)
    mask_tensor = (mask_512_bin > 0).astype(np.float32)[None, None, ...]
    
    t0 = time.perf_counter()
    out = sess.run(None, {"image": img_tensor, "mask": mask_tensor})
    infer_time = time.perf_counter() - t0
    
    out_np = np.clip(np.transpose(out[0][0], (1, 2, 0)), 0, 255).astype(np.uint8)
    return out_np, infer_time

print("Ready to test inference.")

