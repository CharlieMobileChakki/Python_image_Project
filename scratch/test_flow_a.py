import sys
sys.path.insert(0, ".")
from PIL import Image
import numpy as np
from app.core.segmenter import extract_isolated_flow_a, run_segformer_inference
from app.core.detector import analyze_semantic_map

# Test Red Dress
red_dress = Image.open("input images/red_dress_exact.png").convert("RGB")
print("Red dress size:", red_dress.size)

# 1. Test BiRefNet Lite directly
fg_red = extract_isolated_flow_a(red_dress)
fg_red.save("benchmark_output/test_red_dress_birefnet.png")
print("Saved benchmark_output/test_red_dress_birefnet.png")

# 2. Test SegFormer inference on red dress
pred_map, probs_tensor = run_segformer_inference(red_dress)
scene_info = analyze_semantic_map(pred_map)
print("Red dress scene_info:", scene_info)

# Check classes detected by SegFormer
classes, counts = np.unique(pred_map, return_counts=True)
print("SegFormer classes and counts:", list(zip(classes, counts)))
