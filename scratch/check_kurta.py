import sys
sys.path.insert(0, ".")
from PIL import Image
import numpy as np
from app.core.segmenter import run_segformer_inference
from app.core.detector import analyze_semantic_map, ATR_CLASSES

img = Image.open("input images/pngtree_orange_kurta_exact.png").convert("RGB")
pred_map, probs_tensor = run_segformer_inference(img)
classes, counts = np.unique(pred_map, return_counts=True)
print("Classes and counts:")
for c, cnt in zip(classes, counts):
    pct = (cnt / pred_map.size) * 100
    print(f"  Class {c:2d} ({ATR_CLASSES.get(c, 'Unknown'):15s}): {cnt:6d} pixels ({pct:.2f}%)")

scene_info = analyze_semantic_map(pred_map)
print("\nScene Info Result:", scene_info)

