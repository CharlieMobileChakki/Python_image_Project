import json
from pathlib import Path

p = Path("benchmark_output/dataset_manifest.json")
if p.exists():
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    print("Manifest length:", len(data))
    for i, item in enumerate(data):
        print(f"[{i}] {item.get('image_id', '')} - filename: {item.get('filename', '')} - path: {item.get('local_path', '')} exists={Path(item.get('local_path', '')).exists() if item.get('local_path') else False}")

