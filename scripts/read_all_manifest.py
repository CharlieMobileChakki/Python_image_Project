import json
from pathlib import Path

manifest = json.load(open("benchmark_output/dataset_manifest.json"))
for i, item in enumerate(manifest):
    print(f"[{i:02d}] {item['filename']} | {item.get('category')} | {item.get('description')}")

