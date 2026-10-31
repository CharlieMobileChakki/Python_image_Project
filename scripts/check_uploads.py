import json
from pathlib import Path

manifest_path = Path("benchmark_output/dataset_manifest.json")
with open(manifest_path, "r", encoding="utf-8") as f:
    items = json.load(f)

for item in items:
    fn = item.get("filename")
    fp = item.get("full_path")
    if fp:
        p = Path(fp)
        print(f"{fn}: exists={p.exists()} (size={p.stat().st_size if p.exists() else 0})")

