"""Inspect input image files and existing cutouts for Phase 14."""
import json
from pathlib import Path
from PIL import Image

manifest_path = Path("dataset_manifest.json")
if manifest_path.exists():
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total manifest entries: {len(data)}")
    for item in data:
        fname = item.get("filename", "")
        if any(k in fname for k in ["b7901a", "87dbd3", "ffc202", "e20ae6"]):
            p = Path(item.get("local_path", ""))
            print(f"- {fname}: exists={p.exists()}, shape={Image.open(p).size if p.exists() else None}, path={p}")

# Also check existing cutouts in benchmark_output
print("\nChecking existing benchmark_output cutouts:")
candidate_patterns = ["*b7901a*", "*87dbd3*", "*ffc202*", "*e20ae6*"]
for pat in candidate_patterns:
    matches = list(Path("benchmark_output").rglob(pat))
    print(f"Pattern {pat}: {len(matches)} files found")
    for m in matches[:5]:
        print(f"  {m}")

