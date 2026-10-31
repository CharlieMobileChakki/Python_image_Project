import json
from pathlib import Path

p10_metrics = list(Path("benchmark_output/phase10").glob("*/metrics.json"))
for m in p10_metrics:
    with open(m, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"{m.parent.name}:")
    print(f"  image_path: {data.get('image_path')}")
    print(f"  classes: {data.get('classes')}")
    print(f"  image_size: {data.get('image_size')}")

fb_metrics = list(Path("benchmark_output/flow_b_validation").glob("*/flow_b_metrics.json"))
for m in fb_metrics:
    with open(m, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"{m.parent.name}:")
    print(f"  image_path: {data.get('image_path')}")
    print(f"  class_name: {data.get('class_name')}")

