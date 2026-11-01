"""End-to-end verification script for unified Garment AI pipeline."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from PIL import Image
from app.core.pipeline import process_garment_pipeline

def test_samples():
    # Test 1: Human model sample
    p1 = Path("benchmark_output/phase14/sample1_trousers_beige/original.png")
    if p1.exists():
        print(f"Testing Sample 1 (Human Model): {p1}")
        with open(p1, "rb") as f:
            bytes_data = f.read()
        res1 = process_garment_pipeline(bytes_data, auto_repair=True, auto_straighten=True)
        print("Sample 1 Result:")
        print(json.dumps({k: v for k, v in res1.items() if k != "garments"}, indent=2))
        for g in res1["garments"]:
            print(f"  - Garment: {g['category']} ({g['recognized_label']}), Straightened: {g['rotation_angle_deg']} deg, Repaired: {g['repaired']}, PNG: {g['output_png_url']}")

    # Test 2: Isolated product sample
    p2 = Path("benchmark_output/phase14/sample2_tshirt_white_plain/original.png")
    if p2.exists():
        print(f"\nTesting Sample 2 (Isolated Product): {p2}")
        with open(p2, "rb") as f:
            bytes_data = f.read()
        res2 = process_garment_pipeline(bytes_data, auto_repair=True, auto_straighten=True)
        print("Sample 2 Result:")
        print(json.dumps({k: v for k, v in res2.items() if k != "garments"}, indent=2))
        for g in res2["garments"]:
            print(f"  - Garment: {g['category']} ({g['recognized_label']}), Straightened: {g['rotation_angle_deg']} deg, Repaired: {g['repaired']}, PNG: {g['output_png_url']}")

if __name__ == "__main__":
    test_samples()
