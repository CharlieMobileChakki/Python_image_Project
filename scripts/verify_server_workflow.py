"""Integration test script to verify FastAPI endpoints using TestClient."""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from PIL import Image
from fastapi.testclient import TestClient
from app.main import app

def run_verification():
    print("Initializing TestClient for Arangtik Garment AI...")
    with TestClient(app) as client:
        # 1. Health check
        res = client.get("/health")
        print(f"1. GET /health -> HTTP {res.status_code}: {res.json()}")
        assert res.status_code == 200

        # 2. Detailed API v1 Health
        res = client.get("/api/v1/health")
        print(f"2. GET /api/v1/health -> HTTP {res.status_code}: {res.json()}")
        assert res.status_code == 200

        # 3. Test Option 2 API with Human Model (Sample 1)
        sample1_path = Path("benchmark_output/phase14/sample1_trousers_beige/original.png")
        if sample1_path.exists():
            print(f"\n3. Testing POST /api/v1/process-garment with Human Model: {sample1_path.name}")
            with open(sample1_path, "rb") as f:
                files = {"file": (sample1_path.name, f.read(), "image/png")}
                data = {"auto_repair": "true", "auto_straighten": "true", "target_garment": "all"}
                res = client.post("/api/v1/process-garment", files=files, data=data)

            print(f"   Response HTTP {res.status_code}")
            resp_data = res.json()
            print(f"   Mode: {resp_data.get('mode')}, Has Human: {resp_data.get('has_human')}")
            print(f"   Garments Extracted: {resp_data.get('garments_found')}")
            for g in resp_data.get("garments", []):
                print(f"     - {g['category']} ({g['recognized_label']}): {g['output_png_url']}")
                # 4. Verify output download
                download_res = client.get(g["output_png_url"])
                assert download_res.status_code == 200
                assert download_res.headers["content-type"] == "image/png"
                img = Image.open(Path(g["output_png_url"].replace("/api/v1/", "")))
                print(f"       Verified Downloaded PNG: size={img.size}, mode={img.mode}")
                assert img.size == (1024, 1024)
                assert img.mode == "RGBA"

        # 5. Test Option 2 API with Flat-lay/Product (Sample 2)
        sample2_path = Path("benchmark_output/phase14/sample2_tshirt_white_plain/original.png")
        if sample2_path.exists():
            print(f"\n4. Testing POST /api/v1/process-garment with Isolated Product: {sample2_path.name}")
            with open(sample2_path, "rb") as f:
                files = {"file": (sample2_path.name, f.read(), "image/png")}
                data = {"auto_repair": "true", "auto_straighten": "true"}
                res = client.post("/api/v1/process-garment", files=files, data=data)

            print(f"   Response HTTP {res.status_code}")
            resp_data = res.json()
            print(f"   Mode: {resp_data.get('mode')}, Has Human: {resp_data.get('has_human')}")
            print(f"   Garments Extracted: {resp_data.get('garments_found')}")
            for g in resp_data.get("garments", []):
                print(f"     - {g['category']} ({g['recognized_label']}): {g['output_png_url']}")
                download_res = client.get(g["output_png_url"])
                assert download_res.status_code == 200

        print("\nALL API WORKFLOW VERIFICATIONS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_verification()
