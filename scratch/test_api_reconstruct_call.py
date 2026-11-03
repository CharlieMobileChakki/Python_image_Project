import sys
sys.path.insert(0, ".")
from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    with open("test_results_sdxl_inference/original_unnamed.jpg", "rb") as f_orig, \
         open("test_results_sdxl_inference/damaged_garment_unnamed.png", "rb") as f_dam, \
         open("test_results_sdxl_inference/repair_mask_unnamed.png", "rb") as f_mask, \
         open("test_results_sdxl_inference/garment_reference_unnamed.png", "rb") as f_ref:
        files = {
            "original_image": ("original_unnamed.jpg", f_orig, "image/jpeg"),
            "damaged_garment": ("damaged_garment_unnamed.png", f_dam, "image/png"),
            "repair_mask": ("repair_mask_unnamed.png", f_mask, "image/png"),
            "garment_reference": ("garment_reference_unnamed.png", f_ref, "image/png"),
        }
        resp = client.post("/api/v1/reconstruct-garment", files=files)
        print("API Status Code:", resp.status_code)
        print("API Response Headers:", dict(resp.headers))
        print("API JSON Response:", resp.json())

