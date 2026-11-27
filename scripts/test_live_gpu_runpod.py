"""Live RunPod GPU Worker Virtual Try-On Verification Script.

Executes real end-to-end multi-image inference against the remote RunPod A40 GPU worker,
verifying facial biometric identity preservation, landmark detection, dynamic natural poses,
anti-nudity coverage verification, and latency logging.
"""
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Ensure app is on path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.core.pipeline.virtual_tryon import process_virtual_tryon_pipeline
from app.core.image_processing.face_analyzer import (
    detect_face_and_landmarks,
    extract_face_embedding,
    compute_face_similarity,
)

USER_UPLOADED = Path(r"C:\Users\admin\.gemini\antigravity\brain\5677d129-15c2-4786-b187-5ef396092e73\.user_uploaded")
avatar_path = USER_UPLOADED / "media_1795774374012_6a893be2.png"
garment_path = ROOT_DIR / "samples" / "garment_sample.jpg"

if not avatar_path.exists():
    print(f"ERROR: Avatar file not found at {avatar_path}")
    sys.exit(1)

if not garment_path.exists():
    print(f"ERROR: Garment file not found at {garment_path}")
    sys.exit(1)

with open(avatar_path, "rb") as f:
    avatar_bytes = f.read()

with open(garment_path, "rb") as f:
    garment_bytes = f.read()

print("=" * 70)
print("STARTING LIVE RUNPOD GPU VIRTUAL TRY-ON VERIFICATION")
print(f"Avatar: {avatar_path.name} ({len(avatar_bytes)} bytes)")
print(f"Garment: {garment_path.name} ({len(garment_bytes)} bytes)")
print("Endpoint:", os.getenv("QWEN_REMOTE_URL"))
print("=" * 70)

t0 = time.perf_counter()
result = process_virtual_tryon_pipeline(
    avatar_bytes=avatar_bytes,
    garment_entries=[
        {"bytes": garment_bytes, "role_hint": "top", "filename": "garment_sample.jpg"},
    ],
    max_combinations=1,
    occasion="smart_casual",
    base_url="http://localhost:8000",
)
elapsed = time.perf_counter() - t0

print("\n" + "=" * 70)
print(f"PIPELINE COMPLETED IN {elapsed:.2f}s (reported: {result['processing_time_sec']}s)")
print("Status:", result["status"])
print("Combinations:", len(result["combinations"]))

for i, look in enumerate(result["combinations"]):
    print(f"\n--- LOOK {i+1}: {look['title']} ---")
    print("Resolution:", look["resolution"])
    print("Engine:", look["engine"], "| GPU:", look["gpu_name"])
    print("Output URL:", look["url"])
    print("Style reason:", look["style_reason"])

    # Garment & complement details
    print("Garment items:")
    for g in look["garment_details"]:
        comp_tag = " [AI Modest Complement]" if g.get("is_complementary") else ""
        print(f"  - {g['role'].upper()}: {g['label']}{comp_tag} ({g['details']})")

    # Validation metrics
    val = look.get("validation_metrics") or {}
    print("\nBiometric & Quality Validation:")
    print(f"  Grade: {val.get('validation_grade', 'N/A')}")
    print(f"  Face Similarity (SFace): {val.get('face_similarity', 0.0):.3f}")
    print(f"  Same Identity Verified: {val.get('same_identity', False)}")
    print(f"  Face Detected in Output: {val.get('face_detected', False)}")
    print(f"  Landmarks Detected: {len(val.get('facial_landmarks_detected', []))} points")
    print(f"  Head Tilt Angle: {val.get('head_tilt_angle_deg', 0.0)} deg")

    cov = val.get("clothing_coverage") or {}
    print(f"  Clothing Coverage Passed: {cov.get('passed', False)}")
    print(f"    Torso Covered: {cov.get('torso_covered', False)} (skin fraction: {cov.get('torso_skin_fraction', 0.0):.3f})")
    print(f"    Legs Covered: {cov.get('legs_covered', False)} (skin fraction: {cov.get('legs_skin_fraction', 0.0):.3f})")

    print(f"\nDynamic Prompt excerpt:")
    prompt = look.get("prompt_used", "")
    print(f"  Pose excerpt: {prompt[prompt.find('PROFESSIONAL FASHION MODEL'):prompt.find('COMPLETE GARMENT DRESSING')][:180]}...")
    print(f"  Dressing excerpt: {prompt[prompt.find('COMPLETE GARMENT DRESSING'):prompt.find('PHOTOGRAPHY DIRECTION')][:180]}...")

print("\nDiagnostics:")
for k, v in result.get("diagnostics", {}).items():
    print(f"  {k}: {v}")
print("=" * 70)
