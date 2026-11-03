import os
import sys
import shutil
import time
import traceback
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, ".")

TEST_DIR = Path("test_results_sdxl_inference")
TEST_DIR.mkdir(parents=True, exist_ok=True)

print(f"=== SDXL + IP-Adapter Actual Inference Test Suite ===")
print(f"Test output directory: {TEST_DIR.resolve()}\n")

from app.services.inpainting.reconstruction_utils import (
    extract_garment_reference_crop,
    prepare_mask,
    validate_repair_mask,
    composite_garment,
)
from app.services.inpainting.garment_reconstruction import (
    get_sdxl_reconstruction_engine,
    GarmentReconstructionEngine,
)

# ---------------------------------------------------------------------------
# Test Case A: Magenta Suit Salwar (unnamed.jpg)
# ---------------------------------------------------------------------------
print("=====================================================================")
print("=== PREPARING TEST CASE A: Magenta Suit Salwar (unnamed.jpg) ===")
print("=====================================================================")

unnamed_src = Path("input images/unnamed.jpg")
shutil.copy(unnamed_src, TEST_DIR / "original_unnamed.jpg")

orig_img = Image.open(unnamed_src).convert("RGB")
w, h = orig_img.size

# Use existing extracted garment from artifacts or outputs
artifact_dir = Path("C:/Users/admin/.gemini/antigravity/brain/c4f3405d-29bb-4eb1-a6bb-33ba223ef665")
unnamed_art = artifact_dir / "unnamed_current_1024.png"
if unnamed_art.exists():
    shutil.copy(unnamed_art, TEST_DIR / "damaged_garment_unnamed.png")
else:
    # Create valid damaged RGBA
    damaged_a = orig_img.convert("RGBA")
    shutil.copy(unnamed_src, TEST_DIR / "damaged_garment_unnamed.png")

damaged_rgba = Image.open(TEST_DIR / "damaged_garment_unnamed.png").convert("RGBA")
print("Saved damaged garment to:", TEST_DIR / "damaged_garment_unnamed.png")

# Generate Repair Mask identifying missing hand-occluded fabric
# Crossed hands on waist region: center-waist box
mask_img = Image.new("L", orig_img.size, 0)
mask_arr = np.array(mask_img)
# Hands on unnamed.jpg are approx at y: 39% to 47%, x: 37% to 63%
y0, y1 = int(0.39 * h), int(0.47 * h)
x0, x1 = int(0.37 * w), int(0.63 * w)
mask_arr[y0:y1, x0:x1] = 255
mask_img = Image.fromarray(mask_arr, mode="L")
mask_img.save(TEST_DIR / "repair_mask_unnamed.png")
print("Saved repair mask to:", TEST_DIR / "repair_mask_unnamed.png")

# Extract Garment-Only Reference Crop (magenta torso fabric below neckline, above hands)
ref_crop = orig_img.crop((int(0.35 * w), int(0.30 * h), int(0.65 * w), int(0.40 * h)))
ref_crop.save(TEST_DIR / "garment_reference_unnamed.png")
print("Saved garment reference crop to:", TEST_DIR / "garment_reference_unnamed.png")

# 4. Attempt SDXL Inference on Case A
print("\n--- Executing SDXL Inference Attempt for Test Case A ---")
engine = get_sdxl_reconstruction_engine()
is_avail, avail_msg = engine.is_available()
print(f"Engine Availability: {is_avail} ({avail_msg})")

log_case_a = []
log_case_a.append(f"Test Case A: Magenta Suit Salwar (unnamed.jpg)")
log_case_a.append(f"Image Dimensions: {orig_img.size}")
log_case_a.append(f"Target Garment: suit_salwar")
log_case_a.append(f"Engine Available: {is_avail} ({avail_msg})")
log_case_a.append(f"Configured Device: {engine.configured_device}")
log_case_a.append(f"Allow CPU Fallback: {engine.allow_cpu_fallback}")

try:
    engine.load()
    print("Model load succeeded!")
    log_case_a.append("Model Loading: PASS")
    t0 = time.perf_counter()
    res = engine.reconstruct(
        original_reference=orig_img,
        damaged_garment=damaged_rgba,
        repair_mask=mask_img,
        garment_reference=ref_crop,
    )
    inf_sec = time.perf_counter() - t0
    res.image.save(TEST_DIR / "sdxl_generated_unnamed.png")
    log_case_a.append(f"Inference Time: {inf_sec:.2f}s")
    log_case_a.append("Status: PASS")
except Exception as exc:
    err_tb = traceback.format_exc()
    print(f"Inference Execution FAILED: {exc}")
    log_case_a.append("Inference Execution: FAIL")
    log_case_a.append(f"Exception Type: {type(exc).__name__}")
    log_case_a.append(f"Exception Message: {str(exc)}")
    log_case_a.append("\nFull Traceback:\n" + err_tb)

with open(TEST_DIR / "inference_log_unnamed.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_case_a))


# ---------------------------------------------------------------------------
# Test Case B: White Kerala Saree (pexels-abhijith-ts-33843905-18216607.jpg)
# ---------------------------------------------------------------------------
print("\n=====================================================================")
print("=== PREPARING TEST CASE B: White Kerala Saree ===")
print("=====================================================================")

saree_src = Path("input images/pexels-abhijith-ts-33843905-18216607.jpg")
shutil.copy(saree_src, TEST_DIR / "original_saree.jpg")

orig_saree = Image.open(saree_src).convert("RGB")
sw, sh = orig_saree.size

saree_art = artifact_dir / "saree_before_fix.png"
if saree_art.exists():
    shutil.copy(saree_art, TEST_DIR / "damaged_garment_saree.png")
else:
    shutil.copy(saree_src, TEST_DIR / "damaged_garment_saree.png")

damaged_saree_rgba = Image.open(TEST_DIR / "damaged_garment_saree.png").convert("RGBA")
print("Saved damaged saree to:", TEST_DIR / "damaged_garment_saree.png")

# Repair mask for missing waist and hip fabric
mask_s_img = Image.new("L", orig_saree.size, 0)
mask_s_arr = np.array(mask_s_img)
# Waist & right hand region on saree: y: 44% to 54%, x: 38% to 65%
sy0, sy1 = int(0.44 * sh), int(0.54 * sh)
sx0, sx1 = int(0.38 * sw), int(0.65 * sw)
mask_s_arr[sy0:sy1, sx0:sx1] = 255
mask_s_img = Image.fromarray(mask_s_arr, mode="L")
mask_s_img.save(TEST_DIR / "repair_mask_saree.png")
print("Saved saree repair mask to:", TEST_DIR / "repair_mask_saree.png")

# Saree-only reference crop (white pleated skirt fabric)
ref_crop_s = orig_saree.crop((int(0.35 * sw), int(0.58 * sh), int(0.65 * sw), int(0.80 * sh)))
ref_crop_s.save(TEST_DIR / "garment_reference_saree.png")
print("Saved saree reference crop to:", TEST_DIR / "garment_reference_saree.png")

# 4. Attempt SDXL Inference on Case B
print("\n--- Executing SDXL Inference Attempt for Test Case B ---")
log_case_b = []
log_case_b.append(f"Test Case B: White Kerala Saree")
log_case_b.append(f"Image Dimensions: {orig_saree.size}")
log_case_b.append(f"Target Garment: saree")
log_case_b.append(f"Engine Available: {is_avail} ({avail_msg})")
log_case_b.append(f"Configured Device: {engine.configured_device}")

try:
    engine.load()
    log_case_b.append("Model Loading: PASS")
    t0 = time.perf_counter()
    res_b = engine.reconstruct(
        original_reference=orig_saree,
        damaged_garment=damaged_saree_rgba,
        repair_mask=mask_s_img,
        garment_reference=ref_crop_s,
    )
    inf_sec = time.perf_counter() - t0
    res_b.image.save(TEST_DIR / "sdxl_generated_saree.png")
    log_case_b.append(f"Inference Time: {inf_sec:.2f}s")
    log_case_b.append("Status: PASS")
except Exception as exc:
    err_tb = traceback.format_exc()
    print(f"Inference Execution FAILED: {exc}")
    log_case_b.append("Inference Execution: FAIL")
    log_case_b.append(f"Exception Type: {type(exc).__name__}")
    log_case_b.append(f"Exception Message: {str(exc)}")
    log_case_b.append("\nFull Traceback:\n" + err_tb)

with open(TEST_DIR / "inference_log_saree.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(log_case_b))


# ---------------------------------------------------------------------------
# Section 5: Pixel Preservation Test
# ---------------------------------------------------------------------------
print("\n=====================================================================")
print("=== EXECUTING PIXEL PRESERVATION TEST ===")
print("=====================================================================")
# Test composite_garment function logic with mock generated output
test_rgba = Image.new("RGBA", (100, 100), (200, 100, 50, 255))
arr_rgba = np.array(test_rgba)
arr_rgba[40:60, 40:60] = (0, 0, 0, 0)
damaged_test = Image.fromarray(arr_rgba)

test_mask = Image.new("L", (100, 100), 0)
mask_arr_test = np.array(test_mask)
mask_arr_test[40:60, 40:60] = 255
test_mask = Image.fromarray(mask_arr_test)

gen_rgb = Image.new("RGB", (100, 100), (210, 105, 55))

comp_out = composite_garment(damaged_test, gen_rgb, test_mask)
comp_arr = np.array(comp_out)

pixel_diff_outside = np.abs(comp_arr[0, 0] - arr_rgba[0, 0]).max()
repaired_alpha = comp_arr[50, 50, 3]

print(f"Pixel difference outside repair mask: {pixel_diff_outside} (Expected: 0)")
print(f"Alpha inside repaired region: {repaired_alpha} (Expected: 255)")
preservation_pass = bool((pixel_diff_outside == 0) and (repaired_alpha == 255))
print(f"Pixel Preservation Code-Level Status: {'PASS' if preservation_pass else 'FAIL'}")


# ---------------------------------------------------------------------------
# Section 7: LaMa Regression Test
# ---------------------------------------------------------------------------
print("\n=====================================================================")
print("=== EXECUTING LAMA REGRESSION TEST ===")
print("=====================================================================")
from app.core.inpainting.lama_inpainter import repair_garment_cutout
from app.core.model_manager import is_inpainter_loaded

is_sdxl_loaded = engine.is_loaded()
print(f"SDXL loaded during normal startup/LaMa operation: {is_sdxl_loaded} (Expected: False)")

# Test repair_garment_cutout directly
dummy_garment = Image.new("RGBA", (512, 512), (180, 50, 120, 255))
dummy_arr = np.array(dummy_garment)
dummy_arr[200:260, 200:260] = (0, 0, 0, 0)
dummy_cutout = Image.fromarray(dummy_arr)

dummy_notch = np.zeros((512, 512), dtype=np.uint8)
dummy_notch[200:260, 200:260] = 255

t0 = time.perf_counter()
repaired_cutout, lama_time = repair_garment_cutout(dummy_cutout, dummy_notch)
lama_elapsed = time.perf_counter() - t0

repaired_alpha_val = np.array(repaired_cutout)[230, 230, 3]
print(f"LaMa repair completed in {lama_time:.1f}ms (Wall time: {lama_elapsed*1000:.1f}ms)")
print(f"Alpha at repaired hole: {repaired_alpha_val} (Expected > 0)")
print(f"LaMa inpainter loaded: {is_inpainter_loaded()}")
print(f"SDXL loaded after LaMa: {engine.is_loaded()} (Expected: False)")

lama_regression_pass = bool((repaired_alpha_val > 0) and (not engine.is_loaded()))
print(f"LaMa Regression Test Status: {'PASS' if lama_regression_pass else 'FAIL'}")

print(f"\nAll artifacts and logs written to: {TEST_DIR.resolve()}")
