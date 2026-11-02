import sys
sys.path.insert(0, ".")
import os
from PIL import Image
from app.core.pipeline import process_garment_pipeline

def test_image(label, path):
    print(f"\n==================== TESTING: {label} ====================")
    with open(path, "rb") as f:
        img_bytes = f.read()

    res = process_garment_pipeline(img_bytes, target_garment="all", auto_repair=True, auto_straighten=True)
    print("Mode:", res.get("mode"))
    print("Scene Type:", res.get("scene_type"))
    print("Total Detected:", res.get("total_detected"))
    print("Total Extracted:", res.get("total_extracted"))
    print("Total BG Removed:", res.get("total_background_removed"))
    print("Garments count:", len(res.get("garments", [])))

    for idx, g in enumerate(res.get("garments", [])):
        print(f"  [{idx+1}] ID={g['garment_id']}, Cat={g['category']}, Label={g['recognized_label']}, "
              f"BG_Status={g.get('background_removal_status')}, Conf={g.get('background_removal_confidence')}, "
              f"Preserve={g.get('garment_preservation_status')}, File={g.get('image')}")
    return res

# 1. Multi-Garment Blue Test
res_multi = test_image("12-Garment Blue Layout", "input images/istockphoto_850788940_612x612.png")

# 2. Red Dress Test (Neckline & Embroidery Preservation)
res_red = test_image("Red Party Dress on White Fur", "input images/red_dress_exact.png")

# 3. Orange Kurta Test (Chest Placket Preservation, No LaMa Hole)
res_orange = test_image("Orange Embroidered Kurta", "input images/orange_kurta_exact.png")

# 4. Nawab Model Test (Human Parsing: Kurta + Pajama extraction, Human Removal)
res_nawab = test_image("Nawab Kurta Pajama Model", "scratch/nawab_kurta_pajama.png")

