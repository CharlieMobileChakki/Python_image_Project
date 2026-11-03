import os
import sys
sys.path.insert(0, os.path.abspath("."))
from app.core.pipeline import process_garment_pipeline

test_images = {
    '1_nawab_kurta_pajama': 'scratch/nawab_kurta_pajama.png',
    '2_istock_flatlay': 'input images/istockphoto_blue_garments.png',
    '3_kerala_saree': 'input images/pexels-abhijith-ts-33843905-18216607.jpg',
    '4_purple_suit_set': 'input images/pixel-embroidery-3-piece-suit-set.jpg',
    '5_magenta_suit': 'input images/unnamed.jpg'
}

for name, path in test_images.items():
    print(f"\n==========================================")
    print(f"=== Testing {name} ({path}) ===")
    print(f"==========================================")
    with open(path, 'rb') as f:
        img_bytes = f.read()
    try:
        res = process_garment_pipeline(img_bytes)
        print("  has_human:", res.get("has_human"))
        print("  scene:", res.get("scene"))
        print("  detected_count:", res.get("detected_count"))
        print("  extracted_count:", res.get("extracted_count"))
        garments = res.get("garments", [])
        for idx, g in enumerate(garments):
            print(f"    Garment {idx+1}: category={g.get('category')}, subcat={g.get('subcategories')}, label={g.get('label')}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
