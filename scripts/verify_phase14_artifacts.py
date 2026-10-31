from pathlib import Path
from PIL import Image
import json

base = Path("benchmark_output/phase14")
print("Summary exists:", (base / "summary_report.json").exists())
sheet = base / "multigarment_contact_sheet.png"
print("Contact sheet exists:", sheet.exists(), "size:", Image.open(sheet).size if sheet.exists() else None)

samples = ["sample1_trousers_beige", "sample2_tshirt_white_plain", "sample3_tshirt_navy_dark", "sample4_anarkali_turquoise_embroidered"]
required_files = [
    "original.png", "garment_mask.png", "inpaint_mask.png",
    "before_cutout.png", "after_cutout.png", "comparison.png",
    "full_inpainted_transparent.png", "metrics.json"
]

all_ok = True
for s in samples:
    s_dir = base / s
    print(f"\nVerifying {s}:")
    for f in required_files:
        fp = s_dir / f
        exists = fp.exists()
        size = fp.stat().st_size if exists else 0
        if not exists or size == 0:
            all_ok = False
            print(f"  [FAIL] {f} missing or empty")
        else:
            print(f"  [OK] {f} ({size:,} bytes)")

print(f"\nOverall artifact integrity check: {'PASSED' if all_ok else 'FAILED'}")

