from pathlib import Path
from PIL import Image
import numpy as np

items = [
    ("trousers_beige", "img_mv3nprm1_b7901a.jpg", Path(r"D:\my codes\Arangtik\backend\uploads\img_mv3nprm1_b7901a.jpg")),
    ("tshirt_white_plain", "img_mv50gr9c_87dbd3.jpg", Path(r"D:\my codes\Arangtik\backend\uploads\img_mv50gr9c_87dbd3.jpg")),
    ("tshirt_navy_dark", "img_mv53zgfo_ffc202.jpg", Path(r"D:\my codes\Arangtik\backend\uploads\img_mv53zgfo_ffc202.jpg")),
    ("anarkali_turquoise_embroidered", "img_mv3p042e_e20ae6.webp", Path(r"D:\my codes\Arangtik\backend\uploads\img_mv3p042e_e20ae6.webp")),
]

for label, fname, path in items:
    print(f"\n=== {label} ({fname}) ===")
    print(f"Path exists: {path.exists()}")
    if path.exists():
        im = Image.open(path)
        print(f"Original size: {im.size}, format: {im.format}, mode: {im.mode}")
    
    # Check for extracted files
    cutouts = list(Path("benchmark_output").rglob(f"*{fname.split('.')[0]}*"))
    for c in cutouts:
        if c.is_file() and c.suffix.lower() == ".png":
            cim = Image.open(c)
            print(f"  Cutout: {c} | {cim.size} | {cim.mode}")

