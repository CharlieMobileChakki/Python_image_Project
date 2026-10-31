from pathlib import Path
from PIL import Image

samples = {
    "sample1_trousers": "img_mv3nprm1_b7901a",
    "sample2_tshirt_white": "img_mv50gr9c_87dbd3",
    "sample3_tshirt_navy": "img_mv53zgfo_ffc202",
    "sample4_anarkali": "img_mv3p042e_e20ae6",
}

for name, key in samples.items():
    print(f"\n=== {name} ({key}) ===")
    matches = list(Path("benchmark_output").rglob(f"*{key}*"))
    for m in matches:
        if m.is_file():
            try:
                im = Image.open(m)
                print(f"  {m} | {im.size} | {im.mode}")
            except Exception:
                print(f"  {m} | (not an image)")
        elif m.is_dir():
            print(f"  [DIR] {m}")
            for sub in m.glob("*"):
                if sub.is_file():
                    try:
                        im = Image.open(sub)
                        print(f"    {sub.name} | {im.size} | {im.mode}")
                    except Exception:
                        print(f"    {sub.name}")

