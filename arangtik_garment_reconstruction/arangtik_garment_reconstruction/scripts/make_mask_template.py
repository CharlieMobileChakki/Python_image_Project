"""Create an empty black repair mask matching a damaged garment image.

Open the generated mask in an image editor and paint only missing garment fabric WHITE.
Do not paint the transparent background white.
"""
import argparse
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("damaged_image", help="Path to damaged garment PNG")
parser.add_argument("output_mask", help="Path for output black mask PNG")
args = parser.parse_args()

image = Image.open(args.damaged_image)
mask = Image.new("L", image.size, 0)
Path(args.output_mask).parent.mkdir(parents=True, exist_ok=True)
mask.save(args.output_mask)
print(f"Created {args.output_mask} with size {mask.size}.")
print("Paint missing garment pixels white (255); keep all other pixels black (0).")
