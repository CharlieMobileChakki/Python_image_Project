from PIL import Image, ImageDraw

orig = Image.open('benchmark_output/cropped_user_original.png').convert('RGB')
w, h = orig.size

# Broken cutout (before repair)
broken = Image.open('benchmark_output/cropped_user_output.png').convert('RGBA')
broken = broken.resize((w, h))

# Repaired cutout
repaired = Image.open('benchmark_output/anarkali_repaired_fixed.png').convert('RGBA')
repaired = repaired.resize((w, h))

# Checkerboard bg for transparent images
def add_checker(rgba):
    bg = Image.new("RGBA", rgba.size, (24, 27, 34, 255))
    sq = 12
    draw = ImageDraw.Draw(bg)
    for y in range(0, rgba.size[1], sq):
        for x in range(0, rgba.size[0], sq):
            if ((x // sq) + (y // sq)) % 2 == 1:
                draw.rectangle([x, y, x + sq, y + sq], fill=(36, 42, 54, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")

comp = Image.new("RGB", (w * 3 + 20, h + 40), (13, 16, 23))
comp.paste(orig, (0, 30))
comp.paste(add_checker(broken), (w + 10, 30))
comp.paste(add_checker(repaired), (w * 2 + 20, 30))

draw = ImageDraw.Draw(comp)
draw.text((10, 8), "1. Original Model Photo", fill=(200, 200, 200))
draw.text((w + 20, 8), "2. Before (Hole at Waist)", fill=(239, 68, 68))
draw.text((w * 2 + 30, 8), "3. Fixed (LaMa Seamless Inpainted)", fill=(16, 185, 129))

comp.save("benchmark_output/anarkali_repair_comparison.png")
print("Saved benchmark_output/anarkali_repair_comparison.png!")
