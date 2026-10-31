"""Repeatable Garment Segmentation Benchmark Harness for Arangtik.

Evaluates candidate models across a folder of isolated garment images.
Measures latency, alpha-channel distribution, foreground bounding boxes,
and generates JSON/CSV reports with an optional visual contact sheet.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from app.services.extraction import MODEL_NAME, extract_garment, load_model


def make_checkerboard(width: int, height: int, square_size: int = 16) -> Image.Image:
    """Create a white/light-grey checkerboard canvas to visually assess transparency."""
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 1:
                draw.rectangle([x, y, x + square_size - 1, y + square_size - 1], fill=(224, 228, 224, 255))
    return canvas


def evaluate_image(
    image_path: Path,
    output_dir: Path,
    save_cutout: bool = True,
) -> dict[str, Any]:
    """Run extraction on one image and collect technical diagnostics."""
    record: dict[str, Any] = {
        "filename": image_path.name,
        "input_path": str(image_path),
        "file_size_bytes": image_path.stat().st_size,
        "model": MODEL_NAME,
        "success": False,
        "error": None,
        "error_type": None,
        "latency_ms": None,
        "input_dimensions": None,
        "output_dimensions": None,
        "alpha_transparent_count": None,
        "alpha_core_opaque_count": None,
        "alpha_edge_transition_count": None,
        "foreground_coverage_pct": None,
        "bbox": None,
        "bbox_width": None,
        "bbox_height": None,
        "center_offset_x": None,
        "center_offset_y": None,
        "output_path": None,
    }

    try:
        image_bytes = image_path.read_bytes()
    except Exception as exc:
        record["error"] = f"Failed to read file: {exc}"
        record["error_type"] = type(exc).__name__
        return record

    start_time = time.perf_counter()
    try:
        png_bytes, original_size = extract_garment(image_bytes)
        duration_ms = round((time.perf_counter() - start_time) * 1000)
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start_time) * 1000)
        record["latency_ms"] = duration_ms
        record["error"] = str(exc)
        record["error_type"] = type(exc).__name__
        return record

    record["success"] = True
    record["latency_ms"] = duration_ms
    record["input_dimensions"] = list(original_size)

    # Save output PNG
    out_file = output_dir / f"{image_path.stem}_extracted.png"
    if save_cutout:
        out_file.write_bytes(png_bytes)
        record["output_path"] = str(out_file)

    # Technical alpha inspection
    try:
        with Image.open(out_file if save_cutout else image_bytes) as img:
            record["output_dimensions"] = list(img.size)
            alpha_arr = np.array(img.getchannel("A"))

            total_px = alpha_arr.size
            trans_px = int((alpha_arr == 0).sum())
            core_px = int((alpha_arr >= 250).sum())
            edge_px = int(((alpha_arr > 8) & (alpha_arr < 250)).sum())
            fg_px = int((alpha_arr > 8).sum())

            record["alpha_transparent_count"] = trans_px
            record["alpha_core_opaque_count"] = core_px
            record["alpha_edge_transition_count"] = edge_px
            record["foreground_coverage_pct"] = round((fg_px / total_px) * 100, 2)

            # Bounding box
            mask = img.getchannel("A").point(lambda v: 255 if v > 8 else 0)
            bbox = mask.getbbox()
            if bbox:
                left, upper, right, lower = bbox
                w = right - left
                h = lower - upper
                record["bbox"] = list(bbox)
                record["bbox_width"] = w
                record["bbox_height"] = h
                cx = (left + right) / 2.0
                cy = (upper + lower) / 2.0
                record["center_offset_x"] = round(cx - 512.0, 1)
                record["center_offset_y"] = round(cy - 512.0, 1)
    except Exception as exc:
        record["error"] = f"Failed to analyze output PNG: {exc}"
        record["error_type"] = type(exc).__name__

    return record


def build_contact_sheet(
    records: list[dict[str, Any]],
    output_path: Path,
    thumb_size: int = 256,
) -> None:
    """Generate a side-by-side contact sheet of original vs extracted garments."""
    successful = [r for r in records if r.get("success") and r.get("output_path")]
    if not successful:
        print("No successful extractions to include in contact sheet.")
        return

    n_items = len(successful)
    cols = min(4, n_items)
    rows = (n_items + cols - 1) // cols

    card_w = thumb_size * 2 + 30
    card_h = thumb_size + 60
    padding = 20

    sheet_w = cols * card_w + (cols + 1) * padding
    sheet_h = rows * card_h + (rows + 1) * padding

    sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 246, 243))
    draw = ImageDraw.Draw(sheet)

    for idx, rec in enumerate(successful):
        c = idx % cols
        r = idx // cols

        card_x = padding + c * (card_w + padding)
        card_y = padding + r * (card_h + padding)

        # Draw card background
        draw.rounded_rectangle(
            [card_x, card_y, card_x + card_w - 1, card_y + card_h - 1],
            radius=12,
            fill=(255, 255, 255),
            outline=(220, 224, 220),
            width=1,
        )

        # Load and scale original
        try:
            with Image.open(rec["input_path"]) as orig:
                orig = ImageOps.exif_transpose(orig).convert("RGB")
                orig.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                ox = card_x + 10 + (thumb_size - orig.width) // 2
                oy = card_y + 10 + (thumb_size - orig.height) // 2
                sheet.paste(orig, (ox, oy))
        except Exception:
            pass

        # Load and scale cutout over checkerboard
        try:
            with Image.open(rec["output_path"]) as cutout:
                cb = make_checkerboard(thumb_size, thumb_size, square_size=12)
                cutout_thumb = cutout.resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                cb.alpha_composite(cutout_thumb)
                cx = card_x + thumb_size + 20
                cy = card_y + 10
                sheet.paste(cb.convert("RGB"), (cx, cy))
        except Exception:
            pass

        # Caption
        latency = f"{rec.get('latency_ms', 0) / 1000:.1f}s"
        caption = f"{rec['filename'][:24]} | {latency} | cov: {rec.get('foreground_coverage_pct', 0)}%"
        draw.text((card_x + 10, card_y + thumb_size + 20), caption, fill=(60, 65, 60))

    sheet.save(output_path, quality=90)
    print(f"Contact sheet saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Arangtik Garment Segmentation Benchmark")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing garment images")
    parser.add_argument("--output-dir", type=str, default="benchmark_output", help="Directory for extracted images and reports")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--contact-sheet", action="store_true", help="Generate side-by-side visual contact sheet")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find supported images
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    candidates = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in supported_extensions]
    candidates.sort()

    if args.limit:
        candidates = candidates[:args.limit]

    if not candidates:
        raise SystemExit(f"No supported images found in {input_dir}")

    print(f"=== Starting Garment Benchmark ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Found {len(candidates)} images in {input_dir}")
    print(f"Output directory: {output_dir}\n")

    # Ensure model is warmed up once before measuring latency
    print("Pre-loading model weights...")
    load_model()
    print("Model ready.\n")

    records: list[dict[str, Any]] = []
    for idx, path in enumerate(candidates, 1):
        print(f"[{idx}/{len(candidates)}] Processing {path.name} ({path.stat().st_size // 1024} KB)...", end=" ", flush=True)
        rec = evaluate_image(path, output_dir=output_dir, save_cutout=True)
        if rec["success"]:
            print(f"OK ({rec['latency_ms']} ms, coverage: {rec['foreground_coverage_pct']}%)")
        else:
            print(f"FAILED ({rec['error_type']}: {rec['error']})")
        records.append(rec)

    # Save JSON report
    json_path = output_dir / "benchmark_report.json"
    summary_data = {
        "model": MODEL_NAME,
        "total_images": len(records),
        "successful_extractions": sum(1 for r in records if r["success"]),
        "failed_extractions": sum(1 for r in records if not r["success"]),
        "accuracy_measurement_status": "Manual visual verification required (ground-truth reference masks not supplied).",
        "records": records,
    }
    json_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    print(f"\nJSON report written to: {json_path}")

    # Save CSV report
    csv_path = output_dir / "benchmark_report.csv"
    if records:
        fields = [
            "filename", "model", "success", "latency_ms", "input_dimensions",
            "foreground_coverage_pct", "bbox_width", "bbox_height",
            "center_offset_x", "center_offset_y", "error"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                row = dict(r)
                row["input_dimensions"] = f"{r['input_dimensions'][0]}x{r['input_dimensions'][1]}" if r.get("input_dimensions") else ""
                writer.writerow(row)
        print(f"CSV report written to: {csv_path}")

    # Optional contact sheet
    if args.contact_sheet:
        sheet_path = output_dir / "contact_sheet.png"
        build_contact_sheet(records, sheet_path)

    # Print summary table
    latencies = [r["latency_ms"] for r in records if r["success"] and r["latency_ms"]]
    print("\n=== Benchmark Summary ===")
    print(f"Total Processed: {len(records)}")
    print(f"Success Rate:    {sum(1 for r in records if r['success'])} / {len(records)}")
    if latencies:
        print(f"Latency Mean:    {np.mean(latencies):.1f} ms ({np.mean(latencies)/1000:.2f} s)")
        print(f"Latency Median:  {np.median(latencies):.1f} ms ({np.median(latencies)/1000:.2f} s)")
        print(f"Latency Min:     {np.min(latencies)} ms")
        print(f"Latency Max:     {np.max(latencies)} ms")


if __name__ == "__main__":
    main()
