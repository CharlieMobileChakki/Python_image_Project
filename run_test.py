"""Arangtik Garment AI — Unified Developer & Testing CLI Tool.

Run interactively:
    python run_test.py

Or with command-line arguments:
    python run_test.py --image "input_images/unnamed.jpg"
    python run_test.py --pytest
    python run_test.py --server
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("U2NET_HOME", str(PROJECT_ROOT / ".u2net"))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

INPUT_DIR = PROJECT_ROOT / "input_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_single_image(image_path: Path | str) -> bool:
    """Run full unified pipeline extraction and repair on a single image."""
    img_file = Path(image_path)
    if not img_file.is_file():
        print(f"\n❌ Error: File not found: {img_file}")
        return False

    print(f"\n=======================================================")
    print(f"👗 Processing: {img_file.name}")
    print(f"📁 Path: {img_file.resolve()}")
    print(f"=======================================================")

    try:
        from app.core.pipeline import process_garment_pipeline

        img_bytes = img_file.read_bytes()
        t0 = time.perf_counter()
        result = process_garment_pipeline(
            img_bytes,
            target_garment="all",
            auto_repair=True,
            auto_straighten=True,
        )
        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        print(f"\n✅ Pipeline Finished in {elapsed} ms!")
        print(f" - Job ID: {result.get('job_id')}")
        print(f" - Scene Type: {result.get('scene_type')}")
        print(f" - Human Present: {result.get('has_human')}")
        print(f" - Outfit Mode: {result.get('outfit_mode')}")
        print(f" - Total Garments Found: {result.get('garments_found')}")
        print(f"\n📦 Extracted Garment Results:")
        for idx, g in enumerate(result.get("garments", []), 1):
            print(f"   [{idx}] ID: {g.get('garment_id')}")
            print(f"       Category: {g.get('category')}")
            print(f"       Label: {g.get('recognized_label')}")
            print(f"       Fused Ensemble: {g.get('is_fused')}")
            print(f"       Repaired (LaMa): {g.get('repaired')} ({g.get('repair_method')})")
            print(f"       Quality: {g.get('garment_preservation_status')} (confidence: {g.get('confidence')})")
            out_file = OUTPUT_DIR / g.get("image", "")
            print(f"       💾 Output File: {out_file.resolve()}")

        print(f"\n✨ Test completed successfully!")
        return True

    except Exception as exc:
        print(f"\n❌ Pipeline failed with error: {exc}")
        import traceback
        traceback.print_exc()
        return False


def run_unit_tests() -> int:
    """Execute pytest on the test suite."""
    print("\n🧪 Running Pytest Automated Unit Tests...\n")
    pytest_bin = PROJECT_ROOT / ".venv-flowb" / "Scripts" / "pytest.exe"
    cmd = [str(pytest_bin) if pytest_bin.exists() else "pytest", "tests/", "-v"]
    return subprocess.run(cmd).returncode


def start_server(port: int = 8000) -> None:
    """Launch the FastAPI server with auto-reload."""
    print(f"\n🚀 Starting Arangtik Garment AI Web Server on port {port}...")
    print(f"📱 Browser UI will be available at: http://localhost:{port}/")
    print(f"📖 Interactive API Docs: http://localhost:{port}/docs\n")
    uvicorn_bin = PROJECT_ROOT / ".venv-flowb" / "Scripts" / "uvicorn.exe"
    cmd = [
        str(uvicorn_bin) if uvicorn_bin.exists() else "uvicorn",
        "app.main:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    subprocess.run(cmd)


def interactive_menu() -> None:
    """Display interactive CLI testing dashboard."""
    while True:
        print("\n" + "=" * 60)
        print("👗 ARANGTIK GARMENT AI — MASTER TESTING & DEV TOOL")
        print("=" * 60)
        print(" [1] 🖼️  Test Single Garment Image (Suit / Saree / Dress)")
        print(" [2] 🧪  Run Full Automated Pytest Suite")
        print(" [3] 📊  Run Performance Benchmark (Qwen Model)")
        print(" [4] 🚀  Start Local Web Server (UI at http://localhost:8000)")
        print(" [0] ❌  Exit")
        print("=" * 60)

        choice = input("\nEnter choice [0-4]: ").strip()

        if choice == "1":
            images = list(INPUT_DIR.glob("*.*"))
            valid_images = [img for img in images if img.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
            if not valid_images:
                print(f"No sample images found in {INPUT_DIR}.")
                continue

            print("\nSelect a sample image from input_images/:")
            for idx, img in enumerate(valid_images, 1):
                print(f" [{idx}] {img.name}")
            print(" [C] Enter custom path")

            sub_choice = input("\nEnter image number or [C]: ").strip()
            if sub_choice.isdigit() and 1 <= int(sub_choice) <= len(valid_images):
                selected = valid_images[int(sub_choice) - 1]
                test_single_image(selected)
            elif sub_choice.upper() == "C":
                custom_path = input("Enter image path: ").strip().strip('"').strip("'")
                test_single_image(custom_path)
            else:
                print("Invalid selection.")

        elif choice == "2":
            run_unit_tests()

        elif choice == "3":
            benchmark_script = PROJECT_ROOT / "tests" / "benchmarks" / "benchmark_qwen_garment.py"
            py_bin = PROJECT_ROOT / ".venv-flowb" / "Scripts" / "python.exe"
            subprocess.run([str(py_bin) if py_bin.exists() else sys.executable, str(benchmark_script)])

        elif choice == "4":
            start_server()

        elif choice == "0":
            print("\nGoodbye! 👋\n")
            break
        else:
            print("\nInvalid choice. Please select from 0 to 4.")


def main():
    parser = argparse.ArgumentParser(description="Arangtik Garment AI Testing Runner")
    parser.add_argument("--image", "-i", type=str, help="Path to single image to process and test")
    parser.add_argument("--pytest", "-t", action="store_true", help="Run pytest unit tests")
    parser.add_argument("--server", "-s", action="store_true", help="Launch FastAPI web server")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port for web server")

    args = parser.parse_args()

    if args.image:
        test_single_image(args.image)
    elif args.pytest:
        run_unit_tests()
    elif args.server:
        start_server(args.port)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
