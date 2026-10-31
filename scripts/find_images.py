"""Find image files and manifest in the project."""
from pathlib import Path

print("Searching for manifest or image directories...")
for p in Path(".").rglob("*manifest*"):
    print(f"Manifest match: {p}")

print("\nSearching for test images...")
for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
    files = list(Path(".").rglob(ext))
    # Exclude venvs
    files = [f for f in files if not any(v in str(f) for v in [".venv", "venv", ".git"])]
    print(f"Found {len(files)} files with {ext}")
    for f in files[:10]:
        print(f"  {f}")

