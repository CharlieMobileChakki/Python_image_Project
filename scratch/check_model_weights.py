import os
import sys
from pathlib import Path

print("=== Checking Hugging Face Cache ===")
hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface/hub"))
print("HF Cache path:", hf_home)
if os.path.exists(hf_home):
    for item in os.listdir(hf_home):
        print("  -", item)
else:
    print("  HF cache directory does not exist.")

print("\n=== Checking Workspace for Models / Weights ===")
for root, dirs, files in os.walk("."):
    # skip .git, venvs
    dirs[:] = [d for d in dirs if not d.startswith(".venv") and d not in {".git", "node_modules", "__pycache__"}]
    for f in files:
        if f.endswith((".safetensors", ".bin", ".onnx", ".pt", ".pth")):
            full_p = os.path.join(root, f)
            sz_mb = os.path.getsize(full_p) / (1024 * 1024)
            print(f"  {full_p} ({sz_mb:.1f} MB)")

