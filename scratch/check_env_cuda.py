import os
import sys
import subprocess
import shutil

print("=== Python Executable ===")
print("Current:", sys.executable)

venvs = [".venv", ".venv-flowb", ".venv-inpaint"]
for v in venvs:
    py = os.path.join(v, "Scripts", "python.exe")
    if os.path.exists(py):
        print(f"\n--- Checking {v} ({py}) ---")
        cmd = [
            py, "-c",
            "import torch; "
            "print('PyTorch:', torch.__version__); "
            "print('CUDA available:', torch.cuda.is_available()); "
            "print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); "
            "print('Device name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            print(res.stdout.strip())
            if res.stderr.strip():
                print("stderr:", res.stderr.strip())
        except Exception as e:
            print("Error running torch check:", e)

        # Check diffusers
        cmd_diff = [py, "-c", "import diffusers; print('diffusers:', diffusers.__version__)"]
        try:
            res = subprocess.run(cmd_diff, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                print(res.stdout.strip())
            else:
                print("diffusers: NOT INSTALLED")
        except Exception as e:
            print("diffusers check failed:", e)

        # Check huggingface_hub / accelerate
        cmd_acc = [py, "-c", "import accelerate; print('accelerate:', accelerate.__version__)"]
        try:
            res = subprocess.run(cmd_acc, capture_output=True, text=True, timeout=10)
            if res.returncode == 0:
                print(res.stdout.strip())
            else:
                print("accelerate: NOT INSTALLED")
        except Exception:
            pass

print("\n=== Disk Space ===")
usage = shutil.disk_usage(".")
print(f"Total: {usage.total / (1024**3):.2f} GB")
print(f"Used:  {usage.used / (1024**3):.2f} GB")
print(f"Free:  {usage.free / (1024**3):.2f} GB")

