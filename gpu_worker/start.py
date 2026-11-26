"""Unified GPU worker launcher (auto-selects RunPod Serverless vs HTTP FastAPI server)."""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    mode = os.getenv("WORKER_MODE", "auto").strip().lower()
    is_runpod_serverless = mode == "runpod" or (
        mode == "auto" and bool(os.getenv("RUNPOD_ENDPOINT_ID", "").strip())
    )

    if is_runpod_serverless:
        return subprocess.call([sys.executable, "rp_handler.py"])

    port = os.getenv("PORT", "8001").strip() or "8001"
    return subprocess.call(
        [sys.executable, "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", port]
    )


if __name__ == "__main__":
    raise SystemExit(main())
