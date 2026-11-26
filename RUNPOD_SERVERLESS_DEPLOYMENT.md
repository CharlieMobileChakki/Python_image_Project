# RunPod Serverless Deployment Guide
## Pure Qwen/Qwen-Image-2.1-Turbo (`1024×1024` Direct White-BG Studio Worker)

This guide explains how to deploy `gpu_worker/` cleanly onto **RunPod Serverless** directly from GitHub—with zero Google Cloud dependencies, zero background-removal model overhead, pay-per-second billing (`$0.00` when idle), and instant provider switching via `.env`.

---

## 1. Architecture Overview

- **`gpu_worker/engine.py`**: Single shared inference core (`Qwen/Qwen-Image-2.1-Turbo` full `bfloat16`, direct `1024×1024` RGB studio PNG). Automatically detects weights cached by RunPod at `/runpod-volume/huggingface-cache/hub/...` or local HuggingFace cache.
- **`gpu_worker/rp_handler.py`**: RunPod Serverless Queue worker (`runpod.serverless.start`). Handles `generate` and `health` jobs safely without cold-start connection drops (also supports `WORKER_MODE=http` fallback to Uvicorn).
- **`gpu_worker/server.py`**: FastAPI HTTP server (`/generate`, `/jobs`, `/health`) sharing `engine.py` for dedicated HTTP Pods/nodes.
- **`app/core/flow_b/qwen_client.py`**: Backend client that auto-detects **RunPod Serverless** (`https://api.runpod.ai/v2/<ENDPOINT_ID>`) vs **HTTP GPU Server(s)** purely from `.env` with zero code changes.

---

## 2. Deploy on RunPod Console (via GitHub Integration)

1. **Add Wallet Credits:** Load `$10–$25` via **Load credits** in the RunPod dashboard.
2. **Connect GitHub:** Open **RunPod Settings → Connections → Connect GitHub** and authorize access to your repository (`CharlieMobileChakki/Python_image_Project`).
3. **Create Serverless Endpoint:**
   - Go to **Serverless → + New Endpoint → GitHub Repo**.
   - **Repository / Branch:** Select `Python_image_Project` (`main`).
   - **Dockerfile Path:** `gpu_worker/Dockerfile`
   - **Endpoint Type:** **`Queue`**
   - **GPU Configuration:**
     - Select **`Previous generation -> 48 GB`** (`A6000 / A40`, `$1.22/hr`) and optionally **`Latest generation -> 48 GB`** (`L40S / 6000 Ada`) with `Worker priority: Cost`.
     - **Do NOT select 32 GB / 24 GB GPUs** (`Qwen-Image-2.1-Turbo` in `bfloat16` requires `>32 GB` VRAM).
   - **Workers Scaling:**
     - **Active Workers (Min):** `0` *(no idle cost)*
     - **Max Workers:** `1` (or scale up to `5–10` when needed)
     - **Idle Timeout:** `10` seconds
     - **Initialization / Execution Timeout:** `600` seconds
     - **CUDA Version:** `Use minimum CUDA version = 12.4`
   - **Container Disk:** `30 GB` minimum *(Qwen cached weights sit on `/runpod-volume/huggingface-cache`)*.
   - **Cached Models (Important!):** Add **`Qwen/Qwen-Image-2.1-Turbo`** in the **Model** field so RunPod mounts the weights from host NVMe cache.
4. Click **Deploy Endpoint** and wait for the build to complete.

---

## 3. Connect Backend Server (`bg.arangtik.com`)

Copy your **Endpoint ID** and **RunPod API Key** (**Settings → API Keys**) into `.env`:

```env
MAX_CONCURRENT_JOBS=auto
QWEN_REMOTE_URL="https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>"
RUNPOD_API_KEY="rpa_xxxxxxxxxxxxxxxxxxxx"
RUNPOD_MAX_WORKERS=1
QWEN_APPAREL_SIZE=1024
QWEN_TIMEOUT_SECONDS=600
```

Verify health after restart:
```bash
curl -s http://127.0.0.1:8000/health
```

---

## 4. Switching Providers Anytime (Zero Code Edits)

- **Dedicated Pod / Single HTTP Server (`port 8001`):**
  ```env
  QWEN_REMOTE_URL="http://<SERVER_IP>:8001"
  QWEN_API_KEY="your-worker-key"
  ```
- **Multi-Server HTTP Cluster (Automatic Round-Robin + Failover):**
  ```env
  QWEN_REMOTE_URL="http://<IP_1>:8001,http://<IP_2>:8001"
  ```
- **Blackwell Cards (`B200` / `RTX PRO 6000` / `RTX 5090`):** Pass Docker build args `--build-arg CUDA_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128` without editing any source file.
