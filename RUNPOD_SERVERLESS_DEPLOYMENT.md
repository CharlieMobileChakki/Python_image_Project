# RunPod Serverless Deployment Guide
## Qwen/Qwen-Image-2.1-Turbo + BRIA RMBG-1.4 (48 GB Auto-Scaling Worker)

This guide explains how to deploy `gpu_worker/` cleanly onto **RunPod Serverless** directly from GitHub—with zero Google Cloud dependencies, pay-per-second billing (`$0.00` when idle), and instant provider switching via `.env`.

---

## 1. Architecture Overview

- **`gpu_worker/engine.py`**: Single shared inference core (`Qwen/Qwen-Image-2.1-Turbo` full `bfloat16` + `briaai/RMBG-1.4` matting + in-GPU reference pre-clean). Automatically finds weights cached by RunPod at `/runpod-volume/huggingface-cache/hub/...` or local HuggingFace cache.
- **`gpu_worker/rp_handler.py`**: RunPod Serverless Queue worker (`runpod.serverless.start`). Handles `generate`, `remove_background`, and `health` jobs safely without cold-start connection drops (also supports `WORKER_MODE=http` fallback to Uvicorn).
- **`gpu_worker/server.py`**: FastAPI HTTP server (`/generate`, `/jobs`, `/remove-background`, `/health`) sharing `engine.py` for dedicated HTTP Pods/nodes.
- **`app/core/flow_b/qwen_client.py`**: Contabo backend client that auto-detects **RunPod Serverless** (`https://api.runpod.ai/v2/<ENDPOINT_ID>`) vs **HTTP GPU Server(s)** purely from `.env` with zero code changes.

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
     - Select **`48 GB — A6000, A40`** (`$1.22/hr`, `$0.000339/s`, best cost-to-VRAM fit)
     - *(Optional faster tier: **`48 GB — L40, L40S, 6000 Ada`** at `$1.89/hr`)*
     - **Do NOT select 32 GB / 24 GB GPUs** (`Qwen-Image-2.1-Turbo` + `RMBG-1.4` requires `~33–34 GB` VRAM).
   - **Workers Scaling:**
     - **Active Workers (Min):** `0` *(no idle cost)*
     - **Max Workers:** `5` *(or `10` for 10x parallel bursts)*
     - **Idle Timeout:** `10` seconds
     - **Execution Timeout:** `300` seconds
   - **Container Disk:** `30 GB` minimum *(Qwen cached weights sit on `/runpod-volume/huggingface-cache`, and RMBG-1.4 is pre-baked in Docker)*.
   - **Cached Models (Important!):** Add **`Qwen/Qwen-Image-2.1-Turbo`** in the **Model** field so RunPod caches the 30.2 GiB weights on host NVMe instead of billing you to download them every cold start.
4. Click **Deploy Endpoint** and wait for the GitHub Docker build + worker cache warm-up.

---

## 3. Connect Contabo VPS (`bg.arangtik.com`)

Copy your **Endpoint ID** from RunPod and your **RunPod API Key** (**Settings → API Keys**), then update `.env` on Contabo:

```env
MAX_CONCURRENT_JOBS=auto
QWEN_REMOTE_URL="https://api.runpod.ai/v2/<YOUR_ENDPOINT_ID>"
RUNPOD_API_KEY="rpa_xxxxxxxxxxxxxxxxxxxx"
RUNPOD_MAX_WORKERS=5
QWEN_TIMEOUT_SECONDS=300
```

Restart Uvicorn on Contabo (`systemctl restart ...` or your process manager), then verify:
```bash
curl -s http://127.0.0.1:8000/health
```

---

## 4. Switching Providers Anytime (Zero Code Edits)

Because `app/core/flow_b/qwen_client.py` auto-detects the backend from `QWEN_REMOTE_URL`, you can switch setups in seconds purely in `.env`:

- **Back to Dedicated Pod / Single HTTP Server (`port 8001`):**
  ```env
  QWEN_REMOTE_URL="http://<SERVER_IP>:8001"
  QWEN_API_KEY="your-worker-key"
  ```
- **Multi-Server HTTP Cluster (Automatic Round-Robin + Failover):**
  ```env
  QWEN_REMOTE_URL="http://<IP_1>:8001,http://<IP_2>:8001"
  ```
- **Blackwell Cards (`B200` / `RTX PRO 6000` / `RTX 5090`):** Pass Docker build args `--build-arg CUDA_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128` without editing any source file.
