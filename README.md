# Arangtik Garment AI — Direct 1024×1024 Studio Generator

Lightweight FastAPI orchestrator and web UI that pairs **ChatGPT Vision** with a remote **Qwen/Qwen-Image-2.1-Turbo** GPU worker (RunPod Serverless or HTTP GPU servers) to generate clean **`1024×1024` white-background PNG** catalog photos for apparel and jewelry.

## Stack

- **API & Web UI:** Python 3.12 + FastAPI + Uvicorn + HTTPX + OpenCV/Pillow
- **Prompt Generation:** OpenAI Vision (`gpt-4o-mini` / `gpt-4o`)
- **GPU Synthesis:** `Qwen/Qwen-Image-2.1-Turbo` (`bfloat16`) via RunPod Serverless Queue or FastAPI HTTP worker (`gpu_worker/`)

## Quick Setup (Windows / Linux)

```bash
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY, QWEN_REMOTE_URL, RUNPOD_API_KEY in .env
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/` for the Studio UI or `http://127.0.0.1:8000/docs` for Swagger API docs.

## Core Endpoints

- `POST /api/background-removal` — Primary studio generator (`json` response with `url`, or `response_format=image_direct` binary PNG; alias: `/api/generate-garment`)
- `POST /api/qwen-image` — Direct single-image PNG response endpoint
- `POST /api/jewelry-studio` — Single or multi-angle (up to 4 images) jewelry boutique stand generator (`2048×2048`)
- `POST /api/forward-downstream` — Forward generated job PNGs in a single multipart request to an external API
- `GET /api/outputs/{filename}` — Retrieve generated studio PNG
- `GET /health` — Backend and remote GPU cluster health telemetry

## GPU Worker Deployment

See [RUNPOD_SERVERLESS_DEPLOYMENT.md](RUNPOD_SERVERLESS_DEPLOYMENT.md) for RunPod Serverless GitHub deployment and HTTP GPU worker setup.
