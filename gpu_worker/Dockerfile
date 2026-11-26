# =============================================================================
# Slim Production Dockerfile for Arangtik Qwen-Image-2.1-Turbo GPU Worker
# Uses slim official PyTorch 2.6 CUDA 12.4 runtime base (3.09 GB compressed)
# instead of bloated -devel/audio/cv2 stacks for faster RunPod Serverless pulls.
#
# Built from repository root:
#   docker build -f gpu_worker/Dockerfile .
#
# Default CUDA 12.4 matches RunPod 48GB A6000 / A40 / L40S / A100 clusters.
# For Blackwell (RTX 5090 / PRO 6000 / B200), pass:
#   --build-arg BASE_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
# =============================================================================

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8001
ENV CUDA_MODULE_LOADING="LAZY"

WORKDIR /app

# 1. Install minimal git binary for installing latest diffusers cleanly
RUN if ! command -v git >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends git ca-certificates && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# 2. Install slim worker dependencies while pinning base PyTorch/torchvision
COPY gpu_worker/requirements-gpu.txt /app/requirements-gpu.txt
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -c 'import importlib.metadata as m; open("/tmp/c.txt", "w").write("\n".join(p + "==" + m.version(p) for p in ("torch", "torchvision")) + "\n")' && \
    python -m pip install --no-cache-dir -c /tmp/c.txt -r /app/requirements-gpu.txt && \
    python -c 'import jinja2; assert tuple(int(x) for x in jinja2.__version__.split(".")[:3]) >= (3, 1, 6), jinja2.__version__' && \
    rm -f /tmp/c.txt

# 3. Copy pure Qwen-Image-2.1-Turbo worker modules last for fast layer caching
COPY gpu_worker/engine.py /app/engine.py
COPY gpu_worker/rp_handler.py /app/rp_handler.py
COPY gpu_worker/server.py /app/server.py

EXPOSE 8001

# Clear base entrypoint (if overridden via --build-arg) and start Serverless worker directly
HEALTHCHECK NONE
ENTRYPOINT []
CMD ["python", "-u", "/app/rp_handler.py"]
