# =============================================================================
# Production Dockerfile for Arangtik Qwen-Image-2.1-Turbo GPU Worker
# Supports RunPod Serverless (Queue) and Standard HTTP GPU Servers (Pods / Dedicated)
#
# Built from repository root:
#   docker build -f gpu_worker/Dockerfile .
#
# Default CUDA 12.4 matches RunPod 48GB A6000 / A40 / L40S / A100 / H100 clusters.
# For Blackwell (RTX 5090 / PRO 6000 / B200), pass:
#   --build-arg CUDA_IMAGE=nvidia/cuda:12.8.0-runtime-ubuntu22.04
#   --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128
# =============================================================================

ARG CUDA_IMAGE=nvidia/cuda:12.4.1-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8001

ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV CUDA_MODULE_LOADING="LAZY"

# 1. System packages & Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    ca-certificates \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.org/get-pip.py | python3.11

WORKDIR /app

# 2. Install CUDA PyTorch wheels
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url ${TORCH_INDEX}

# 3. Install GPU worker dependencies (including RunPod Serverless SDK + FastAPI)
COPY gpu_worker/requirements-gpu.txt /app/requirements-gpu.txt
RUN pip install --no-cache-dir -r /app/requirements-gpu.txt

# 4. Pre-bake lightweight RMBG-1.4 (~170MB) so startup matting never stalls on download
# Qwen-Image-2.1-Turbo (~30.2 GiB) is loaded dynamically from RunPod Cached Models or HF cache
RUN python3.11 -c "import torch.nn as nn; nn.Module.all_tied_weights_keys = {}; from transformers import AutoModelForImageSegmentation; AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-1.4', trust_remote_code=True)"

# 5. Copy worker modules
COPY gpu_worker/engine.py /app/engine.py
COPY gpu_worker/rp_handler.py /app/rp_handler.py
COPY gpu_worker/server.py /app/server.py

EXPOSE 8001

# RunPod Serverless Queue handler (set WORKER_MODE=http for standalone FastAPI server)
CMD ["python3.11", "-u", "/app/rp_handler.py"]
