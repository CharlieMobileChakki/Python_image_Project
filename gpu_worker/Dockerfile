# =============================================================================
# Production Dockerfile for Arangtik Qwen-Image-2.1-Turbo GPU Inference Worker
# Target Platform: Google Cloud Run with NVIDIA RTX PRO 6000 Blackwell
# Architecture: Linux x86_64, CUDA 12.8 Runtime, Python 3.11, PyTorch >=2.7 (cu128)
# Native Blackwell GPU Compute Architecture: sm_100 / sm_120
# =============================================================================

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8001

# NVIDIA Blackwell CUDA 12.8 Runtime Environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# 1. Install base system utilities and Python 3.11
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

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app

# 2. Install PyTorch >=2.7 with native Blackwell architecture CUDA 12.8 wheels (sm_100 / sm_120)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# 3. Install remaining application dependencies
COPY requirements-gpu.txt /app/requirements-gpu.txt
RUN pip install --no-cache-dir -r requirements-gpu.txt

# 3.5. Pre-cache high-speed SOTA background removal model (RMBG-1.4, 170MB) for ultra-fast GPU matting (<50ms)
RUN python3.11 -c "import torch.nn as nn; nn.Module.all_tied_weights_keys = {}; from transformers import AutoModelForImageSegmentation; AutoModelForImageSegmentation.from_pretrained('briaai/RMBG-1.4', trust_remote_code=True)"

# 4. Copy worker application code (Zero model weights baked into image)
COPY server.py /app/server.py

# Expose microservice port
EXPOSE 8001

# Run FastAPI worker with Uvicorn
CMD ["python3.11", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
