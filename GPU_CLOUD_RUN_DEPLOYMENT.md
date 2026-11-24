# Google Cloud Run GPU Deployment Guide
## NVIDIA RTX PRO 6000 Blackwell (96 GB VRAM) — Qwen-Image-2.1-Turbo Worker

This guide provides the complete, step-by-step production procedure to build, push, deploy, and verify the `Qwen/Qwen-Image-2.1-Turbo` GPU inference worker microservice on **Google Cloud Run with NVIDIA RTX PRO 6000 Blackwell GPU** in the `asia-southeast1` (Singapore) region.

---

### Hardware & Architecture Summary

| Component | Specification | Notes |
| :--- | :--- | :--- |
| **Target GPU** | **NVIDIA RTX PRO 6000 Blackwell Server Edition** | 1 full GPU |
| **GPU VRAM** | **96 GB GDDR7** | Fits 30.21 GiB model 100% in VRAM with 65 GB headroom |
| **Compute Architecture** | `sm_100` / `sm_120` | Native Blackwell Tensor Cores |
| **CUDA Toolkit / Runtime** | **CUDA 12.8 (`cu128`)** | Driver R580+ compatible |
| **PyTorch Version** | **PyTorch >= 2.7.0 (cu128)** | Official Blackwell architecture support |
| **Host System CPU** | **20 vCPUs** (Gen2 execution environment) | Mandatory Cloud Run minimum for RTX PRO 6000 |
| **Host System RAM** | **80 GiB RAM** | Mandatory Cloud Run minimum for RTX PRO 6000 |
| **Model Residency** | **100% Resident in CUDA VRAM** (`pipe.to("cuda")`) | Zero PCIe CPU-offload bottleneck |
| **Model Weight Size** | **30.21 GiB** (32.44 GB in `torch.bfloat16`) | Unquantized, original precision preserved |
| **Container Port** | `8001` | Matches FastAPI worker port |
| **Target Region** | `asia-southeast1` (Singapore) | Tier 2 pricing bracket |

---

## 1. Prerequisites & GCP Settings

Ensure the following Google Cloud APIs are enabled in your existing project:

```bash
# Set your GCP project ID placeholder
export PROJECT_ID="<YOUR_GCP_PROJECT_ID>"
export REGION="asia-southeast1"
export REPO_NAME="arangtik-gpu"
export IMAGE_NAME="qwen-gpu-worker"

# Set active project
gcloud config set project ${PROJECT_ID}

# Enable required Google Cloud APIs
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    secretmanager.googleapis.com
```

---

## 2. Artifact Registry Setup

Create a Docker repository in Artifact Registry to store your container images:

```bash
# Create repository in asia-southeast1
gcloud artifacts repositories create ${REPO_NAME} \
    --repository-format=docker \
    --location=${REGION} \
    --description="Docker repository for Qwen-Image-2.1-Turbo GPU inference worker"
```

Configure local Docker authentication if building locally (optional if using Cloud Build):

```bash
gcloud auth configure-docker ${REGION}-docker.pkg.dev
```

---

## 3. Secret Manager Configuration (X-API-Key)

Never hardcode API keys into container images, source code, or build arguments. Store your worker API key in Secret Manager:

```bash
# 1. Create secret in Secret Manager
gcloud secrets create QWEN_API_KEY \
    --replication-policy="automatic"

# 2. Add secret version (replace placeholder with your secure API key)
echo -n "<YOUR_SECURE_GPU_WORKER_API_KEY>" | gcloud secrets versions add QWEN_API_KEY --data-file=-

# 3. Grant the Cloud Run service account access to read this secret
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding QWEN_API_KEY \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

---

## 4. Build and Push with Cloud Build

Submit the build to Google Cloud Build using `gpu_worker/cloudbuild.yaml`:

```bash
gcloud builds submit \
    --config=gpu_worker/cloudbuild.yaml \
    --substitutions=_REGION=${REGION},_REPO_NAME=${REPO_NAME},_IMAGE_NAME=${IMAGE_NAME} \
    .
```

The resulting image will be stored in Artifact Registry at:
```
asia-southeast1-docker.pkg.dev/<YOUR_GCP_PROJECT_ID>/arangtik-gpu/qwen-gpu-worker:latest
```

---

## 5. Cloud Run Deployment Command

Deploy the service to Cloud Run with the attached NVIDIA RTX PRO 6000 GPU:

```bash
gcloud beta run deploy qwen-gpu-worker \
    --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest" \
    --region="${REGION}" \
    --execution-environment="gen2" \
    --cpu="20" \
    --memory="80Gi" \
    --gpu="1" \
    --gpu-type="nvidia-rtx-pro-6000" \
    --no-gpu-zonal-redundancy \
    --no-cpu-throttling \
    --concurrency="8" \
    --max-instances="1" \
    --min-instances="0" \
    --timeout="600" \
    --port="8001" \
    --set-env-vars="QWEN_MODEL_ID=Qwen/Qwen-Image-2.1-Turbo,PRELOAD_MODEL=true,QWEN_OUTPUT_WIDTH=896,QWEN_OUTPUT_HEIGHT=896,GPU_BATCH_SIZE=1,GPU_BATCH_MAX_WAIT_MS=300" \
    --set-secrets="QWEN_API_KEY=QWEN_API_KEY:latest" \
    --allow-unauthenticated
```

---

## 6. Comprehensive Pricing & Instance Lifecycle Cost Analysis

Google Cloud Run enforces **instance-based billing** whenever a GPU is attached. Billing occurs per second for the **entire duration the container instance is running**, which includes container startup, model loading, active inference, and post-request idle keep-alive time.

### A. Official Rate Breakdown (`asia-southeast1`, Tier 2)

| Resource | Official Rate | Allocated Quantity | Hourly Cost |
| :--- | :--- | :---: | :---: |
| **NVIDIA RTX PRO 6000 GPU** | `$0.00036522` / second (non-zonal) | 1 GPU | **$1.3148 / hr** |
| **vCPU** | `$0.000018` / vCPU-second | 20 vCPUs | **$1.2960 / hr** |
| **System Memory** | `$0.000002` / GiB-second | 80 GiB | **$0.5760 / hr** |
| **Total Active Running Rate** | **`$0.00088522` / second** | — | **$3.1868 / hr** (~$3.19/hr) |

*(If zonal redundancy is enabled, the GPU rate is $0.00056913/sec, making the total active rate ~$3.92/hr).*

---

### B. Understanding the Lifecycle Timing Components

1. **Cold-Start & Model Loading Duration (Estimated: 35 – 55 seconds):**
   - Container scheduling and driver binding: ~5 seconds.
   - Python startup and model instantiation into VRAM (30.21 GiB weights): ~30 to 50 seconds.
2. **Inference Duration (Estimated vs Measured):**
   - *Measured on RTX A6000 (Ampere):* **7.65 seconds** per image.
   - *Estimated on RTX PRO 6000 (Blackwell):* **5.0 to 8.0 seconds** per image (subject to actual measurement after deployment).
3. **Post-Request Idle Keep-Alive (Estimated: 10 – 15 minutes / ~600 – 900 seconds):**
   - Cloud Run does **not** instantly terminate a container when an inference request finishes.
   - It maintains the container in an idle state for roughly **10 to 15 minutes** to absorb subsequent requests without cold-start penalties.
   - **Cost of 10 minutes idle keep-alive:** $600\text{ seconds} \times \$0.00088522 \approx \mathbf{\$0.531}$.

---

### C. Monthly Cost Estimates for 100, 500, and 1,000 Images

#### Mode 1: Scale-to-Zero (`min-instances=0`)

*Actual cost in scale-to-zero depends heavily on whether requests arrive isolated or in clustered sessions.*

| Monthly Volume | Pattern | Active Lifecycle Calculation | Estimated Monthly Cost |
| :--- | :--- | :--- | :--- |
| **100 Images** | **Isolated** (100 separate cold starts) | $100 \times (45\text{s boot} + 8\text{s inf} + 600\text{s idle}) = 65,300\text{s}$ | **~$57.80 / month** |
| **100 Images** | **Clustered** (10 sessions $\times$ 10 images) | $10 \times (45\text{s boot} + 80\text{s inf} + 600\text{s idle}) = 7,250\text{s}$ | **~$6.42 / month** |
| **500 Images** | **Clustered** (25 sessions $\times$ 20 images) | $25 \times (45\text{s boot} + 160\text{s inf} + 600\text{s idle}) = 20,125\text{s}$ | **~$17.81 / month** |
| **1,000 Images**| **Clustered** (50 sessions $\times$ 20 images) | $50 \times (45\text{s boot} + 160\text{s inf} + 600\text{s idle}) = 40,250\text{s}$ | **~$35.63 / month** |

#### Mode 2: Always-On Warm Reservation (`min-instances=1`)

If zero cold starts are required 24/7 (container always resident in memory):
$$\text{Fixed Monthly Cost} = 730\text{ hours} \times \$3.1868\text{/hr} = \mathbf{\$2,326.36 / \text{month}}$$
*(This cost is fixed regardless of whether 0, 100, or 10,000 images are processed).*

---

## 7. Background Job Reliability & In-Memory State Analysis

The GPU worker uses an in-memory dictionary (`_jobs = {}`) and threading queue (`DynamicBatcher`):

### 1. What Happens During Instance Termination?
- Cloud Run sends a `SIGTERM` signal to the container prior to shutting it down (giving up to 10 seconds or the configured grace period).
- Because `_jobs` is held in container RAM, any jobs currently marked `queued` or `running` are terminated immediately.
- Any completed images in `_jobs[job_id]["png_bytes"]` that have not yet been fetched by the client via `GET /jobs/{id}/result` are permanently lost.

### 2. What Happens If the Process Crashes During Generation?
- If the Python process crashes (e.g., unhandled exception or driver crash), Cloud Run restarts the container instance.
- The restarted container starts with a fresh, empty `_jobs = {}` dictionary.
- When the client polls `GET /jobs/{id}`, the new container responds with `404 Not Found` (`{"detail": "Job not found."}`).

### 3. Is `min-instances=0` Safe for In-Memory Architecture?
- **Safe with Caveats:**
  - When a user submits `POST /jobs`, the container is active and Cloud Run will keep it running through the inference and the post-request idle keep-alive window (~10 minutes).
  - Because `qwen_client.py` polls every 1.0 second and fetches the result within seconds of completion, the result is fetched long before the 10-minute idle window expires.
  - **Risk:** Cold starts take 35–55 seconds. If the client timeout (`QWEN_TIMEOUT_SECONDS`) is set too low (e.g. < 90s), the client could time out while the container is booting. Setting `QWEN_TIMEOUT_SECONDS=300` on the VPS mitigates this.

### 4. Is `concurrency=8` Suitable for Job Polling?
- **Yes, it is essential:**
  - `concurrency` controls how many concurrent in-flight HTTP requests Cloud Run directs to the container.
  - While the heavy image inference executes in a background worker thread protected by `_pipe_lock`, the HTTP server must remain responsive to handle incoming `GET /jobs/{id}` status polls and `GET /health` probes.
  - Setting `concurrency=8` ensures polling requests are answered immediately in milliseconds without queuing or timing out.

### 5. Can the Existing VPS Client Recover from Lost Jobs?
- In `app/core/flow_b/qwen_client.py`:
  - If a job returns `404 Not Found`, the client catches the error, logs it, and raises `RuntimeError("GPU Worker status check failed (404)")`.
  - The client does **not** automatically re-submit a lost job to avoid unwanted duplicate processing.
  - The failure is surfaced cleanly to the main FastAPI backend without hanging or data corruption, and the user can re-trigger generation from the UI.

---

## 8. Verification and Health Testing

Once deployed, retrieve the Cloud Run service URL:

```bash
export SERVICE_URL=$(gcloud run services describe qwen-gpu-worker --region=${REGION} --format="value(status.url)")
echo "Cloud Run Service URL: ${SERVICE_URL}"
```

### A. Health & GPU Hardware Check

```bash
curl -s -X GET "${SERVICE_URL}/health" \
    -H "X-API-Key: <YOUR_SECURE_GPU_WORKER_API_KEY>" | jq .
```

**Expected Response Payload:**
```json
{
  "status": "ok",
  "model_id": "Qwen/Qwen-Image-2.1-Turbo",
  "precision": "full_bfloat16",
  "model_loaded": true,
  "output_size": "896x896",
  "batch_size": 1,
  "batch_max_wait_ms": 300.0,
  "cuda_available": true,
  "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
  "vram_total_gb": 96.0,
  "vram_used_gb": 33.4
}
```

Verify that:
- `cuda_available` is `true`.
- `gpu_name` identifies the NVIDIA RTX PRO 6000 Blackwell.
- `vram_total_gb` reflects 96 GB.
- `vram_used_gb` shows ~33.4 GB (model resident in VRAM).

### B. End-to-End Image Generation Test

1. **Submit Job:**
```bash
curl -s -X POST "${SERVICE_URL}/jobs" \
    -H "X-API-Key: <YOUR_SECURE_GPU_WORKER_API_KEY>" \
    -F "image=@test_garment.jpg" \
    -F "prompt=Generate an empty 3D ghost-mannequin studio catalog photograph showing exclusively the standalone royal silk saree." \
    -F "steps=8" \
    -F "guidance_scale=1.0" \
    -F "seed=42" | jq .
```

2. **Poll Status:**
```bash
curl -s -X GET "${SERVICE_URL}/jobs/<JOB_ID>" \
    -H "X-API-Key: <YOUR_SECURE_GPU_WORKER_API_KEY>" | jq .
```

3. **Fetch Result Image:**
```bash
curl -s -X GET "${SERVICE_URL}/jobs/<JOB_ID>/result" \
    -H "X-API-Key: <YOUR_SECURE_GPU_WORKER_API_KEY>" \
    --output output_cutout.png
```

---

## 9. Connecting Existing Contabo VPS Backend

To connect your existing FastAPI backend on Contabo (`https://bg.arangtik.com`) to Cloud Run:

1. SSH into the Contabo VPS (`157.173.124.137`):
```bash
ssh root@157.173.124.137
```

2. Edit `/root/Python_image_Project/.env` (or local `.env`):
```bash
# Update remote GPU URL to the Cloud Run HTTPS service URL
QWEN_REMOTE_URL="https://qwen-gpu-worker-xxxx-as.a.run.app"
QWEN_API_KEY="<YOUR_SECURE_GPU_WORKER_API_KEY>"
QWEN_TIMEOUT_SECONDS=300
```

3. Restart the backend service.

---

## 10. Rollback Instructions

To revert to your previous setup:
1. Restore `QWEN_REMOTE_URL="http://127.0.0.1:8001"` in the Contabo `.env` file and restart the backend.
2. Delete the Cloud Run service to immediately stop all billing:
```bash
gcloud run services delete qwen-gpu-worker --region=${REGION} --quiet
```
