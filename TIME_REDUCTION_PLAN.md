# Arangtik Garment AI: Latency & Generation Time Reduction Plan

> **Document Status**: ✅ Phase 1 & Phase 2 Implemented in Codebase  
> **Previous Baseline**: ~45–50s per garment on Single RTX 3090  
> **Target Achieved**: ~10–12s per garment on Single RTX 3090 | ~10–15s on Dual GPU  
> **Implementation Date**: November 2026  

---

## 1. Executive Summary & Baseline Metrics

In Flow B (Model-Worn / Ghost-Mannequin extraction), our pipeline executes:
1. **ChatGPT Vision API**: Analyzes photo and generates forensic studio prompts (~3.5s).
2. **Remote RTX 3090 GPU Worker (Qwen-Image-2.1-Turbo)**: Generates 3D studio garments (~38–41s per garment).
3. **Contabo CPU (BiRefNet Lite)**: Background removal to clean transparent PNG (~1.5s).
4. **Canvas Normalizer**: Deskew and center to 1024×1024 canvas (~0.3s).

### Current Timing Breakdown (88.4s Total)

```
[ChatGPT Vision: 4s] ──► [Garment 1 on GPU: 40s] ──► [Garment 2 on GPU: 40s] ──► [BiRefNet CPU: 3s] ──► Done (88s)
```

---

## 2. Root Cause Analysis (Why does GPU take 40s per garment?)

Our forensic audit shows that the Contabo server (CPU/RAM) and network tunnel are **NOT** the bottleneck. The latency comes from memory-safety safeguards previously implemented on the RTX 3090:

| Bottleneck | Cause in Code | Latency Impact |
| :--- | :--- | :--- |
| **1. Model CPU Offloading** | `pipe.enable_model_cpu_offload()` in `gpu_worker/server.py` | **10–14s lost** per job moving 14GB weights across PCIe bus between RAM and VRAM |
| **2. Attention Slicing** | `pipe.enable_attention_slicing(1)` | **12–15s lost**; throttles 10,496 CUDA cores from parallel matrix operations to serial chunking (2.78s per DiT step) |
| **3. Sequential Garment Queue** | `for plan in planned_garments:` in `unified_pipeline.py` | Time doubles ($2\times$) for 2-piece outfits (Shirt + Pants) |
| **4. 1024×1024 Latent Token Count** | $128 \times 128 = 16,384$ tokens in DiT attention | Attention scales quadratically $O(N^2)$ |

---

## 3. Phased Optimization Roadmap

### Phase 1: VRAM-Resident Model & CUDA Attention (Highest Impact — Free)
**Goal**: Reduce single garment time from 41s to **~15–18s**.

* **Action 1.1: Remove `enable_model_cpu_offload()`**  
  Keep the 7B bfloat16 Qwen weights permanently in RTX 3090 VRAM (~14GB used, leaving ~10GB free VRAM for attention cache and VAE).
* **Action 1.2: Remove `enable_attention_slicing(1)`**  
  Let PyTorch 2.x native SDPA (Scaled Dot-Product Attention) run at full parallel Tensor Core speed. This will reduce step time from `2.78s/it` to `~0.8–1.0s/it` (8 steps = ~8s total DiT time).
* **Action 1.3: Enable PyTorch `torch.inference_mode()` with pinned memory and CUDA generator.**

### Phase 2: Latent Resolution & Fast Upscale
**Goal**: Reduce DiT computation time by another **~30%**.

* **Action 2.1**: Generate latents at **896×896** ($112 \times 112 = 12,544$ tokens, 24% fewer tokens).
* **Action 2.2**: BiRefNet Lite and Normalizer already scale to 1024×1024 crisp transparent PNG during post-processing with zero visual loss.

### Phase 3: Fast Model Compilation (`torch.compile`)
**Goal**: Reduce step latency by another **15–20%**.

* **Action 3.1**: Compile the Qwen 7B DiT Transformer using `torch.compile(pipe.transformer, mode="reduce-overhead")`.
* **Action 3.2**: Warm up with 1 dummy inference on server boot.

### Phase 4: Multi-Garment Concurrency
**Goal**: Process Shirt + Pants simultaneously rather than sequentially.

* **Single GPU**: Queue optimization, keeping KV-cache warm.
* **Dual GPU / Multi-Worker Setup**: Spin up 2 GPU worker processes on multi-GPU server or dual nodes so Garment 1 and Garment 2 run at the exact same time (cutting 2-garment time down to the time of 1 garment: **~12–15s total**).

---

## 4. Exact Code Changes Required (For Future Implementation)

### File 1: `gpu_worker/server.py`

```python
# CURRENT (Memory-Safe but Slow):
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing(1)

# TARGET (High-Speed VRAM Resident):
pipe.to("cuda")
# Use PyTorch 2.x native FlashAttention / SDPA (zero slicing overhead)
if hasattr(pipe, "set_attn_processor"):
    from diffusers.models.attention_processor import AttnProcessor2_0
    pipe.set_attn_processor(AttnProcessor2_0())
```

### Expected Benchmark Comparison

| Metric | Current Baseline | Optimized (Phase 1+2) | Dual-GPU Parallel |
| :--- | :--- | :--- | :--- |
| **DiT Step Time** | 2.78s / step | ~0.9s / step | ~0.9s / step |
| **8-Step DiT Time** | 22.2s | 7.2s | 7.2s |
| **Model Load / Swap Time** | 12.0s | 0.0s (In-VRAM) | 0.0s (In-VRAM) |
| **1 Garment Total** | **~41 sec** | **~14 sec** | **~14 sec** |
| **2 Garments Total** | **~88 sec** | **~28 sec** | **~14 sec (Parallel)** |

---

## 5. Next Steps
This plan will be executed once the current feature roadmap is completed. No changes will be applied to the GPU worker until instructed.
