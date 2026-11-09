# Bharat-EarthFM — Full Pretraining Recipe

Concrete, end-to-end training spec for the flagship **Bharat-EarthFM-L** (1.3 B
active / 3.8 B total, MoE). Numbers are the recommended starting configuration;
all are tunable. Compute anchor: Prithvi-EO-2.0-600M ≈ 58k GPU-hours for
~3.6×10²¹ FLOPs (≈15–25 effective TFLOPS/GPU is normal for EO).

---

## 1. Model configuration

| | S (warm-start) | **L (flagship)** | XL (reserve) |
|---|---|---|---|
| d_model | 1024 | **1408** | 1792 |
| layers | 24 | **40** | 48 |
| heads (dim) | 16 (64) | **16 (88)** | 16 (112) |
| FFN | dense GeGLU ×4 | **MoE: 8 exp, top-2, alt layers** | MoE: 16 exp, top-2 |
| active / total params | 300 M | **1.3 B / 3.8 B** | 3 B / 12 B |
| init | Prithvi-EO-2.0 ViT-L | warm-start S optical branch | scale-up from L |

Shared: pre-norm, RMSNorm, QK-norm, FlashAttention-3, rotary positions, GeGLU.

### 1.1 Dynamic patch tokenizer
- Per-channel conditioning vector `c` (≈80-d): `[λ_center (log-norm), bandwidth,
  polarization one-hot {none,VV,VH,HH,HV}, GSD (log), sensor-id embed (32-d),
  incidence-angle, orbit-flag]`.
- Hypernetwork `H`: 2-layer MLP `c → 512 → (p·p·d_model)` generates the conv-stem
  weights → admits **any** sensor (S2/Landsat/AWiFS/LISS, S1/EOS-04/NISAR L&S,
  INSAT geostationary) with no per-sensor retraining. (Patent **P1**.)
- Patch 16×16 at 10 m reference GSD; multi-scale 8/16/32 → Laplacian-pyramid
  tokens (field-scale branch, Patent **P6**).

### 1.2 Positional / conditioning encodings (added to tokens)
- **Space:** 2-D RoPE scaled by GSD.
- **Time (MPATE, Patent P2):** `f(date) ⊕ f(monsoon-phase∈{pre,onset,active,
  break,withdrawal}) ⊕ f(agro-day: kharif/rabi/zaid index)`.
- **Spectral/sensor:** learned embedding of `c`. **Geo:** Fourier(lat,lon, 16 freq).

### 1.3 Sequence construction
- Sample = cube `224×224 (=14×14 patches) × T timesteps × G channel-groups`.
- Groups G: {optical-VNIR, optical-SWIR, SAR-C, SAR-L, geostationary, weather,
  static(DEM/GRACE)}. Typical T=6, G=4 → ~4.7k tokens; max length 8192.
- **Factorized attention:** spatial-windowed → temporal (continuous-time,
  gap-aware) → every 8th block global (few summary tokens for basin/monsoon scale).

---

## 2. Five-stage curriculum (objectives, losses, schedules)

| Stage | Steps | Objective (loss = Σ λᵢ·Lᵢ) | LR peak | Notes |
|---|---|---|---|---|
| 1 SSL | 300k | 1.0·MIM + 0.5·TemporalMIM + 0.3·Contrastive(global InfoNCE + local DINO) | 1.5e-4 | masking 0.6–0.75; modality-dropout p=0.3; timestep-dropout p=0.2 |
| 2 Generative | 150k | += 1.0·TiM(cross-modal recon) + 0.3·ContrastiveAlign | 1.0e-4 | decoder D2 on; monsoon-conditioned negatives (P17) |
| 3 Physics+SL | 80k | += 0.7·ΣTaskHeads + 0.3·PhysicsConsistency | enc 2e-5 / head 1e-3 | water/energy/mass-balance soft penalties (P7) |
| 4 Forecast | 120k | += 1.0·ThiT(multi-horizon) + 0.3·PhysicsConsistency | 8e-5 | horizons 7/15/30/60/90 d; latent diffusion or AR (P3) |
| 5 Align | 30k | Indic-GeoLLM connector (encoder frozen) | 5e-5 | instruction-tune NL↔geo (P26) |

**Loss term definitions**
- **MIM:** masked-token reconstruction (pixel L1 + deep-feature cosine), high
  mask ratio; **structured masking** = whole-modality / whole-timestep dropout so
  the model learns to operate when optical is absent (kharif cloud).
- **TemporalMIM:** mask future/intermediate timesteps; predict masked-time tokens.
- **Contrastive:** InfoNCE on global pooled tokens (co-located SAR↔optical↔weather
  positives) + DINO-style local self-distillation (teacher EMA).
- **TiM:** synthesize a masked modality from the others (TerraMind-style), tuned
  for India (SAR+INSAT → optical-equivalent).
- **PhysicsConsistency:** soft penalties for P−ET−Q−ΔS≠0, surface energy-balance
  residual, reservoir mass-balance residual.
- **ThiT:** multi-horizon latent roll-forward; loss = denoising MSE (diffusion) or
  next-step NLL (AR) against future EO + in-situ (CWC/CGWB/IMD), + physics reg.

**Stage data mix (by token share):** Stage 1 = 60% optical / 25% SAR / 10%
geostationary+weather / 5% static; Stage 3–4 weighted toward labeled agro/hydro
regions and active-monsoon phases (agro-stratified sampler, P14).

---

## 3. Optimizer, precision, parallelism

- **Optimizer:** AdamW, β=(0.9, 0.95), weight-decay 0.05, ε=1e-8.
- **Schedule:** cosine decay to 10% peak; linear warmup 10k steps; per-stage LR above.
- **Precision:** bf16 compute, fp32 master weights + optimizer states.
- **Grad clip:** global-norm 1.0. **EMA:** decay 0.9999 (eval uses EMA weights).
- **Batch:** global 4096 sequences (Stage 1), micro-batch sized to fit 80 GB.
- **Parallelism:** FSDP/ZeRO-3 (data) + **expert-parallel** (MoE all-to-all) +
  activation checkpointing; tensor-parallel only at XL. Rail-optimized NDR-400 IB.

### 3.1 MoE settings
- Experts 8 (L) / 16 (XL); **top-2** routing; capacity factor 1.25.
- Aux losses: **load-balance 1e-2**, **router z-loss 1e-3**; expert-dropout 0.1.
- **Modality+region-biased gating** prior (Patent P4): router logits add a learned
  bias from the dominant channel-group and agro-zone of each token.

### 3.2 Stability mitigations
QK-norm; RMSNorm; logit z-loss; grad-clip 1.0; LR warmup; EMA; **loss-spike
rollback** (revert to last good checkpoint if step loss > 1.3× running mean for
>50 steps); deterministic intra-shard order; expert re-initialization on collapse.

---

## 4. Compute & throughput budget (Bharat-EarthFM-L)

FLOPs ≈ 6 · N_active · N_tokens. Token budget ≈ **3 T**; N_active = 1.3 B.

| Stage | token share | core FLOPs | GPU-hours* |
|---|---|---|---|
| 1 SSL | 45% | 1.05×10²² | ~45k |
| 2 Generative | 25% | 0.58×10²² | ~25k |
| 3 Physics+SL | 10% | 0.23×10²² | ~10k |
| 4 Forecast | 15% | 0.35×10²² | ~15k |
| 5 Align | 5% | 0.12×10²² | ~5k |
| **Total** | 100% | **2.3×10²²** | **~100k** (×~4 overhead applied) |

\* at ~25% MFU on H100 (≈2.5×10¹⁴ eff FLOP/s), incl. MoE/restart overhead.
**Wall-clock:** 256 H100 ≈ 16 days; 384 H100 ≈ 11 days. (-S ~15–25k GPU-h; -XL
~450–700k.)

**Data-loader throughput:** 256 H100 @ 25% MFU ⇒ ~8×10⁶ tokens/s; at ~2–5 KB/token
raw ⇒ **~30–50 GB/s sustained read** → parallel FS ≥200–500 GB/s aggregate + NVMe
burst buffer; data as shuffled WebDataset shards (see DATA_ENGINEERING_DAG.md).

---

## 5. Evaluation during training
- **Bharat-EO-Bench** every 5k steps: linear-probe + few-shot on crop-type,
  flood, drought (SPI/SPEI), groundwater-anomaly, reservoir-level, LULC, urban,
  coastal, GLOF. Track per-domain macro-F1 / R².
- Baselines to beat: Prithvi-EO-2.0, Clay, TerraMind, Galileo, AlphaEarth embeddings.
- Promote a checkpoint only if it improves the weighted per-domain mean without
  regressing any single domain >2%.

---

## 6. Pre-flight checklist
1. Corpus: ~0.75 PB ARD landed; WebDataset shards built (DATA_ENGINEERING_DAG.md).
2. Warm-start: Prithvi-EO-2.0 weights mapped into -S; -S optical branch → -L init.
3. Compute: 256–384 H100 on AIRAWAT/PARAM/IndiaAI; NDR-400 fabric verified.
4. Storage: ≥1.5 PB hot parallel-FS @ ≥200 GB/s benchmarked.
5. Code: TerraTorch fork (`bharat-eo`) with dynamic tokenizer, MPATE, MoE,
   TiM/ThiT decoders, physics losses, Bharat-EO-Bench eval hooks.
