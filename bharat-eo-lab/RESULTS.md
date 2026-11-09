# Bharat-EO-Lab — Empirical Results

Five Indian geospatial tasks, all run on a 16 GB CPU-only laptop + free Google
Earth Engine, using **AlphaEarth foundation-model embeddings + lightweight
heads**. Every benchmark uses a **leakage-free spatial-block split**.

| # | Task | Region | Best result | Headline finding |
|---|------|--------|-------------|------------------|
| 1 | Land cover (11 classes) | All India | **0.86 OA / 0.80 macro-F1** | embedding ≫ geography (0.80 vs 0.23 macro-F1) |
| 2 | Rare-class targeted sampling | All India | mangroves **0.00→0.98**, moss 0.27→0.98 | real targeted data > synthetic SMOTE |
| 3 | Deep model vs gradient boosting | All India | XGBoost 0.80 vs MLP 0.78 macro-F1 | head barely matters; the embedding does the work |
| 4 | Flood susceptibility | Assam (Brahmaputra 2022) | flood-class **F1 0.56** | embedding alone F1 0.51 ≫ terrain 0.01 |
| 5 | Groundwater depletion (GRACE) | Punjab / NW India | Punjab **−0.86 cm/yr**; regression R² **0.89** | embedding lifts R² 0.34→0.89 |
| 5b| GRACE downscaling (110 km→1.5 km) | Punjab + Haryana | downscaling R² **0.924** | district hotspot ranking (Rohtak, Sonepat…) |
| 5c| Depletion validation (independent) | Punjab / NW India | GRACE vs GRACE-FO agree (r 0.70) | structural overdraft, not drought; −36 cm since 2002 |
| 6 | **Prithvi-EO-2.0 fine-tune (real FM)** | India (150 S2 chips) | pixel-OA 0.75, macro-F1 0.43, mIoU 0.33 | a real EO foundation model fine-tuned on a free T4 GPU |
| 6b| **Targeted-sampling fine-tune** | India (1271 chips: 1000 random + 271 targeted) | macro-F1 0.575, mIoU 0.431 | targeted regional data lifted rare classes off 0.00 (mangrove 0.86, snow 0.39) — macro-F1 +56% |

## The through-line
Across **agriculture, disaster, and hydrology**, the AlphaEarth embedding was the
dominant signal — beating geography (land cover), terrain (flood), and
rainfall+agriculture (groundwater). This is the empirical case for Bharat-EarthFM:
**invest in the representation; lightweight heads then solve diverse Indian
problems.** A sovereign Indian foundation model trained on Indian sensors
(NISAR, EOS-04, INSAT, Resourcesat) + in-situ (CGWB/CWC/IMD) should beat a
generic global embedding on every one of these.

## Task detail

### 1–3 · Land cover, rare classes, deep model
- 42,045 points (40k random + 2k targeted rare-class), ESA WorldCover labels,
  AlphaEarth-2021 embeddings.
- Leaderboard B0–B4: majority 0.06 → geography 0.23 → embedding+XGBoost **0.80**
  macro-F1; neural-net (sklearn MLP) 0.78.
- Macro-F1 arc: 12k random 0.68 → 40k random 0.69 → **42k targeted 0.80**.
- Rare-class rescue via `RARE_BOXES` targeted sampling: mangroves 0.00→0.98,
  snow 0.84→1.00, moss 0.27→0.98.

### 4 · Assam flood susceptibility
- 12k points; labels from **Sentinel-1 SAR** (cloud-proof) minus JRC permanent
  water → {dry, permanent_water, flood_water} (16.7% flooded).
- Predicted from **static pre-event features only** (non-circular):
  all-static flood-F1 **0.56**; embedding-alone 0.51; terrain-only 0.01.
- The annual embedding encodes the valley's flood-prone geomorphology.

### 5 · Groundwater (GRACE)
- GRACE TWS trend 2003–2016, 1,321 cells. **Punjab −0.86 cm/yr (depletion).**
- Predicting the trend from surface drivers (spatial CV): rainfall-only R² −0.14
  → climate+agri 0.34 (cropland is #2 driver) → **+embedding 0.89**.

### 5b · GRACE downscaling (the InSAR-groundwater task)
- Downscaled GRACE 110 km → **~1.5 km** over Punjab+Haryana; model spatial-CV
  **R² 0.924**, MAE 0.14 cm/yr; 7,134 fine cells.
- **Most-depleted districts:** Rohtak (−1.85), Sonepat (−1.84), Jhajjar (−1.82),
  Faridabad (−1.78), Rewari, Panipat, Jind, Gurgaon… (southern/central Haryana).
- **Real InSAR path:** `notebooks/insar_hyp3_punjab.py` requests Sentinel-1 SLC
  InSAR subsidence from ASF HyP3 (free cloud), to fuse with this downscaling
  (patent concept P4: InSAR + GRACE + wells).

### 5c · Independent validation of the depletion
HyP3 InSAR needs an Earthdata login + ASF cloud time, so the downscaling target
was validated with independent satellites/processing instead:
- **Cross-mission:** Punjab GRACE 2003–16 = −0.79 cm/yr vs **GRACE-FO 2018–24 = −0.84**
  (a different satellite mission), spatial r = 0.70 across India.
- **Structural, not drought:** India-wide TWS *recovered* post-2018 (+1.14 cm/yr,
  wet years) yet Punjab kept declining (−0.84) → groundwater overdraft.
- **Cross-center/method:** CSR/GFZ/JPL agree (r = 1.00); mascon vs spherical-harmonic
  pattern r = 0.81 (SH magnitudes attenuated, use mascon for rates).
- **Decline curve:** −1.68 cm/yr, **~36 cm cumulative loss 2002–2024**; matches
  Rodell/Famiglietti literature. ([gw_validation_punjab_decline.png](outputs/figures/gw_validation_punjab_decline.png))

### 6 · Prithvi-EO-2.0 fine-tune (the real foundation-model step)
- 150 Sentinel-2 6-band chips (224x224) + WorldCover masks; 116 train / 34 val,
  leakage-free spatial split. Fine-tuned the **real Prithvi-EO-2.0 (300M)** EO
  foundation model (TerraTorch, pretrained 1.33 GB weights) + a light decoder, on
  a free Kaggle **Tesla T4**.
- 30 epochs: loss 1.59 -> 0.37; val **pixel-OA 0.747, macro-F1 0.428, mIoU 0.329**.
- This is **dense per-pixel segmentation** from a single S2 mosaic with only 116
  training chips -- a much harder task than the point-wise embedding probe (tasks
  1-3), so the numbers are NOT directly comparable. The model is clearly learning
  (loss falling, OA climbing); macro-F1 is held down by near-absent rare classes
  (water/snow/mangrove <1%, wetland 0% in the chip sample).
- Significance: the full blueprint chain -- India data -> S2 image chips -> fine-tune
  a real EO foundation model on commodity/free GPU -> spatial-split evaluation --
  now runs end-to-end. Scale-up: `export-chips --n 1000`, more epochs, the 600M
  backbone, add Sentinel-1 bands, a UPerNet decoder, class-balanced loss.

### 7 · Bharat-EO-Bench v1 head-to-head (the credibility table)
On the rigorous stratified benchmark (see `data/bench_v1/`, `docs/BENCHMARK_SPEC.md`):
- **T2 representation** (laptop): AlphaEarth **FM embedding macro-F1 0.721** vs hand-crafted
  seasonal time-series **0.562** — the foundation-model representation wins by +16 points.
- **T1 backbones — full fine-tune** (Kaggle T4, 5,271 chips, identical splits, 35 epochs):
  Prithvi-EO-2.0 300M **macro-F1 0.593** vs ImageNet **U-Net (ResNet34) 0.582–0.589** — a
  **TIE**: run-to-run noise (±0.01, unseeded) *flips the ordering* across two 35-epoch runs.
  Under full fine-tuning with a strong multi-scale decoder, a generic ImageNet backbone
  catches up. (The apparent +0.06 FM lead at 25 epochs was a convergence-speed artifact.)
- **T1 encoders — FROZEN-PROBE** (identical frozen encoder + the **SAME simple decoder** on
  all, 3 seeds, 30 epochs — removes the decoder confound and isolates *representation quality*):

  | encoder | macro-F1 (mean±std) | mIoU |
  |---|---|---|
  | Prithvi-EO-2.0 **600M** | **0.574 ± 0.010** | 0.435 ± 0.009 |
  | Prithvi-EO-2.0 300M | 0.559 ± 0.023 | 0.417 ± 0.025 |
  | ImageNet ViT-B | 0.483 ± 0.005 | 0.345 ± 0.005 |

  **EO pretraining beats ImageNet by +0.08–0.09 macro-F1 — ~8× the ±std, not noise.** Same ViT
  architecture + same decoder, so the gap is cleanly attributable to *pretraining* (EO masked-
  autoencoding vs ImageNet supervised). Scale helps modestly and stabilises (600 > 300, tighter
  std). The frozen 600M **finally fits a 16 GB T4** (no encoder optimizer states solved the OOM).
- **Prithvi-Seg v2 — the improvised model, DECODER-MATCHED** (UPerNet-on-ViT decoder from 4
  transformer depths + PPM/FPN, loss CE+Dice+**Lovász**, **layer-wise LR decay**, **TTA**;
  ImageNet ViT gets the *identical* decoder, 35 epochs):

  | model (full fine-tune, same UPerNet) | OA | macro-F1 | mIoU |
  |---|---|---|---|
  | **Prithvi-EO-2.0 300M** | **0.804** | **0.663** | **0.527** |
  | Prithvi-EO-2.0 600M (frozen) | 0.772 | 0.618 | 0.481 |
  | ImageNet ViT-B | 0.763 | 0.621 | 0.479 |

  Two things land here. **(1) The improvised model is far stronger:** Prithvi-300 jumps
  **0.593 → 0.663 macro-F1 (+0.070)** and **0.451 → 0.527 mIoU (+0.076)** over v1 — the UPerNet
  decoder + Lovász + LLRD + TTA earned it. **(2) The v1 "tie" was the decoder confound, confirmed:**
  with the *same* UPerNet on the *same* ViT architecture, **EO pretraining now beats ImageNet by
  +0.042 macro-F1 / +0.048 mIoU even under full fine-tuning** — the ImageNet backbone no longer
  catches up once it can't lean on a stronger decoder. (Clay/TerraMind skipped: registry/API
  mismatch. Soft-vote ensemble 0.660 < best single, dragged by the weaker frozen/ImageNet members.)
- **Synthesis (updated):** once the decoder is **matched**, EO pretraining beats ImageNet on
  India land cover in **both** regimes — frozen probe **+0.08** and full fine-tune **+0.042**
  macro-F1. The only "tie" was v1's *mismatched-decoder* comparison, where a ResNet-U-Net's strong
  decoder masked a weaker encoder. Decoder-controlled, the FM advantage is real and consistent.
- **Prithvi-Seg v3 — improvised model on improvised data (Dataset v2).** The v3 model (aux-stem
  for the 11 extra bands + soft labels from Dynamic World + confidence-masking + copy-paste +
  balanced sampling) on the 942-chip **dsv2** set: **OA 0.869 / macro-F1 0.732 / mIoU 0.614**,
  with strong rare-class IoU (**mangrove 0.74, wetland 0.64, snow 0.63**). ⚠️ **Not directly
  comparable to the 0.663 above:** dsv2 uses a *non-spatial* zone-stratified chip-level split
  (original chip coords were lost, so no leakage-free spatial split), which is easier than the
  bench_v1 spatial split. So 0.732 conflates an easier split + richer data + the improvisations.
  Clean attribution comes only from the v3-vs-v3 **ablation sweep** (`prithvi_seg_v3_ablation.ipynb`,
  each flag toggled on the fixed dsv2 split). For the rigorous benchmark, **0.663 on bench_v1
  spatial split stays the headline number**; dsv2/v3 is an engineering result pending a spatial re-export.
- Ground truth: T3 groundwater vs 2,599 CGWB wells (r=−0.284, p=1e−49); T2 land cover vs
  239 WorldCereal crop points (90% agreement).

## Honest caveats
- AlphaEarth is a **global, closed** embedding (Google) — used here as the
  baseline to beat, and proof the sovereignty argument matters.
- Groundwater results are **coarse (~110 km GRACE) and correlational**; the
  downscaling is **unvalidated against CGWB wells** and assumes scale-invariance.
- Rare-class flood/land-cover F1s reflect performance within sampled regions.
- These are **frozen embeddings + a head**, not a fine-tuned foundation model —
  that step (image chips → Prithvi/Clay on a Kaggle GPU) is staged in
  `notebooks/finetune_gpu.py`.

## Reproduce
```bash
python run.py boost         --project <ee>   # land cover + rare classes (1–3)
python run.py flood         --project <ee>   # Assam flood (4)
python run.py groundwater   --project <ee>   # GRACE depletion (5)
python run.py gw-downscale  --project <ee>   # downscaling + districts (5b)
```
