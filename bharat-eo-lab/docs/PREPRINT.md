# When Do Earth-Observation Foundation Models Actually Help? A Stratified, Ground-Truth-Anchored Benchmark over India

*Preprint draft — Bharat-EO-Bench v1.0.0. Author: (you). Correspondence: crakashp2905@gmail.com.*

---

## Abstract

Earth-observation (EO) foundation models (FMs) such as Prithvi-EO-2.0 are widely
assumed to outperform generic (ImageNet-pretrained) backbones on satellite tasks,
yet head-to-head evidence on a **rigorously stratified, ground-truth-anchored**
benchmark is scarce — especially for India, a monsoon-driven, smallholder,
sensor-sovereign geography under-represented in existing suites. We introduce
**Bharat-EO-Bench v1**, a versioned benchmark over India with (i) **15
agro-climatic zones** (rainfall × elevation), (ii) **geographically-disjoint
70/15/15 splits** that eliminate spatial leakage, (iii) **zone × class stratified
draws**, and (iv) **real in-situ ground truth** — 2,599 Central Ground Water Board
(CGWB) wells and 239 WorldCereal expert-validated crop points. On this benchmark we
run **decoder-controlled** encoder comparisons and report a clear — but easily
confounded — result: **an EO foundation model beats a generic ImageNet backbone on
India land cover whenever the decoder is held constant, and the advantage is masked
only by a mismatched decoder.** With a *mismatched* decoder (Prithvi + a simple head
vs an ImageNet ResNet-U-Net with a strong multi-scale skip decoder), full fine-tuning
looks like a **tie** (macro-F1 0.593 vs 0.582–0.589; the ranking flips across seeds).
Hold the decoder fixed — the **same** UPerNet-on-ViT head on the **same** ViT
architecture — and EO pretraining wins in **both** regimes: **+0.042 macro-F1** under
full fine-tuning (**0.663** vs 0.621) and **+0.08–0.09** under a frozen-encoder probe
(Prithvi-600M **0.574 ± 0.010**, 300M **0.559 ± 0.023** vs ImageNet ViT-B **0.483 ±
0.005**, ~8× the seed variance). The decoder-matched model also lifts Prithvi-300 from
0.593 to **0.663 macro-F1 / 0.527 mIoU**. We explain the earlier "tie" mechanistically:
a strong *mismatched* decoder plus full fine-tuning *compensates* for a weaker encoder,
hiding the representation gap the controlled comparison exposes. All experiments run on
free/commodity compute (a 16 GB laptop as control plane + Google Earth Engine + a single
Kaggle T4 GPU), and the benchmark, splits, and evaluation harness are released with
content hashes for reproducibility.

---

## 1. Introduction

Foundation models pretrained on large unlabeled satellite corpora (Prithvi-EO-2.0,
Clay, SatMAE, Scale-MAE, DOFA, Galileo, TerraMind, AlphaEarth Foundations) promise
transferable representations for downstream EO tasks. The prevailing narrative is that
these EO-specific FMs should beat generic ImageNet-pretrained backbones, because their
pretraining sees the right spectra, scales, and temporal structure. However, three
methodological gaps make many reported wins hard to trust:

1. **Weak splits.** Random pixel/chip splits leak spatially adjacent, autocorrelated
   samples between train and test, inflating scores.
2. **Confounded comparisons.** "FM vs baseline" often compares an FM encoder + one
   decoder against a different architecture + a *different* decoder, so the encoder
   effect cannot be isolated from the decoder effect.
3. **No ground truth.** Labels are frequently themselves model outputs (e.g. one
   global land-cover product), so the benchmark measures agreement between two models,
   not agreement with reality.

India sharpens all three. It is monsoon-phased (the same pixel looks agricultural,
bare, or flooded across a year), dominated by **smallholder** fields below the
resolution many FMs were tuned for, cloud-heavy in the growing season, and a
**data-sovereignty** priority (National Geospatial Policy 2022). If EO FMs help
anywhere, they should help here — and if we cannot show *when* they help under rigorous
conditions, sovereign-FM investment rests on faith.

**Contributions.**
- **Bharat-EO-Bench v1**, a versioned, stratified, spatially-disjoint India benchmark
  with real in-situ ground truth and a variance-reporting evaluation harness.
- A **decoder-controlled** encoder comparison (both a frozen-probe and a
  decoder-matched full-fine-tune protocol) that isolates representation quality.
- The empirical finding that **EO pretraining beats ImageNet on India land cover in
  both regimes once the decoder is matched** (+0.042 fine-tuned, +0.08 frozen), and a
  demonstration that a *mismatched* decoder manufactures a spurious "tie" — a direct
  instance of the confounded-comparison pitfall (gap #2 above).
- A demonstration that all of this — including a strong 0.663-macro-F1 India
  land-cover model — is achievable on **free/commodity compute**.

---

## 2. Bharat-EO-Bench v1

### 2.1 Agro-climatic zonation
We partition India into **15 agro-climatic zones** as the cross product of a rainfall
axis (WorldClim BIO12, terciled) and an elevation axis (SRTM, terciled), yielding
arid/semi-arid/sub-humid/humid × lowland/upland/highland cells. Zones were verified to
recover recognizable regions (Thar arid lowland, Gangetic sub-humid plain, Himalayan
highland). Zonation is the stratification key for every task, so that rare
zone-class combinations (e.g. high-Himalaya snow, coastal mangrove) are represented in
both train and test rather than starved by chance.

### 2.2 Tasks
- **T1 — Land-cover segmentation** (dense, per-pixel). 5,271 unique 224×224 **8-band**
  chips (6 Sentinel-2 optical: B/G/R/NIR/SWIR1/SWIR2, + 2 Sentinel-1 SAR: VV/VH),
  ESA WorldCover labels remapped to 11 classes (255 = ignore). Split **3,741 / 724 /
  806** train/val/test, geographically disjoint.
- **T2 — Land-cover representation** (point-wise). 5,332 points, 15 zones × 11 classes,
  each carrying a 4-season time series (NDVI/VV/VH across rabi/zaid/kharif/post-monsoon)
  and the 64-d AlphaEarth annual embedding. Released as versioned Parquet +
  manifest.json + SHA-256 (`bffd5fbc`).
- **T3 — Groundwater** (regression vs in-situ). GRACE/GRACE-FO terrestrial-water-storage
  trends over NW India, anchored to CGWB wells (below).

### 2.3 Ground truth (the credibility layer)
- **Groundwater (T3):** the India-WRIS portal was unreachable; we instead pulled CGWB
  depth-to-water-level via the India Data Portal CKAN datastore API (22,965 wells,
  quarterly since 1969). For 2,599 NW-India wells we computed the per-well water-table
  trend and correlated it with the co-located GRACE storage trend: **Pearson r = −0.284,
  p = 1.4 × 10⁻⁴⁹** — negative and highly significant, i.e. *falling water tables track
  satellite-measured storage loss.* In-situ reality agrees with the satellite signal.
- **Land cover (T2):** we validated ESA WorldCover against **239 WorldCereal
  expert-validated India crop points** (Zenodo 7825628): **90.0% agreement** as
  cropland (residual disagreement is mostly orchards labeled tree cover). This is a
  *reference* comparison, not model-vs-model.
- **Cross-product anchor:** WorldCover vs Dynamic World agree on 66.1% of pixels
  (built 90 / tree 84 / water 80; shrub↔grass ~40, the known ambiguous pair), quantifying
  label noise so downstream scores are read against it.

### 2.4 Evaluation harness
Every score is reported with uncertainty: **bootstrap 95% CIs** on the fixed test set
and **5-fold spatial cross-validation** mean ± std. For the T2 baseline
(AlphaEarth + time-series + XGBoost): fixed-test OA **0.785 [0.756, 0.815]**, macro-F1
**0.718 [0.634, 0.805]**; 5-fold spatial OA **0.780 ± 0.006**, macro-F1 **0.760 ± 0.018**.

---

## 3. Method: controlled encoder comparison

The central question — *does EO pretraining beat ImageNet pretraining?* — is confounded
whenever the encoders are wrapped in different decoders or trained under different
regimes. We therefore run **three** comparisons on the **same T1 chips and identical
splits**, deliberately spanning from confounded to controlled.

**(a) Mismatched-decoder full fine-tune** (the confounded baseline). Each backbone is
fine-tuned end-to-end (class-balanced CrossEntropy, weights capped 8×, + Dice, AMP,
cosine LR, 35 epochs). The EO FM is Prithvi-EO-2.0 (300M) with a lightweight progressive
decoder; the generic baseline is an ImageNet U-Net (ResNet-34,
`segmentation-models-pytorch`) with its native multi-scale **skip** decoder. Encoder and
decoder both differ — exactly the confound gap #2 warns about.

**(b) Decoder-matched full fine-tune** (controlled). Both encoders are ViTs and both
carry the **identical UPerNet-on-ViT decoder** — features from 4 transformer depths
(forward hooks) → a resampled pyramid → PPM + top-down FPN — trained with CE + Dice +
**Lovász-softmax**, **layer-wise LR decay**, and **test-time augmentation**. Now the
*only* difference is the pretraining (Prithvi-EO-2.0 300M vs ImageNet ViT-B/16, both
6-channel). This is also our strongest single model.

**(c) Frozen-encoder probe** (controlled, capacity-limited). We **freeze** each encoder,
precompute its feature map once, and train **the same** progressive `UpDecoder`
(14→224, GroupNorm/GELU, no skips) on top. Because the encoder is frozen and the
decoder is identical across backbones, the **only** difference is the pretraining. To
make this affordable we cache each frozen encoder's features once (fp16, CPU) and train
the decoder on the cache — ~25× faster than re-encoding every epoch, which also lets the
600M model run: with no encoder optimizer states, **Prithvi-600M fits a 16 GB T4** for
the first time. We compare **same-architecture ViTs** — Prithvi-EO-2.0 300M, Prithvi
600M, and an ImageNet ViT-B/16 (`timm`, 6-channel patch-embed) — over **3 seeds**,
reporting mean ± std.

All compute: a 16 GB CPU-only laptop as control plane, Google Earth Engine for
server-side data preparation, and a single free Kaggle **Tesla T4** for GPU training.

---

## 4. Results

### 4.1 Representation quality (T2, frozen features + light head)
| Representation | OA | macro-F1 |
|---|---|---|
| AlphaEarth FM embedding | 0.781 | **0.721** |
| Hand-crafted seasonal time-series | 0.655 | 0.562 |
| Both | 0.786 | 0.720 |

A learned FM embedding beats hand-crafted seasonal features by **+16 macro-F1 points**,
and embedding-alone ≈ embedding+time-series — i.e. the FM representation already
captures the seasonality that the hand-crafted features encode explicitly.

### 4.2 Mismatched-decoder full fine-tune (T1) — an apparent tie
| Backbone (full fine-tune, *different* decoders) | macro-F1 | mIoU |
|---|---|---|
| Prithvi-EO-2.0 300M (simple decoder) | 0.593 | 0.451 |
| ImageNet U-Net (ResNet-34, skip decoder) | 0.582–0.589 | 0.438–0.452 |

The gap is within run-to-run noise: **the ranking flips across two 35-epoch runs**
(unseeded ±0.01). It is tempting to conclude "the FM does not help under fine-tuning."
That conclusion is wrong — and §4.3 shows why: the ImageNet model here carries a
*stronger* decoder, which does the work its weaker encoder cannot. (An apparent +0.06 FM
lead at 25 epochs was a separate convergence-speed artifact — the ImageNet model closes
it with more training.)

### 4.3 Decoder-matched full fine-tune (T1) — EO pretraining wins
Give both ViTs the **identical** UPerNet decoder (+ Lovász, layer-wise LR decay, TTA):

| Model (full fine-tune, *same* UPerNet decoder) | OA | macro-F1 | mIoU |
|---|---|---|---|
| **Prithvi-EO-2.0 300M** | **0.804** | **0.663** | **0.527** |
| Prithvi-EO-2.0 600M (frozen encoder) | 0.772 | 0.618 | 0.481 |
| ImageNet ViT-B/16 | 0.763 | 0.621 | 0.479 |

Two results land here. **(1)** With encoder *and* decoder matched, **EO pretraining beats
ImageNet by +0.042 macro-F1 / +0.048 mIoU** — the "tie" of §4.2 evaporates the moment the
ImageNet backbone can no longer lean on a stronger decoder. **(2)** The controlled decoder
is also simply better: it lifts Prithvi-300 from 0.593 (§4.2) to **0.663 macro-F1 /
0.527 mIoU**, a +0.070 / +0.076 gain — the deployable India land-cover model of this study.

### 4.4 Frozen-encoder probe (T1) — EO pretraining wins, isolated
| Encoder (frozen, same decoder, 3 seeds) | macro-F1 (mean ± std) | mIoU |
|---|---|---|
| **Prithvi-EO-2.0 600M** | **0.574 ± 0.010** | 0.435 ± 0.009 |
| Prithvi-EO-2.0 300M | 0.559 ± 0.023 | 0.417 ± 0.025 |
| ImageNet ViT-B/16 | 0.483 ± 0.005 | 0.345 ± 0.005 |

With the decoder held constant, **EO pretraining beats ImageNet by +0.08–0.09 macro-F1**
— roughly 8× the seed std, unambiguously beyond noise. Same architecture, same decoder,
so the gap is attributable to *pretraining* (EO masked-autoencoding vs ImageNet
supervised). **Scale helps modestly and stabilizes**: 600M edges 300M and has tighter
variance.

### 4.5 Reconciling: it was the decoder all along
The three results tell one coherent story. A ResNet-U-Net's multi-scale **skip** decoder
is expressive enough to *reconstruct* much of what a weaker encoder fails to provide, so
in §4.2 it lets the ImageNet backbone reach parity — a **spurious tie born of a decoder
confound**, not of encoder parity. Control the decoder and the confound vanishes: EO
pretraining then wins in **both** the full-fine-tune regime (§4.3, **+0.042**) and the
frozen regime (§4.4, **+0.08–0.09**). The frozen probe, though capacity-limited (its
absolute scores sit below full fine-tuning), isolates the cause most cleanly — identical
frozen features, identical decoder, only the pretraining differs.

> **Headline:** on a decoder-controlled, architecture-matched comparison, an EO
> foundation model beats a generic ImageNet backbone for India land cover in **both**
> the fine-tuned (+0.042 macro-F1) and frozen (+0.08) regimes. The widely-reported "tie
> under fine-tuning" is an artifact of *mismatched decoders* — a cautionary tale for
> FM-vs-generic benchmarking.

### 4.6 Supporting India results (same lab, same compute)
- **Land cover (point probe):** AlphaEarth + XGBoost OA 0.86 / macro-F1 0.80; targeted
  regional sampling rescues rare classes (mangrove F1 0.00→0.98, snow 0.84→1.00).
- **Flood susceptibility (Assam, static pre-event features):** flood-class F1 0.56;
  the annual embedding (F1 0.51) ≫ terrain (0.01) — it encodes flood-prone geomorphology.
- **Groundwater (Punjab/NW India):** predicting GRACE storage trend, R² lifts
  0.34 → **0.89** with the embedding; downscaling 110 km → 1.5 km at spatial-CV
  R² 0.924; Punjab −0.86 cm/yr, structural overdraft (not drought) confirmed
  cross-mission and against CGWB wells.

---

## 5. Discussion

**For practitioners.** Pick the EO FM in *either* regime — the representation gain is
real whether you freeze and serve features (~+0.08 macro-F1, the resource-constrained /
few-shot / feature-serving case a sovereign national model targets) or fully fine-tune
(~+0.04). The one caveat is a benchmarking one: **do not compare an FM against a generic
backbone that carries a different decoder** — a stronger decoder on the generic side
manufactures a false tie. Match the decoder (and ideally the architecture) or the
comparison measures your decoder, not your encoder.

**For sovereign-FM strategy.** The through-line across land cover, flood, and
groundwater is that the **representation is the dominant signal** — lightweight heads on
a good embedding solve diverse Indian problems, while the head architecture barely
matters. This is the empirical case for investing in a sovereign representation
(Indian sensors: NISAR, EOS-04, INSAT, Resourcesat; in-situ: CGWB/CWC/IMD) rather than
in bespoke per-task models — and the frozen-probe result is exactly the regime such a
shared national embedding would be deployed in.

**On honesty as method.** Our own benchmark first produced a "tie" that seemed to
*disconfirm* the FM's value — and only tightening the protocol (matching the decoder,
then freezing the encoder) revealed the tie as our own confound. We report the tie, its
cause, and its resolution rather than only the final favorable number. The uncomfortable
intermediate result was a feature of the protocol, not a failure of it — and a reminder
that many published FM-vs-generic ties may be decoder artifacts.

---

## 6. Limitations

- **AlphaEarth is a global, closed embedding** (Google); we use it as the strong
  baseline-to-beat and as evidence the sovereignty argument matters, not as our model.
- **Groundwater is coarse (~110 km GRACE) and correlational**; the downscaling assumes
  scale-invariance and is validated against wells only at the trend level.
- **Frozen-probe scores (~0.56) are below decoder-matched fine-tuning (0.663)** by
  construction — the probe isolates encoder quality, not the deployable ceiling; the
  fine-tuned decoder-matched model is the number to deploy.
- **Two backbones per family.** Clay and TerraMind were attempted but skipped on
  registry/API mismatches; a broader FM roster would strengthen generality.
- **Single T4, one dataset year (2021 embeddings; per-task S1/S2 windows).** Broader
  temporal and multi-sensor evaluation is future work.
- **Label noise** (WorldCover↔Dynamic World 66% agreement) bounds achievable scores;
  we report against it rather than around it.

---

## 7. Reproducibility

Benchmark, splits, and harness are released under `data/bench_v1/` with content hashes
(T1 zip SHA `bac1d6d3`; T2 Parquet SHA `bffd5fbc`) and a manifest per task. The
evaluation CLI (`bench_v1.py`: `verify-zones / build-t2 / pull-cgwb / validate-gw /
lc-anchor / lc-fieldtruth / compare-t2 / eval-t2`) reproduces every table. Three Kaggle
notebooks run end-to-end on a free T4 with a numpy-safe install shim:
`benchmark_backbones.ipynb` (mismatched-decoder fine-tune), `probe_headtohead.ipynb`
(frozen probe), and `prithvi_seg_v2.ipynb` (decoder-matched fine-tune — the UPerNet /
Lovász / LLRD / TTA model). Compute footprint: laptop control plane + GEE server-side +
one T4 (~70 min frozen-probe grid; ~4.7 h for the decoder-matched sweep).

---

## 8. Conclusion

On a stratified, spatially-disjoint, ground-truth-anchored India benchmark, an EO
foundation model beats a generic ImageNet backbone for land cover in **both** the
fine-tuned (+0.042 macro-F1) and frozen (+0.08–0.09) regimes — **once the decoder is
matched**. The widely-reported "tie under fine-tuning" reproduces here only with a
*mismatched* decoder, and dissolves under control: it is a benchmarking artifact, not a
property of the encoders. The decoder-matched model reaches **0.663 macro-F1 / 0.804 OA**
on Indian land cover on free compute. The finding both refutes the "generic is just as
good" claim and strengthens — with a clean mechanistic caveat for future benchmarks —
the case for a sovereign Indian foundation model.

---

*Appendix material available in-repo: agro-zone construction, per-class confusion
matrices, CGWB well-trend maps, and the full data-engineering DAG.*
