# Bharat-EO-Bench: A Spatially-Disjoint India Land-Cover Benchmark, and What a Foundation Model Actually Buys You on Weak Labels

*Preprint draft v2 — supersedes the v1 draft, whose land-cover numbers came from a non-spatial split and are withdrawn (§7). Correspondence: crakashp2905@gmail.com.*

---

## Abstract

Earth-observation foundation models (FMs) are widely assumed to help on India-scale land
cover, but published gains are routinely measured on random splits and against labels that
are themselves model outputs. We build **Bharat-EO-Bench**, a dense land-cover
segmentation benchmark over India with 11 classes, 15 agro-climatic zones, and a
**spatial-block split** (1° blocks; train chips within 25 km of any evaluation chip are
dropped; minimum train-to-evaluation distance ≥ 25 km). Labels are consensus products
(Dynamic World, ESRI, WorldCover/WorldCereal), so we additionally score every model on the
**confident pixels** (≥ 2 label sources agree; 84% of evaluation pixels) and on **all
pixels**, and we report that the choice of evaluation set alone moves mIoU by 0.06–0.08.
Using Prithvi-EO-2.0 300M with a UPerNet head, all numbers mean-of-two-seeds, we find:
(i) **EO pretraining matters a great deal** — fine-tuned pretrained 0.448 mIoU vs
random-initialisation 0.343 (confident pixels; +0.105), and vs a frozen encoder 0.402
(+0.046); on all pixels 0.388 vs 0.300 and 0.343; seed-to-seed spread is < 0.005 in every
arm. (ii) **A bundle of four training additions ("v3": auxiliary-band stem, confidence
masking, soft labels, copy-paste rare-class augmentation) does not reliably help**: +0.017 mIoU on confident pixels but −0.029 on all pixels versus the plain
fine-tuned baseline, and a leave-one-out ablation attributes the confident-pixel gain to
confidence masking, which trades away accuracy on the ambiguous 16% of pixels. (iii)
**Two of eleven classes (wetland, mangrove) are effectively unmeasurable** in the current
data (IoU ≈ 0 in every arm; ~1 chip each). We therefore position this release as a
*protocol and negative-results* contribution: a leakage-controlled split, an evaluation
protocol that exposes label-noise confounds, and a measured value for pretraining — not a
new method.

---

## 1. Introduction

Three methodological gaps make many FM-for-EO claims hard to trust: **weak splits** (random
chip or pixel splits leak spatially autocorrelated neighbours), **confounded comparisons**
(FM-plus-one-decoder against a different architecture and decoder), and **weak labels**
(the reference is itself a model product, so a score is agreement between models). India
sharpens all three — monsoon-phased phenology, smallholder fields, heavy growing-season
cloud, and a data-sovereignty motive for domestic models.

This paper does three things. It provides a split whose leakage can be quoted as a
distance, not asserted. It shows how much an evaluation-set choice under label noise
changes conclusions — including one reversal we made and retracted ourselves (§7). And it
measures, on that split, what pretraining, fine-tuning and a set of commonly proposed
training additions are each worth.

**Contributions.**
1. A spatial-block, zone-stratified India land-cover split with a reported minimum
   train-to-evaluation distance, and chip-level label-agreement metadata.
2. A two-view evaluation protocol (confident pixels / all pixels) and an account of the
   eval-mask confound it removes.
3. A controlled value-of-pretraining measurement: pretrained fine-tune vs random-init vs
   frozen, same architecture, same head, same split, two seeds.
4. A leave-one-out ablation of four training additions, mostly null on this benchmark.
5. An honest rare-class finding: wetland and mangrove cannot be evaluated from this data.

---

## 2. Benchmark

### 2.1 Data
Chips are 224×224 with 17 input bands: 6 Sentinel-2 optical bands (fed to the ViT patch
embedding) and 11 auxiliary bands (fed to a separate stem), exported with Google Earth
Engine. Labels are 11 classes — tree, shrub, grass, crop, built, bare, snow, water,
wetland, mangrove, moss — with 255 as ignore. Sampling is stratified over **15
agro-climatic zones** (rainfall × elevation terciles), 150 chips per zone target, with
additional targeted sampling for rare classes.

### 2.2 Spatial split
Chips are bucketed into 1°×1° blocks; whole blocks are assigned to train/val/test. Train
chips within 25 km of any val/test chip are dropped (never from evaluation, so the
evaluation set is intact and leakage is one-directional). The minimum train-to-evaluation
distance is verified ≥ 25 km, and the build asserts the split type and refuses to run
otherwise. The code is `src/finalize_dsv2_spatial.py`; a coordinate-less chip set can only
be split at chip level and is flagged `noncoord` and excluded from every number here.

### 2.3 Label confidence
Aux band 12 stores the number of label sources that agree with the assigned class. A pixel
is **confident** if ≥ 2 sources agree (`CONF_MIN=2`). 84.0% of evaluation pixels are
confident. Class composition differs sharply between views (Table 1): confidence removes
79% of shrub and 96% of wetland pixels, which is where the two views diverge.

**Table 1 — evaluation pixels per class (all / confident).**

| Class | All px | Confident px | Kept |
|---|---|---|---|
| tree | 12,594,363 | 11,598,776 | 92% |
| shrub | 884,127 | 187,600 | 21% |
| grass | 3,652,171 | 2,545,397 | 70% |
| crop | 8,816,234 | 7,668,759 | 87% |
| built | 382,793 | 361,040 | 94% |
| bare | 4,981,733 | 4,125,573 | 83% |
| snow | 2,035,112 | 1,474,589 | 72% |
| water | 371,507 | 291,874 | 79% |
| wetland | 65,063 | 2,493 | 4% |
| mangrove | 54,483 | 54,483 | 100% |
| moss | 683,502 | 683,502 | 100% |

### 2.4 Metrics
mIoU (primary), macro-F1 and OA, each on confident pixels and all pixels, with per-class
IoU. Every arm is run with **two seeds** on the identical split; we report the mean and
both seeds.

---

## 3. Models and protocol

**Backbone / head.** Prithvi-EO-2.0 300M ViT (via `terratorch`) with a UPerNet head over
four encoder depths; fine-tuned end to end for 35 epochs (3 warm-up), batch 2 with gradient
accumulation to an effective 8, AMP, cosine LR, on a single Kaggle T4.

**Shared by every arm.** UPerNet head, LLRD, CE + Dice + Lovász losses and test-time
augmentation. These are not ablated.

**"Baseline" arm.** Pretrained encoder, fine-tuned, with the four additions below all off.

**"v3" arm.** Baseline plus four switchable additions: (a) auxiliary-band stem for the 11 non-optical bands,
(b) confidence masking of low-agreement pixels *in training only*, (c) soft labels,
(d) copy-paste augmentation of snow/wetland/mangrove/moss instances. The class-balanced sampler is **off** in all arms
reported here.

**Reference arms.** *Random-init*: the same ViT, `pretrained=False`, fine-tuned. *Frozen*:
pretrained encoder frozen, only head trained. All reference arms have every v3 addition
off, so each differs from the baseline in exactly one thing.

**Eval-mask protocol.** Training may mask low-agreement pixels; **evaluation never drops
pixels at load time.** Low-agreement pixels are tagged and then scored in the all-pixels
view and excluded in the confident view. This corrects an earlier pipeline in which the
evaluation set itself changed with the arm (§7).

---

## 4. Results

### 4.1 What pretraining buys (main result)

**Table 2 — value of pretraining (mean of 2 seeds).**

| Model | mIoU conf. | mIoU all | macro-F1 conf. | macro-F1 all | OA conf. | OA all |
|---|---|---|---|---|---|---|
| Random-init ViT | 0.343 | 0.300 | 0.451 | 0.416 | 0.736 | 0.675 |
| Frozen pretrained encoder | 0.402 | 0.343 | 0.507 | 0.463 | 0.773 | 0.711 |
| Fine-tuned pretrained (baseline) | 0.448 | 0.388 | 0.550 | 0.509 | 0.819 | 0.761 |
| v3 (fine-tuned, all additions) | 0.465 | 0.359 | 0.570 | 0.478 | 0.826 | 0.743 |

Seed-level mIoU (conf. / all): random-init 0.342, 0.344 / 0.300, 0.300; frozen 0.402,
0.402 / 0.342, 0.344; baseline 0.445, 0.451 / 0.386, 0.389; v3 0.462, 0.469 / 0.358, 0.359.

- Pretraining is worth **+0.105 mIoU** (confident) / **+0.088** (all) over random init.
- Fine-tuning is worth **+0.046** / **+0.045** over a frozen encoder.
- Both gaps are ≥ 10× the largest seed spread (< 0.005). This is the robust finding.
- Random-init still reaches 0.343, i.e. the consensus labels carry real learnable
  structure (and, given their origin, partly the structure of the label models). It is a
  floor on label-model agreement, not a measure of ground truth.

### 4.2 v3 versus baseline: a trade-off, not a gain
On confident pixels v3 is +0.017 mIoU over the baseline; on all pixels it is **−0.029**.
The sign flips with the evaluation set, so no single "v3 is better" statement is
supportable. The two views disagree because confidence masking removes the hard,
ambiguous pixels from training (§4.3).

### 4.3 Component ablation (leave-one-out from v3, 2 seeds)

**Table 3 — mIoU, mean of 2 seeds, Δ vs v3.**

| Arm | conf. | Δ | all | Δ |
|---|---|---|---|---|
| v3 (full) | 0.4652 | – | 0.3586 | – |
| − aux stem | 0.4627 | −0.002 | 0.3573 | −0.001 |
| − confidence masking | 0.4515 | −0.014 | 0.3860 | **+0.027** |
| − soft labels | 0.4656 | +0.000 | 0.3585 | −0.000 |
| − copy-paste | 0.4772 | **+0.012** | 0.3693 | +0.011 |
| baseline (all off) | 0.4479 | −0.017 | 0.3877 | +0.029 |

- **Aux stem and soft labels: no measurable effect** (|Δ| ≤ 0.002 on both views).
- **Confidence masking** moves the two views in opposite directions (−0.014 / +0.027): it
  is a **specialisation to confident pixels**, paid for on ambiguous ones.
- **Removing copy-paste helps** (+0.012 / +0.011), but the two seeds disagree
  (conf. 0.4845 vs 0.4700; all 0.3777 vs 0.3609), so we do not claim it from two seeds.
- Two seeds cannot establish effects below ≈ 0.01; every individual Δ in this table
  should be read with that limit.

### 4.4 Per-class IoU

**Table 4 — IoU on confident pixels, mean of 2 seeds.**

| Class | Random-init | Frozen | Baseline | v3 |
|---|---|---|---|---|
| tree | 0.834 | 0.859 | 0.881 | 0.893 |
| shrub | 0.074 | 0.143 | 0.155 | 0.301 |
| grass | 0.357 | 0.388 | 0.447 | 0.480 |
| crop | 0.712 | 0.823 | 0.855 | 0.884 |
| built | 0.283 | 0.613 | 0.716 | 0.724 |
| bare | 0.307 | 0.352 | 0.496 | 0.475 |
| snow | 0.358 | 0.354 | 0.397 | 0.373 |
| water | 0.649 | 0.706 | 0.741 | 0.756 |
| wetland | 0.000 | 0.000 | 0.000 | 0.013 |
| mangrove | 0.000 | 0.003 | 0.001 | 0.015 |
| moss | 0.196 | 0.180 | 0.236 | 0.202 |

Pretraining's gain is concentrated in **built** (0.28 → 0.72), **crop** (0.71 → 0.86) and
**grass/bare**; classes with weak or ambiguous labels (snow, moss, shrub) barely move.
v3's mIoU gain on confident pixels comes largely from **shrub** (0.155 → 0.301), a class
for which only 21% of pixels are confident — a plausible by-product of evaluating on the
residue that confidence masking leaves, rather than a better shrub model (on all pixels
shrub is 0.293 for baseline vs 0.209 for v3).

### 4.5 Rare classes are not evaluable
Wetland (2,493 confident pixels; ~1 chip) and mangrove (54,483 px; ~1 chip) score IoU ≤
0.015 in every arm. A single chip makes IoU a measure of one scene, so these two classes
contribute noise to mIoU, and any rare-class intervention (the copy-paste above) cannot be
validated on them. They are retained for completeness, and mIoU is reported with all 11
classes so the gap is visible rather than hidden.

---

## 5. Discussion

**Use a pretrained FM and fine-tune it.** This is the one result that is large, stable
across seeds, and holds on both evaluation views (+0.09–0.10 over random init, +0.045 over
frozen). A shared pretrained encoder is a defensible investment; the "more training
tricks" layer on top of it is not demonstrated here.

**Report both pixel sets.** On this data the evaluation-set choice changes mIoU by
0.06–0.08, larger than every method delta except pretraining. With consensus labels a
single number can be made to favour either design simply by choosing the mask.

**Do not read the v3 recipe as a method contribution.** Its only consistent effect is the
confident/ambiguous trade-off; the remaining components are null at our resolution.

---

## 6. Limitations

- **Weak labels.** All labels derive from other models (Dynamic World, ESRI, WorldCover,
  WorldCereal). We have no independent human-labelled reference for dense land cover;
  scores are agreement with consensus, not accuracy. This is the main scientific weakness.
- **Two seeds, one split.** Deltas below ≈ 0.01 are not resolved; split-to-split variance
  is untested.
- **One backbone family.** No ImageNet/ResNet baseline and no other EO FM (Clay,
  TerraMind, DOFA, SatMAE) on this split. The v1 draft's ImageNet-ViT comparison is not
  reproduced here (§7).
- **Rare classes** (wetland, mangrove) are unmeasurable; unbalanced-sampler and 17-band
  versus RGB comparisons have not been rerun on the spatial split.
- **Single T4, one reference year.** No temporal or multi-sensor evaluation.
- **Selection.** No hyper-parameter search per arm; arms share one configuration, which
  can favour the arm it was developed for (v3).

---

## 7. Corrections to earlier results

We discovered and corrected the following; they are stated in full because each changed a
conclusion.

1. **Non-spatial split (v1 draft).** The v1 land-cover numbers (0.663 macro-F1 / 0.527
   mIoU, an "EO beats ImageNet by +0.042 / +0.08" claim, the "0.86 OA" point-probe
   results) were computed on a chip-level, zone-stratified split without coordinates.
   They are **withdrawn** and not comparable to the numbers here; on the spatial split the
   comparable baseline is 0.388–0.448 mIoU. The ImageNet-ViT and decoder-matching claims
   are untested on the spatial split.
2. **Eval-mask confound.** An earlier pipeline masked low-agreement pixels at load time
   for *every* split, so a model trained with confidence masking was scored on a
   different, easier pixel set than the baseline. We had reported a +0.074 mIoU v3 gain,
   a "confidence masking carries the model" attribution, a −0.109 ablation effect and that
   "half the gain was leakage"; all four are **retracted**. With a fixed, identical
   evaluation set the v3 effect is +0.017 / −0.029 (§4.2).
3. **Run bookkeeping.** An exporter ran as 13 concurrent writers and produced chips with
   conflicting coordinates; the chip set was rebuilt behind a directory lock before any
   number reported here. Out-of-memory retries that pinned a failed model alive were fixed
   (retry outside the exception handler plus explicit purge).

---

## 8. Reproducibility

Code is in `bharat-eo-lab/`: split construction (`src/finalize_dsv2_spatial.py`), the
Earth Engine exporter (`src/export_dataset_v2.py`), and the Kaggle kernels
(`notebooks/kaggle_v3_spatial/`, builders `build_spatial.py`, `build_eval.py`,
`build_abl.py`, `build_bl.py`). Raw per-seed JSONs and logs for every arm are in
`outputs/spatial_v3_results/`. Chip arrays and raw CSVs are not in git (regenerable with
the exporter / on Kaggle as `bharat-eo-chips-dsv2-sp`). Compute: a 16 GB laptop for
control, Earth Engine for export, one T4 per kernel (~2.5 h for the random-init arm).
A release licence and a dataset card still need to be written.

---

## 9. Conclusion

On a leakage-controlled India land-cover split with consensus labels, a pretrained EO
foundation model fine-tuned end to end reaches 0.448 mIoU on confident pixels (0.388 on
all), +0.105 (+0.088) over random initialisation and +0.046 over a frozen encoder, with
seed spread under 0.005. A four-part training recipe adds at most a confident-pixel
specialisation and loses on ambiguous pixels; wetland and mangrove cannot be evaluated.
The immediate work that would make this a research paper rather than a benchmark release
is independent ground truth for dense land cover, rare-class data (e.g. Global Mangrove
Watch, Dynamic World flooded vegetation), and a second backbone family on the same split.
