# Bharat-EO-Bench v1 — Specification

**Version:** 1.0.0   **Status:** under construction   **Owner:** Bharat-EarthFM team

A rigorous, stratified, versioned benchmark for Earth-observation foundation
models on India. v0 (in RESULTS.md) proved feasibility on random samples; v1
fixes the four calibre gaps: **stratification, real labels, temporal depth, and
fixed geographically-disjoint splits.**

---

## 1. Design principles (non-negotiable)
1. **Stratified, not random** — sampled across agro-climatic zone × task class so
   every class is represented in every zone it occurs in.
2. **Geographically-disjoint splits** — train / val / test never share a spatial
   block; test is a held-out set of blocks fixed forever.
3. **Multi-source labels** — a primary (silver) label per task PLUS a
   ground-truth-anchored subset validated against authoritative Indian data.
4. **Temporal** — monthly time-series features (not single composites) so
   phenology / monsoon-phase / forecasting tasks are supported.
5. **Versioned & reproducible** — every release is a manifest + content hash;
   fixed seeds; provenance recorded per sample.
6. **Variance reported** — metrics with bootstrap CIs or k spatial folds, never a
   single point estimate.

---

## 2. Agro-climatic zonation (the stratification axis)
India's EO variance is driven by rainfall regime × physiography. v1 defines
**15 zones** from open climatology (transparent, reproducible):

- **Rainfall** (WorldClim bio12, mm/yr): arid <500 · semi-arid 500–1000 ·
  sub-humid 1000–1500 · humid 1500–2500 · per-humid >2500  → 5 classes
- **Physiography** (SRTM elevation, m): plains <300 · uplands 300–1000 ·
  hills/mountains >1000  → 3 classes
- `zone_id = rainfall_class * 3 + elev_class`  → 0..14

(Defensible proxy for the Planning Commission's 15 agro-climatic zones; swap for
NBSS&LUP agro-ecological regions when institutional data is available.)

---

## 3. Tasks (v1)

| ID | Task | Type | Label (silver) | Ground-truth anchor | Target size |
|---|---|---|---|---|---|
| **T1** | Land-cover / crop **segmentation** | dense (chips 224²) | ESA WorldCover | validated LC reference / manual | ≥5,000 chips |
| **T2** | Land-cover **classification** w/ time-series | point + 12-mo series | ESA WorldCover | field/gov subset | ≥8,000 pts |
| **T3** | **Groundwater** depletion | GRACE cell / downscaled | GRACE trend | **CGWB piezometer wells (India-WRIS)** | all NW-India cells |
| **T4** | **Flood** susceptibility (Assam) | point | Sentinel-1 SAR + JRC | gov flood-extent maps | ≥5,000 pts |

v1 ships **T2 (flagship, stratified time-series)** + **T1** + **T3 with real CGWB
anchor**. T4 carried from v0, upgraded later.

---

## 4. Stratified sampling plan (T2 flagship)
- Draw per **(zone × class)** cell: target `min(class_available, N_cell)` with
  `N_cell` set so total ≈ 8,000–10,000, floor 20/cell where the class exists.
- Rare classes (mangrove/wetland/moss/snow) additionally targeted via the
  region-box sampler (as validated in v0: 0.00→0.86 mangrove IoU).
- Record per sample: `lon, lat, zone_id, class, split, source`.

## 5. Features (per T2 point)
- **AlphaEarth** 64-d annual embedding (baseline-to-beat).
- **Monthly time-series** (12×): NDVI, NDWI, S2 [B,G,R,NIR,SWIR1,SWIR2] median,
  S1 VV/VH — enables phenology, MPATE, forecasting.
- **Static**: elevation, slope, zone_id.

## 6. Splits (fixed, versioned)
- Spatial-block hold-out, **1° blocks**: **70 train / 15 val / 15 test**.
- Blocks assigned by fixed seed 42; **test blocks frozen** across all versions.
- Rare-class targeted points split per-class-stratified into the same block sets
  so each class appears in train **and** test.

## 7. Metrics & protocol
- Overall accuracy, **macro-F1**, per-class F1/IoU, **macro-F1 (supported ≥10)**.
- **Variance**: 5 spatial folds (rotate held-out blocks) → mean ± std, OR 1,000×
  bootstrap CIs on the test set.
- Every candidate model (Prithvi, Clay, SatMAE, AlphaEarth, Bharat-*) evaluated
  on the **identical** split with the **identical** protocol.

## 8. Ground-truth anchoring (the credibility step)
- **T3 groundwater:** CGWB monitoring-well water-level trends from **India-WRIS**
  → correlate against the downscaled GRACE field per district (real cross-sensor
  validation, not GRACE-vs-GRACE).
- **T2 land cover (subset):** anchor ≥500 points to a validated reference
  (Bhuvan LULC 50k / a field-checked source) — even a small anchored subset
  converts "reproduces WorldCover" into "agrees with reality X%."

## 9. Versioning & provenance
- Release = `bharat_eo_bench_v1.0.0/` with: per-task Parquet (features+labels+
  split+zone+source), `manifest.json` (counts per zone×class×split), `SPEC.md`
  (this file), and a `SHA256SUMS` content hash. Seeds and GEE asset ids recorded.

---

## 10. Completion checklist
- [x] Agro-zone layer built + area-per-zone verified (15 zones, 12 with >1% area)
- [x] T2 stratified time-series built — 5,332 pts, 15/15 zones, 11/11 classes, disjoint splits, versioned
- [x] **T1 chip set — 5,271 8-band chips** (3741/724/806 split), versioned (t1_manifest.json, sha bac1d6d3), zipped for Kaggle
- [x] **T2 field-truth anchor** — WorldCover vs 239 real WorldCereal India crop points: **90.0% cropland agreement**
- [x] **T3 CGWB well anchor pulled + correlated to GRACE** — 3,981 wells / 71,003 obs;
      well depth-trend vs GRACE storage-trend **r = -0.284, p = 1.4e-49** (2,599 wells). REAL ground truth.
- [x] LC label-reliability anchor — WorldCover vs Dynamic World, 1,367 pts, 66.1% agree (built 90/tree 84/water 80)
- [x] manifest.json + SHA256 + **variance protocol** (bootstrap CIs + 5-fold spatial CV) — all wired & run
