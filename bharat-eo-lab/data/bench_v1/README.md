# Bharat-EO-Bench v1.0.0 — Release

A stratified, temporally-resolved, ground-truth-anchored, versioned benchmark for
EO models on India. See `docs/BENCHMARK_SPEC.md` for the full spec.

## Contents
| File | What |
|---|---|
| `t2_landcover_ts_v1.0.0.parquet` | **T2** — 5,332 pts, 64-d AlphaEarth + 4-season NDVI/VV/VH + terrain; label (11 cls) + zone + split |
| `manifest.json` | T2 counts per zone×class×split + SHA256 |
| `t2_baseline_variance.json` | T2 baseline result with bootstrap CIs + spatial folds |
| `t2_lc_anchor_agreement.csv` | T2 label-reliability anchor (WorldCover vs Dynamic World) |
| `t3_cgwb_grace_validation.csv` | **T3** — 2,599 CGWB well trends vs GRACE (real ground truth) |

## Design (all requirements met)
- **Stratified** by 15 agro-climatic zones (rainfall×physiography) × 11 classes — balanced (61–1,112/class vs raw 0.1–50%).
- **Geographically-disjoint** splits: whole 1° blocks, 70/15/15, test frozen.
- **Time-series**: 4 seasonal quarters (rabi/zaid/kharif/post-monsoon), NDVI + S1 VV/VH.
- **Ground-truth-anchored**: T3 = real CGWB wells; T2 = cross-product label reliability.
- **Versioned**: manifest + SHA256 (`bffd5fbcd9c32e46`); seeds & GEE assets recorded.
- **Variance-reported**: bootstrap CIs + 5-fold spatial CV.

## Baseline results (AlphaEarth + seasonal-TS + XGBoost)
- **T2 land cover** — fixed test (n=696): **OA 0.785** [0.756, 0.815], **macro-F1 0.718** [0.634, 0.805];
  5-fold spatial: **OA 0.780 ± 0.006, macro-F1 0.760 ± 0.018** (generalises across regions).
- **T3 groundwater** — CGWB well depth-trend vs GRACE storage-trend: **Pearson r = −0.284, p = 1.4×10⁻⁴⁹**
  (2,599 wells; in-situ agrees with satellite → real cross-sensor validation).
- **T2 FIELD TRUTH** — WorldCover vs **239 expert-validated India crop points** (WorldCereal validation,
  Zenodo 7825628): **90.0% cropland agreement** (misses are mostly orchards WorldCover calls "tree").
  Real reference, not a model. -> `t2_fieldtruth_india.csv`
- **T2 label reliability** — WorldCover vs Dynamic World: 66.1% overall (built 90 / tree 84 / water 80;
  shrub-grass 40 = the genuinely ambiguous boundary).

## Reproduce
```bash
python -m src.bench_v1 verify-zones      # agro-climatic zonation
python -m src.bench_v1 export-t2 --n 32000 ; python -m src.bench_v1 build-t2
python -m src.bench_v1 pull-cgwb ; python -m src.bench_v1 validate-gw
python -m src.bench_v1 lc-anchor ; python -m src.bench_v1 eval-t2
```

## Honest status / backlog (v1.1)
- **T1 (segmentation):** v1.0.0 ships the existing **1,271 8-band chips** (Prithvi-EO-2.0 fine-tune result
  macro-F1 0.575). Scale-up to ~5.3k chips **running now** (`data/chips_v1/`, ~20 h GEE export, resumable).
- **T2 scale:** 5.3k (rares are pool-limited); raise `per_cell` / bigger boost to reach ~8k.
- **Ground-truth: DONE with real field data** — T3 = CGWB wells (r=−0.284); T2 = WorldCereal crop
  validation points (90% agreement). Both authentic references, not models.
- **T4 (flood):** carried from v0 (12k Assam pts); to be re-cut to the v1 spec.
