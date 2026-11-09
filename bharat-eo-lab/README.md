# bharat-eo-lab

The **laptop-feasible slice** of the Bharat-EarthFM blueprint: India-scale
Earth-observation modelling on a CPU-only laptop. Google Earth Engine does the
heavy lifting server-side; only small tables come home. Five Indian tasks —
land cover, flood, groundwater, GRACE downscaling, and an independent
validation — all run on a 16 GB laptop. See **[RESULTS.md](RESULTS.md)** for the
headline numbers.

```
Earth Engine (servers)                    your laptop (CPU)
─────────────────────                     ─────────────────
India / region AOI                        load table
AlphaEarth 64-d embeddings   ──sample──▶  XGBoost / MLP head (minutes)
labels (WorldCover, S1 SAR, GRACE)        Bharat-EO-Bench + figures + maps
```

## Setup
```bash
python -m venv ../.venv
../.venv/Scripts/python -m pip install -r requirements.txt
# one-time Earth Engine auth (opens a browser):
../.venv/Scripts/earthengine authenticate
# set your project id in src/config.py -> EE_PROJECT, or pass --project
```

## Commands
```bash
python run.py smoke                          # synthetic end-to-end test (no account)
python run.py export-local  --project <ee>   # India land-cover sample (batched getInfo)
python run.py export-drive  --project <ee>   # large async Export to Google Drive
python run.py boost         --project <ee>   # + targeted rare-class sampling, re-benchmark
python run.py analyze       --csv <file.csv> # run Bharat-EO-Bench on any sample table
python run.py flood         --project <ee>   # Assam 2022 flood susceptibility (Sentinel-1)
python run.py groundwater   --project <ee>   # GRACE depletion regression (Punjab/NW India)
python run.py gw-downscale  --project <ee>   # downscale GRACE 110km->~1.5km + districts
python run.py gw-validate   --project <ee>   # validate depletion: GRACE-FO + cross-center
python run.py export-chips  --project <ee> --n 150   # S2 image chips + masks for Prithvi fine-tune
```

## Tasks & headline results (full detail in RESULTS.md)
| Task | Command | Best result |
|---|---|---|
| Land cover (11 cls) + rare-class boost | `boost` | 0.86 OA / 0.80 macro-F1; mangroves 0→0.98 |
| Deep model vs XGBoost (B0–B4) | `analyze` | XGBoost ≈ MLP (~0.86) — head barely matters |
| Flood susceptibility (Assam) | `flood` | flood-F1 0.56; embedding ≫ terrain |
| Groundwater depletion (GRACE) | `groundwater` | Punjab −0.86 cm/yr; R² 0.34→0.89 with embedding |
| GRACE downscaling 110km→1.5km | `gw-downscale` | R² 0.924; district hotspots (Rohtak, Sonepat…) |
| Independent validation | `gw-validate` | GRACE vs GRACE-FO agree; structural, −36 cm since 2002 |

**Benchmark models:** B0 majority · B1 geography · B2 AlphaEarth+XGBoost
(+B2w weighted, B2s SMOTE) · B3 +geography · B4 AlphaEarth+neural-net (sklearn MLP).

## Project layout
```
src/config.py        all knobs (year, sampling, classes, RARE_BOXES, paths)
src/utils.py         leakage-free spatial-block split, IO
src/gee_export.py    Earth Engine: AlphaEarth + WorldCover sampling (+ rare-class boost)
src/train.py         XGBoost head, shared metrics (summarize), SMOTE/weighting
src/mlp.py           sklearn MLP 'deep model' baseline (B4)
src/benchmark.py     Bharat-EO-Bench (B0–B4 leaderboard)
src/viz.py           confusion matrix, per-class F1, prediction map
src/flood.py         Assam flood task (Sentinel-1 SAR labels + susceptibility)
src/groundwater.py   GRACE depletion + surface-driver regression
src/gw_downscale.py  GRACE -> ~1.5km statistical downscaling + district ranking
src/gw_validate.py   cross-mission (GRACE-FO) + cross-center validation + decline curve
src/export_chips.py  Sentinel-2 224x224 6-band chips + WorldCover masks (for Prithvi)
src/smoke_test.py    synthetic end-to-end test
run.py               CLI orchestrator (all commands above)
notebooks/
  finetune_gpu.ipynb     MLP on frozen embeddings, Kaggle/Colab GPU (proven on T4)
  prithvi_finetune.ipynb fine-tune Prithvi-EO-2.0 (TerraTorch) on S2 chips; smp U-Net fallback
  insar_hyp3_punjab.py   real InSAR subsidence via ASF HyP3 (needs Earthdata login)
outputs/             figures + reports          RESULTS.md  capstone summary
docs/
  PRETRAINING_RECIPE.md      full Bharat-EarthFM-L training recipe (stages, losses, FLOPs)
  DATA_ENGINEERING_DAG.md    Bharat-EO-Corpus build pipeline (nodes, ARD schema, MPATE labeler)
  PATENTS.md                 drafted claims for crown-jewel filings (P01/P02/P03/P04/P07)
requirements.txt     pinned core dependencies
```

## Honest caveats
- AlphaEarth is a **global, closed** embedding (Google) — used here as the baseline
  to beat; a sovereign Indian FM is the goal.
- These are **frozen embeddings + a lightweight head**, not a fine-tuned foundation
  model. Groundwater is **coarse (~110 km GRACE)** and the downscaling is
  unvalidated vs CGWB wells (scale-invariance assumed).

## Remaining steps (need a GPU / your credentials — can't run on this laptop)
- **Fine-tune Prithvi/Clay** on image chips via `notebooks/finetune_gpu.py` on a
  free Kaggle/Colab GPU (vs. frozen embeddings).
- **Real InSAR subsidence** via `notebooks/insar_hyp3_punjab.py` (free ASF HyP3,
  needs a NASA Earthdata login) to fuse with the GRACE downscaling.
```
