# Bharat-FM / Bharat-EO-Bench

A spatially-disjoint **India land-cover segmentation benchmark** (11 classes, 15
agro-climatic zones) and the experiments measuring what an Earth-observation foundation
model (Prithvi-EO-2.0 300M + UPerNet) actually buys on it.

> Status: benchmark and negative-results release, not a new method. Labels are
> consensus products (Dynamic World / ESRI / WorldCover), so scores measure agreement with
> those products, not independent accuracy. Read [the preprint](bharat-eo-lab/docs/PREPRINT.md)
> before quoting any number.

## Headline results (spatial-block split, mean of 2 seeds)

| Model | mIoU, confident px | mIoU, all px |
|---|---|---|
| Random-init ViT | 0.343 | 0.300 |
| Frozen pretrained encoder | 0.402 | 0.343 |
| Fine-tuned pretrained (baseline) | 0.448 | 0.388 |
| v3 (baseline + 4 additions) | 0.465 | 0.359 |

- Pretraining is worth about +0.09 to +0.10 mIoU over random init; fine-tuning about
  +0.045 over a frozen encoder. Seed spread is under 0.005.
- The v3 additions (aux stem, confidence masking, soft labels, copy-paste) do not reliably
  help: the sign flips with the evaluation pixel set.
- Wetland and mangrove (about 1 chip each) are not measurable yet.

Earlier land-cover numbers (0.663 macro-F1, "0.86 OA") came from a non-spatial split and
are **withdrawn**; see section 7 of the preprint.

## Repository layout

```
bharat-eo-lab/
  src/                     exporters (Earth Engine), spatial split, older task code
  notebooks/
    kaggle_v3_spatial/     Kaggle kernels + builders for the spatial-split experiments
  outputs/
    spatial_v3_results/    per-seed JSONs and logs for every arm in the preprint
  docs/                    PREPRINT.md, BENCHMARK_SPEC.md, PRETRAINING_RECIPE.md, ...
  RESULTS.md               older AlphaEarth-embedding tasks (flood, groundwater, ...)
Bharat-EO-Bench_Preprint.md   copy of docs/PREPRINT.md
```

Not in git (too large or regenerable): chip arrays and zips, `data/raw/` CSVs, virtual
environments. The chips are published privately on Kaggle as `bharat-eo-chips-dsv2-sp`.

## Method in brief

- **Split:** 1-degree blocks assigned whole to train/val/test; train chips within 25 km of
  any evaluation chip are dropped; minimum train-to-eval distance is checked (>= 25 km).
  Code: `bharat-eo-lab/src/finalize_dsv2_spatial.py`.
- **Two evaluation views:** *confident pixels* (>= 2 label sources agree, 84% of eval
  pixels) and *all pixels*. Training may mask low-agreement pixels; evaluation never does.
- **Model:** Prithvi-EO-2.0 300M via `terratorch`, UPerNet head, LLRD, CE + Dice + Lovasz,
  TTA; 35 epochs on one Kaggle T4.

## Reproducing

1. Export chips with `bharat-eo-lab/src/export_dataset_v2.py` (needs an Earth Engine
   account), then build the split with `finalize_dsv2_spatial.py`. Or attach the Kaggle
   dataset `bharat-eo-chips-dsv2-sp`.
2. Generate kernels with the builders in `bharat-eo-lab/notebooks/kaggle_v3_spatial/`:
   `build_spatial.py` (single run), `build_eval.py` (v3 vs baseline, both pixel sets),
   `build_abl.py` (leave-one-out ablation), `build_bl.py` (random-init and frozen).
3. Push with the Kaggle CLI (`kaggle kernels push -p <push_dir>`); outputs are the
   `seg_v3_*.json` files committed under `outputs/spatial_v3_results/`.

Windows note: the chip folder `aux` is a reserved device name and `os.walk` silently skips
it; the local copy is named `auxb`, and notebooks resolve it via `AUXD`.

## Known gaps / next steps

- Independent ground truth for dense land cover (all labels are model products).
- Rare-class data (Global Mangrove Watch, Dynamic World flooded vegetation) for wetland and
  mangrove.
- A second backbone family (ImageNet ResNet/ViT, other EO FMs) on the same split.
- More seeds and split replicates; dataset card and release licence.

## Contact

crakashp2905@gmail.com
