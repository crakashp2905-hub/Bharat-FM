"""Build a LEAKAGE-FREE spatial manifest for a coord-complete chip dir (data/chips_dsv2_sp).

`finalize_dsv2.py` exists for chips_dsv2, whose chips carry no lon/lat: it can only do a
zone-stratified *chip-level* split and honestly flags it `..._noncoord`. The _sp re-export
logs lon/lat per chip (`_rows.csv`, written crash-safe as each chip lands), so here we can
do the real thing: a **geographic block split**. Chips are bucketed into square blocks of
`block_deg` degrees; whole blocks are assigned to train/val/test, so no two chips from the
same neighbourhood ever straddle a split boundary. Optionally buffer the split further by
dropping TRAIN chips within `buffer_km` of any val/test chip -- spatial autocorrelation
reaches past a block edge, and dropping from train (never from eval) keeps the evaluation
set intact and the leakage one-directional.

Reported min train->test distance is the honest number to quote in a paper.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C
from .export_dataset_v2 import N_ZONES

OUT = C.ROOT / "data" / "chips_dsv2_sp"
R_EARTH_KM = 6371.0088


def _haversine_km(lon1, lat1, lon2, lat2):
    """Pairwise great-circle distance, shape (len(a), len(b)), in km."""
    la1, la2 = np.radians(lat1)[:, None], np.radians(lat2)[None, :]
    dla = la2 - la1
    dlo = np.radians(lon2)[None, :] - np.radians(lon1)[:, None]
    h = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def finalize_spatial(out=OUT, seed=C.SEED, block_deg=1.0, val_frac=0.15, test_frac=0.15,
                     buffer_km=0.0):
    out = Path(out)
    df = pd.read_csv(out / "_rows.csv")
    on_disk = {p.stem for p in (out / "img").glob("*.npy")}
    missing = set(df.chip) - on_disk
    if missing:
        print(f"[spatial] dropping {len(missing)} row(s) with no chip on disk")
        df = df[df.chip.isin(on_disk)]
    orphan = on_disk - set(df.chip)
    if orphan:
        raise SystemExit(f"[spatial] {len(orphan)} chip(s) on disk have no row "
                         f"(e.g. {sorted(orphan)[:3]}) -- repair before finalizing")
    df = df.drop_duplicates("chip").reset_index(drop=True)

    # --- geographic blocks: whole blocks go to one split, so neighbours never split
    df["block"] = (np.floor(df.lon / block_deg).astype(int).astype(str) + "_" +
                   np.floor(df.lat / block_deg).astype(int).astype(str))
    blocks = df.block.unique()
    rng = np.random.default_rng(seed); rng.shuffle(blocks)
    size = df.block.value_counts()
    n_val, n_test = len(df) * val_frac, len(df) * test_frac

    split_of, acc_v, acc_t = {}, 0, 0
    for b in blocks:                       # fill test, then val, remainder trains
        if acc_t < n_test:   split_of[b] = "test"; acc_t += size[b]
        elif acc_v < n_val:  split_of[b] = "val";  acc_v += size[b]
        else:                split_of[b] = "train"
    df["split"] = df.block.map(split_of)

    # --- optional buffer: drop TRAIN chips too close to any eval chip
    ev = df[df.split != "train"]
    tr = df[df.split == "train"]
    d = _haversine_km(tr.lon.to_numpy(), tr.lat.to_numpy(), ev.lon.to_numpy(), ev.lat.to_numpy())
    nearest = d.min(axis=1) if d.size else np.full(len(tr), np.inf)
    if buffer_km > 0:
        drop = tr.index[nearest < buffer_km]
        print(f"[spatial] buffer {buffer_km} km -> dropping {len(drop)} train chip(s)")
        df = df.drop(index=drop).reset_index(drop=True)
        tr = df[df.split == "train"]
        d = _haversine_km(tr.lon.to_numpy(), tr.lat.to_numpy(), ev.lon.to_numpy(), ev.lat.to_numpy())
        nearest = d.min(axis=1) if d.size else np.full(len(tr), np.inf)

    df["split_type"] = f"spatial_block_{block_deg}deg" + (f"_buf{buffer_km}km" if buffer_km else "")
    df.to_csv(out / "manifest.csv", index=False)

    # --- verification: a block must never appear in two splits
    bad = df.groupby("block").split.nunique()
    assert (bad <= 1).all(), f"block leak: {bad[bad > 1].index.tolist()}"

    vc = df.split.value_counts().to_dict()
    print(f"[spatial] {len(df)} chips -> {out/'manifest.csv'}")
    print(f"  splits: {vc}  ({ {k: f'{v/len(df):.0%}' for k, v in vc.items()} })")
    print(f"  blocks: {df.block.nunique()} @ {block_deg}deg | no block spans two splits: OK")
    print(f"  zones per split: { {s: int(g.zone.nunique()) for s, g in df.groupby('split')} } / {N_ZONES}")
    print(f"  min train->eval distance: {nearest.min():.1f} km  (median {np.median(nearest):.1f} km)")
    print(f"  split_type: {df.split_type.iloc[0]}")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--block-deg", type=float, default=1.0)
    ap.add_argument("--buffer-km", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=C.SEED)
    a = ap.parse_args()
    finalize_spatial(out=Path(a.dir) if a.dir else OUT, seed=a.seed,
                     block_deg=a.block_deg, buffer_km=a.buffer_km)
