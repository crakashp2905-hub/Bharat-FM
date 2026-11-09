"""Build a usable manifest.csv for data/chips_dsv2 from the chips on disk.

The v2 export writes the manifest only at the end; a mid-run crash (network
outage) therefore left 825 rich chips WITHOUT a manifest, and the chips do not
store lon/lat, so a geographic-block split is impossible for them. We recover the
metadata that IS in the arrays (dominant agro-zone from the aux 'zone' band, valid
and confidence fractions, dominant class) and assign a ZONE-STRATIFIED chip-level
70/15/15 split (each zone present in every split). This is documented as a
non-spatial split -- chips_dsv2 is augmentation TRAINING data; the leakage-free
spatial evaluation remains Bharat-EO-Bench v1 (bench_v1/, original T1/T2).
"""
from __future__ import annotations
import glob, os
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C
from .export_dataset_v2 import N_ZONES

OUT = C.ROOT / "data" / "chips_dsv2"
AUX_ZONE_IDX = 13          # aux band order: [label_dw,label_esri,dwp0..8,wc_crop,conf,zone]
AUX_CONF_IDX = 12


def _meta(chip):
    aux = np.load(OUT / "auxb" / f"{chip}.npy"); lbl = np.load(OUT / "lbl" / f"{chip}.npy")
    zb = aux[..., AUX_ZONE_IDX]; zv = zb[(zb >= 0) & (zb < N_ZONES)]
    zone = int(np.bincount(zv.ravel()).argmax()) if zv.size else -1
    v = lbl[lbl != 255]
    dom = int(np.bincount(v).argmax()) if v.size else -1
    return dict(chip=chip, zone=zone, dominant=dom, n_classes=int(len(np.unique(v))),
                valid_frac=round(float((lbl != 255).mean()), 3),
                conf_frac=round(float((aux[..., AUX_CONF_IDX] >= 2).mean()), 3),
                target=int(chip.split("_t")[1].split("_")[0]) if "_t" in chip else -1)


def finalize(seed=C.SEED, val_frac=0.15, test_frac=0.15):
    chips = sorted(os.path.basename(p)[:-4] for p in glob.glob(str(OUT / "img" / "*.npy")))
    rows = [_meta(c) for c in chips]
    df = pd.DataFrame(rows)
    # zone-stratified split: within each dominant-zone group, shuffle -> 70/15/15
    rng = np.random.default_rng(seed); df["split"] = "train"
    for z, g in df.groupby("zone"):
        idx = g.index.to_numpy().copy(); rng.shuffle(idx)
        n = len(idx); nv = max(1, int(n * val_frac)); nt = max(1, int(n * test_frac))
        df.loc[idx[:nv], "split"] = "val"; df.loc[idx[nv:nv + nt], "split"] = "test"
    df["split_type"] = "zone_stratified_chiplevel_noncoord"     # honest provenance flag
    df.to_csv(OUT / "manifest.csv", index=False)
    print(f"[finalize] {len(df)} chips -> {OUT/'manifest.csv'}")
    print(f"  splits: {df.split.value_counts().to_dict()}")
    print(f"  zones covered: {df.zone.nunique()}/{N_ZONES} | "
          f"dominant-class spread: {dict(df.dominant.value_counts().sort_index())}")
    print(f"  mean valid {df.valid_frac.mean():.3f} | mean conf>=2 {df.conf_frac.mean():.3f}")
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None, help="chip dir to finalize (default data/chips_dsv2)")
    a = ap.parse_args()
    if a.dir:
        OUT = Path(a.dir)          # rebinds the module global read by _meta/finalize
    finalize()
