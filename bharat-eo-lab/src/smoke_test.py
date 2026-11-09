"""Synthetic end-to-end smoke test -- proves the laptop pipeline runs with no
Earth Engine account. Generates class-separable fake embeddings + Indian
coordinates, then runs train -> benchmark -> viz exactly like the real data.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


def make_synthetic(n_per_class=500, n_dim=64, seed=C.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = list(C.CLASS_NAMES.keys())
    # each class gets a random centroid; samples are centroid + noise -> learnable
    centroids = rng.normal(0, 1.0, size=(len(classes), n_dim))
    rows = []
    # rough India bounding box
    lon_lo, lon_hi, lat_lo, lat_hi = 68.0, 97.5, 8.0, 37.0
    for ci, c in enumerate(classes):
        feats = centroids[ci] + rng.normal(0, 0.9, size=(n_per_class, n_dim))
        lon = rng.uniform(lon_lo, lon_hi, n_per_class)
        lat = rng.uniform(lat_lo, lat_hi, n_per_class)
        for k in range(n_per_class):
            rec = {C.EMB_COLS[d]: float(feats[k, d]) for d in range(n_dim)}
            rec.update(label=c, lon=float(lon[k]), lat=float(lat[k]),
                       state_code=int(rng.integers(0, 35)))
            rows.append(rec)
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main():
    df = make_synthetic()
    df.to_csv(C.SMOKE_CSV, index=False)
    print(f"[smoke] synthetic table: {df.shape[0]} rows x {df.shape[1]} cols "
          f"-> {C.SMOKE_CSV}")
    return C.SMOKE_CSV


if __name__ == "__main__":
    main()
