"""Shared utilities: spatial splitting, IO, metrics helpers."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from . import config as C


def spatial_block_split(df: pd.DataFrame, block_deg=C.BLOCK_DEG,
                        test_frac=C.TEST_FRAC, seed=C.SEED):
    """Leakage-free split: whole lon/lat blocks go entirely to train OR test.

    Random point-wise splits overstate accuracy in EO because neighbouring
    pixels are spatially autocorrelated. Holding out entire geographic blocks
    forces the model to generalise to unseen regions.
    Returns (train_mask, test_mask) as boolean numpy arrays.
    """
    bx = np.floor(df["lon"].to_numpy() / block_deg).astype(int)
    by = np.floor(df["lat"].to_numpy() / block_deg).astype(int)
    blocks = np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))
    uniq = np.unique(blocks)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_blocks = set(uniq[:n_test].tolist())
    test_mask = np.array([b in test_blocks for b in blocks])
    return ~test_mask, test_mask


def load_table(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in (C.EMB_COLS + ["label", "lon", "lat"]) if c not in df.columns]
    # embeddings may be partially absent only in synthetic edge-cases; be strict on core
    core_missing = [c for c in ["label", "lon", "lat"] if c not in df.columns]
    if core_missing:
        raise ValueError(f"CSV missing required columns: {core_missing}")
    df = df.dropna(subset=["label", "lon", "lat"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def present_classes(df: pd.DataFrame):
    labs = sorted(df["label"].unique().tolist())
    return labs, [C.CLASS_NAMES.get(l, str(l)) for l in labs]


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    return path
