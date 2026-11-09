"""Bharat-EO-Bench v0 -- a tiny, honest, reproducible benchmark.

Task card:  India Land Cover 2021 (10 m), ESA WorldCover classes.
Split:      spatial block hold-out (1 deg blocks, 20% test) -- identical across models.
Metrics:    overall accuracy, macro-F1 (all present classes),
            macro-F1 (supported) = classes with >= MIN_SUPPORT test samples.

Baselines establish whether the AlphaEarth embedding adds REAL signal beyond
guessing the majority class / pure geographic location, and whether
class-balancing (weight / SMOTE) lifts performance on rarer classes.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from . import config as C
from . import utils
from .train import train_eval


def _metrics(y_true, y_pred):
    oa = float(accuracy_score(y_true, y_pred))
    mf = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    vc = pd.Series(y_true).value_counts()
    keep = [c for c in sorted(set(y_true.tolist())) if vc.get(c, 0) >= C.MIN_SUPPORT]
    mfs = (float(f1_score(y_true, y_pred, labels=keep, average="macro",
                          zero_division=0)) if keep else 0.0)
    return oa, mf, mfs


def _majority_baseline(df):
    _, test_mask = utils.spatial_block_split(df)
    ytr = df.loc[~test_mask, "label"].to_numpy()
    yte = df.loc[test_mask, "label"].to_numpy()
    pred = np.full(len(yte), int(pd.Series(ytr).mode().iloc[0]))
    return _metrics(yte, pred)


def run_benchmark(df: pd.DataFrame):
    """Return a leaderboard DataFrame + the full result of the best model."""
    rows = []
    emb = [c for c in C.EMB_COLS if c in df.columns]
    has_geo = {"lon", "lat"}.issubset(df.columns)
    best, best_f1 = None, -1.0

    def add(name, res, nfeat):
        nonlocal best, best_f1
        rows.append((name, res["overall_accuracy"], res["macro_f1"],
                     res["macro_f1_supported"], nfeat))

    # B0 majority
    oa, mf, mfs = _majority_baseline(df)
    rows.append(("B0  Majority class", oa, mf, mfs, 1))

    # B1 geography only
    if has_geo:
        r = train_eval(df, ["lon", "lat"])
        add("B1  Geography only (lat/lon)", r, 2)

    if emb:
        # B2 embedding (no balancing)
        r2 = train_eval(df, emb)
        add("B2  AlphaEarth + XGBoost", r2, len(emb))
        if r2["macro_f1"] > best_f1: best, best_f1 = r2, r2["macro_f1"]

        # B2w embedding + class-balanced weights
        r2w = train_eval(df, emb, balance="weight")
        add("B2w AlphaEarth + XGBoost (class-weighted)", r2w, len(emb))
        if r2w["macro_f1"] > best_f1: best, best_f1 = r2w, r2w["macro_f1"]

        # B2s embedding + SMOTE oversampling
        r2s = train_eval(df, emb, balance="smote")
        add("B2s AlphaEarth + XGBoost (SMOTE)", r2s, len(emb))
        if r2s["macro_f1"] > best_f1: best, best_f1 = r2s, r2s["macro_f1"]

        # B3 embedding + geography
        if has_geo:
            r3 = train_eval(df, emb + ["lon", "lat"])
            add("B3  Embedding + geography", r3, len(emb) + 2)
            if r3["macro_f1"] > best_f1: best, best_f1 = r3, r3["macro_f1"]

        # B4 embedding + neural net (the 'deep model'; sklearn MLP on CPU,
        # PyTorch GPU version in notebooks/finetune_gpu.py)
        try:
            from .mlp import train_eval_mlp
            r4 = train_eval_mlp(df, emb)
            add("B4  AlphaEarth + MLP (neural net)", r4, len(emb))
            if r4["macro_f1"] > best_f1: best, best_f1 = r4, r4["macro_f1"]
        except Exception as e:
            print(f"[benchmark] skipping B4 MLP: {e}")

    lb = pd.DataFrame(rows, columns=["model", "overall_accuracy", "macro_f1",
                                     "macro_f1_supported", "n_features"])
    lb = lb.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    return lb, best


def leaderboard_markdown(lb: pd.DataFrame, df: pd.DataFrame, best=None) -> str:
    labs, names = utils.present_classes(df)
    lines = [
        "# Bharat-EO-Bench v0 -- India Land Cover 2021",
        "",
        f"- **Region:** {C.COUNTRY}   **Year:** {C.YEAR}   **Resolution:** {C.SCALE} m",
        f"- **Samples:** {len(df):,}   **Classes present:** {len(labs)} "
        f"({', '.join(names)})",
        f"- **Split:** spatial block hold-out ({C.BLOCK_DEG} deg, {int(C.TEST_FRAC*100)}% test)",
        f"- **macro_f1_supported** = macro-F1 over classes with >= {C.MIN_SUPPORT} test samples",
        "",
        "| Rank | Model | Overall acc | Macro-F1 | Macro-F1 (supported) | #features |",
        "|---|---|---|---|---|---|",
    ]
    for i, r in lb.iterrows():
        lines.append(f"| {i+1} | {r['model']} | {r['overall_accuracy']:.3f} | "
                     f"{r['macro_f1']:.3f} | {r['macro_f1_supported']:.3f} | "
                     f"{int(r['n_features'])} |")
    lines += [
        "",
        "**Reading the result:** the embedding rows (B2*) should dominate B1 "
        "(geography) and B0 (majority). Class-balancing (B2w weighted / B2s "
        "SMOTE) targets rarer classes -- compare their macro-F1 against plain B2. "
        "Ultra-rare, spatially-clustered classes (e.g. mangroves) may remain hard "
        "because a spatial-block split can place them entirely in train or test.",
    ]
    return "\n".join(lines)
