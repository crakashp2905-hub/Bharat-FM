"""Neural-network ('deep model') baseline for the benchmark.

Uses scikit-learn's MLPClassifier so it runs on CPU with no extra downloads
(the PyTorch version for GPU scaling lives in notebooks/finetune_gpu.py).
Reuses the SAME spatial-block split and metrics as the XGBoost heads, so the
numbers drop straight into the Bharat-EO-Bench leaderboard.
"""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from . import config as C
from .train import prepare, summarize, _oversample


def train_eval_mlp(df, feature_cols, hidden=(256, 128), max_iter=400,
                   balance=None, seed=C.SEED):
    Xtr, ytr, Xte, yte, classes, test_idx = prepare(df, feature_cols)

    scaler = StandardScaler().fit(Xtr)
    Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)

    if balance in ("smote", "weight"):
        Xtr, ytr = _oversample(Xtr, ytr)   # neural nets have no class_weight knob

    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu",
                        alpha=1e-4, batch_size=256, learning_rate_init=1e-3,
                        max_iter=max_iter, early_stopping=True,
                        n_iter_no_change=15, random_state=seed)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    res = summarize(classes, yte, pred, len(ytr), len(yte), test_idx,
                    feature_cols, f"mlp/{balance or 'plain'}", model=None)
    res["device"] = "cpu (sklearn MLPClassifier)"
    return res
