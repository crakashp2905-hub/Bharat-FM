"""Train a lightweight head (XGBoost) on satellite embeddings -> land cover.

This is the laptop-feasible core: frozen pretrained embeddings + a small
trainable classifier. Trains in seconds-to-minutes on CPU.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config as C
from . import utils


def prepare(df, feature_cols):
    """Spatial-block split + label remap. Returns the pieces every model needs."""
    train_mask, test_mask = utils.spatial_block_split(df)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["label"].to_numpy(dtype=int)
    classes = sorted(np.unique(y_raw).tolist())
    to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([to_idx[v] for v in y_raw])
    return (X[train_mask], y[train_mask], X[test_mask], y[test_mask],
            classes, np.where(test_mask)[0])


def summarize(classes, yte, pred, n_train, n_test, test_idx, feature_cols,
              balance, model=None):
    """Build the standard result dict (shared by XGBoost and the MLP)."""
    oa = float(accuracy_score(yte, pred))
    macro_f1 = float(f1_score(yte, pred, average="macro", zero_division=0))
    sup = np.bincount(yte, minlength=len(classes))
    keep = [i for i in range(len(classes)) if sup[i] >= C.MIN_SUPPORT]
    macro_f1_sup = (float(f1_score(yte, pred, labels=keep, average="macro",
                                   zero_division=0)) if keep else 0.0)
    per_class_f1 = f1_score(yte, pred, average=None,
                            labels=list(range(len(classes))), zero_division=0)
    per_class = {C.CLASS_NAMES.get(classes[i], str(classes[i])): float(per_class_f1[i])
                 for i in range(len(classes))}
    cm = confusion_matrix(yte, pred, labels=list(range(len(classes))))
    report = classification_report(
        yte, pred, labels=list(range(len(classes))),
        target_names=[C.CLASS_NAMES.get(c, str(c)) for c in classes],
        zero_division=0, output_dict=True)
    return {
        "overall_accuracy": oa,
        "macro_f1": macro_f1,
        "macro_f1_supported": macro_f1_sup,
        "n_supported_classes": len(keep),
        "balance": balance or "none",
        "per_class_f1": per_class,
        "n_train": int(n_train),
        "n_test": int(n_test),
        "classes": classes,
        "class_names": [C.CLASS_NAMES.get(c, str(c)) for c in classes],
        "confusion_matrix": cm.tolist(),
        "report": report,
        "model": model,
        "test_idx": test_idx,
        "y_true": yte.tolist(),
        "y_pred": [int(p) for p in pred],
        "feature_cols": list(feature_cols),
    }


def _oversample(Xtr, ytr):
    """Balance the TRAIN set with SMOTE (after the spatial split -> no leakage)."""
    from collections import Counter
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    cnt = Counter(ytr.tolist())
    if min(cnt.values()) < 6:
        strat = {c: max(n, 6) for c, n in cnt.items() if n < 6}
        Xtr, ytr = RandomOverSampler(sampling_strategy=strat,
                                     random_state=C.SEED).fit_resample(Xtr, ytr)
        cnt = Counter(ytr.tolist())
    k = max(1, min(5, min(cnt.values()) - 1))
    return SMOTE(k_neighbors=k, random_state=C.SEED).fit_resample(Xtr, ytr)


def train_eval(df: pd.DataFrame, feature_cols, params=None, balance=None):
    """Fit XGBoost on feature_cols, evaluate on a spatial-block test set.

    balance: None | "weight" (class-balanced sample weights) | "smote".
    """
    params = dict(params or C.XGB_PARAMS)
    Xtr, ytr, Xte, yte, classes, test_idx = prepare(df, feature_cols)
    model = XGBClassifier(num_class=len(classes), objective="multi:softprob", **params)
    if balance == "smote":
        Xtr, ytr = _oversample(Xtr, ytr)
        model.fit(Xtr, ytr)
    elif balance == "weight":
        sw = compute_sample_weight(class_weight="balanced", y=ytr)
        model.fit(Xtr, ytr, sample_weight=sw)
    else:
        model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return summarize(classes, yte, pred, len(ytr), len(yte), test_idx,
                     feature_cols, balance, model)
