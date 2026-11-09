"""Visualisation: confusion matrix, per-class F1, interactive prediction map."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless / no display needed
import matplotlib.pyplot as plt

from . import config as C


def confusion_png(result, path):
    cm = np.array(result["confusion_matrix"], dtype=float)
    names = result["class_names"]
    cmn = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)  # row-normalised
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Bharat-EO-Bench v0 -- normalised confusion matrix")
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cmn[i,j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def per_class_f1_png(result, path):
    pc = result["per_class_f1"]
    names = list(pc.keys()); vals = list(pc.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(names)), vals, color="#2E5496")
    ax.set_ylim(0, 1); ax.set_ylabel("F1")
    ax.set_title("Per-class F1 (AlphaEarth + XGBoost)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    return path


def prediction_map_html(df, result, path):
    """Interactive map of held-out test points coloured by predicted class."""
    try:
        import folium
    except Exception:
        return None
    idx = result["test_idx"]
    sub = df.iloc[idx].copy()
    classes = result["classes"]
    sub["pred"] = [classes[p] for p in result["y_pred"]]
    sub["correct"] = (sub["label"].to_numpy() == np.array(
        [classes[p] for p in result["y_pred"]]))
    center = [float(sub["lat"].mean()), float(sub["lon"].mean())]
    m = folium.Map(location=center, zoom_start=5, tiles="CartoDB positron")
    for _, r in sub.iterrows():
        col = C.CLASS_COLORS.get(int(r["pred"]), "#333333")
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])], radius=3,
            color=col, fill=True, fill_opacity=0.8,
            popup=f"pred: {C.CLASS_NAMES.get(int(r['pred']))} | "
                  f"true: {C.CLASS_NAMES.get(int(r['label']))}",
        ).add_to(m)
    m.save(str(path))
    return path
