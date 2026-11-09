"""Task: Assam flood (Brahmaputra, June-July 2022).

Labels (3-class) come from Sentinel-1 SAR during the flood (cloud-proof) minus
JRC permanent water:  0 dry | 1 permanent_water | 2 flood_water.

The benchmark is FLOOD SUSCEPTIBILITY: predict that flood extent from STATIC,
pre-event predictors only (AlphaEarth embedding + terrain + pre-flood SAR) --
during-flood SAR is excluded from the features so the task is non-circular.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

from . import config as C
from . import utils

FLOOD_CLASSES = {0: "dry", 1: "permanent_water", 2: "flood_water"}
FLOOD_COLORS = {0: "#d9c9a3", 1: "#0064c8", 2: "#e8000b"}
PRE = ("2022-01-01", "2022-03-15")       # dry-season reference
FLD = ("2022-06-15", "2022-07-20")       # flood window
EXTRA = ["VV_pre", "VH_pre", "elevation", "slope"]   # static non-embedding feats
FLOOD_CSV = C.DATA_RAW / "assam_flood_2022.csv"


# ----------------------------- Earth Engine --------------------------------
def _init(project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(); ee.Initialize(project=project)
    return ee


def _build(ee):
    assam = (ee.FeatureCollection(C.GAUL1)
             .filter(ee.Filter.eq("ADM1_NAME", "Assam"))
             .geometry().simplify(maxError=2000))
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filterBounds(assam).select(["VV", "VH"]))
    pre = s1.filterDate(*PRE).median()                       # dry-season backscatter
    fld = s1.filterDate(*FLD).min()                          # min over flood = inundation
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    slope = ee.Terrain.slope(dem)
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0)
    emb = (ee.ImageCollection(C.EMBED_COLLECTION)
           .filterDate("2021-01-01", "2022-01-01").filterBounds(assam).mosaic())

    # SAR water during the flood (dark in VV & VH, on near-flat terrain)
    water = fld.select("VV").lt(-16).And(fld.select("VH").lt(-22)).And(slope.lt(5))
    permanent = jrc.gte(50)
    label = (ee.Image(0)
             .where(permanent, 1)
             .where(water.And(permanent.Not()), 2)
             .rename("label").toInt())

    feats = (emb
             .addBands(pre.rename(["VV_pre", "VH_pre"]))
             .addBands(dem.rename("elevation"))
             .addBands(slope.rename("slope")))
    return assam, feats.addBands(label)


def export_flood(project=C.EE_PROJECT, n_points=12000, out_csv=None, batch=4500):
    ee = _init(project)
    assam, stacked = _build(ee)
    cols = C.EMB_COLS + EXTRA + ["label", "lon", "lat"]

    def add_ll(f):
        c = f.geometry().coordinates()
        return f.set({"lon": c.get(0), "lat": c.get(1)})

    frames, done, i = [], 0, 0
    while done < n_points:
        npts = min(batch, n_points - done)
        print(f"[flood] batch {i+1}: {npts} points (seed {C.SEED+i}) ...")
        pts = ee.FeatureCollection.randomPoints(assam, npts, C.SEED + i)
        s = stacked.sampleRegions(collection=pts, scale=C.SCALE,
                                  geometries=True, tileScale=8)
        s = s.filter(ee.Filter.notNull(["label", "A00", "VV_pre", "elevation"])).map(add_ll)
        feats = s.getInfo()["features"]
        frames.append(pd.DataFrame.from_records(
            [{k: ft["properties"].get(k) for k in cols} for ft in feats]))
        done += npts; i += 1
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["lon", "lat"]).reset_index(drop=True)
    out_csv = out_csv or FLOOD_CSV
    df.to_csv(out_csv, index=False)
    print(f"[flood] wrote {len(df):,} rows -> {out_csv}")
    return out_csv


# ----------------------------- benchmark -----------------------------------
def _eval(df, feature_cols):
    tr, te = utils.spatial_block_split(df)
    X = df[feature_cols].to_numpy("float32"); y = df["label"].to_numpy(int)
    m = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1,
                      subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                      objective="multi:softprob", num_class=3, n_jobs=-1,
                      random_state=C.SEED)
    m.fit(X[tr], y[tr]); p = m.predict(X[te]); yte = y[te]
    flood_f1 = f1_score(yte, p, labels=[2], average="macro", zero_division=0)
    flood_rec = recall_score(yte, p, labels=[2], average="macro", zero_division=0)
    return {"overall_accuracy": float(accuracy_score(yte, p)),
            "macro_f1": float(f1_score(yte, p, average="macro", zero_division=0)),
            "flood_f1": float(flood_f1), "flood_recall": float(flood_rec),
            "cm": confusion_matrix(yte, p, labels=[0, 1, 2]).tolist(),
            "test_idx": np.where(te)[0], "pred": p.tolist(), "model": m}


def run_flood(csv, tag="assam_flood_2022"):
    df = pd.read_csv(csv).dropna(subset=["label", "lon", "lat"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    dist = df["label"].value_counts().sort_index()
    print(f"[flood] {len(df):,} pts | " +
          " ".join(f"{FLOOD_CLASSES[k]}={v}" for k, v in dist.items()))

    groups = {
        "F0  Majority (dry)": None,
        "F1  Terrain only (elev+slope)": ["elevation", "slope"],
        "F2  Pre-flood SAR only": ["VV_pre", "VH_pre"],
        "F3  AlphaEarth embedding": list(C.EMB_COLS),
        "F4  Embedding + terrain + pre-SAR (all static)": list(C.EMB_COLS) + EXTRA,
    }
    rows, best, best_f1 = [], None, -1
    tr, te = utils.spatial_block_split(df)
    maj = int(df.loc[tr, "label"].mode().iloc[0]); yte = df.loc[te, "label"].to_numpy()
    from sklearn.metrics import accuracy_score as acc, f1_score as f1
    rows.append(("F0  Majority (dry)", float(acc(yte, np.full(len(yte), maj))),
                 float(f1(yte, np.full(len(yte), maj), average="macro", zero_division=0)),
                 0.0, 0.0))
    for name, feats in groups.items():
        if feats is None:
            continue
        r = _eval(df, feats)
        rows.append((name, r["overall_accuracy"], r["macro_f1"], r["flood_f1"], r["flood_recall"]))
        if r["flood_f1"] > best_f1:
            best, best_f1 = r, r["flood_f1"]
    lb = pd.DataFrame(rows, columns=["model", "overall_accuracy", "macro_f1",
                                     "flood_f1", "flood_recall"]).sort_values(
        "flood_f1", ascending=False).reset_index(drop=True)
    print("\n=== Assam Flood 2022 -- susceptibility leaderboard ===")
    print(lb.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # report + figures
    md = ["# Assam Flood 2022 -- Bharat-EO-Bench (flood susceptibility)", "",
          f"- Region: Assam   Event: Jun-Jul 2022 (Brahmaputra)   Points: {len(df):,}",
          f"- Labels from Sentinel-1 SAR (cloud-proof) + JRC permanent water",
          f"- Predictors: STATIC only (no during-flood SAR) -> non-circular susceptibility",
          f"- Class mix: " + ", ".join(f"{FLOOD_CLASSES[k]} {v}" for k, v in dist.items()),
          "", "| Rank | Model | Overall acc | Macro-F1 | Flood F1 | Flood recall |",
          "|---|---|---|---|---|---|"]
    for i, r in lb.iterrows():
        md.append(f"| {i+1} | {r['model']} | {r['overall_accuracy']:.3f} | "
                  f"{r['macro_f1']:.3f} | {r['flood_f1']:.3f} | {r['flood_recall']:.3f} |")
    (C.OUT_REP / f"{tag}_leaderboard.md").write_text("\n".join(md), encoding="utf-8")
    lb.to_csv(C.OUT_REP / f"{tag}_leaderboard.csv", index=False)

    if best is not None:
        _flood_map(df, best, C.OUT_FIG / f"{tag}_pred_map.html")
        _cm_png(best, C.OUT_FIG / f"{tag}_confusion.png")
    print(f"[flood] report -> {C.OUT_REP / (tag + '_leaderboard.md')}")
    return lb


def _cm_png(best, path):
    cm = np.array(best["cm"], float); names = [FLOOD_CLASSES[i] for i in range(3)]
    cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(5, 4.2)); im = ax.imshow(cmn, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Assam flood -- confusion")
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{cmn[a,b]:.2f}", ha="center", va="center",
                    color="white" if cmn[a, b] > 0.5 else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig); return path


def _flood_map(df, best, path):
    try:
        import folium
    except Exception:
        return None
    sub = df.iloc[best["test_idx"]].copy(); sub["pred"] = best["pred"]
    m = folium.Map([float(sub.lat.mean()), float(sub.lon.mean())], zoom_start=7,
                   tiles="CartoDB positron")
    for _, r in sub.iterrows():
        folium.CircleMarker([float(r.lat), float(r.lon)], radius=2.5,
                            color=FLOOD_COLORS.get(int(r["pred"]), "#333"), fill=True,
                            fill_opacity=0.8,
                            popup=f"pred {FLOOD_CLASSES[int(r['pred'])]} / true {FLOOD_CLASSES[int(r['label'])]}").add_to(m)
    m.save(str(path)); return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=C.EE_PROJECT)
    ap.add_argument("--points", type=int, default=12000)
    a = ap.parse_args()
    csv = export_flood(a.project, a.points)
    run_flood(csv)
