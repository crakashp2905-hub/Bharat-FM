"""Groundwater downscaling: GRACE (~110 km) -> ~1.5 km via ML, for NW India.

Method (statistical downscaling):
  * TRAIN: random points over India; sample FINE-resolution surface predictors
    (AlphaEarth embedding + rainfall + NDVI + elevation + cropland) and the
    GRACE TWS trend (coarse value at each point). Learn surface -> trend.
  * APPLY: dense fine grid over Punjab + Haryana; predict the trend at fine
    resolution -> a downscaled groundwater-stress map GRACE alone can't make.
  * AGGREGATE: mean predicted trend per district (GAUL L2) -> ranked hotspots.

HONEST CAVEATS: no well validation (CGWB not in GEE); assumes the surface<->
groundwater relationship is scale-invariant; real InSAR subsidence (Sentinel-1
SLC, off-GEE via ASF HyP3) would add a direct compaction signal.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

from . import config as C
from . import utils
from .groundwater import _init

NW_STATES = ["Punjab", "Haryana"]
FEATS = ["rain", "ndvi", "elevation", "cropland"]
FINE_SCALE = 1500          # ~1.5 km downscaling target
TRAIN_CSV = C.DATA_RAW / "gw_downscale_train.csv"
GRID_CSV = C.DATA_RAW / "gw_downscale_grid_nw.csv"


def _fine_stack(ee):
    india = (ee.FeatureCollection(C.GAUL0)
             .filter(ee.Filter.eq("ADM0_NAME", C.COUNTRY))
             .geometry().simplify(maxError=5000))

    grace = (ee.ImageCollection("NASA/GRACE/MASS_GRIDS/LAND")
             .select("lwe_thickness_jpl").filterDate("2003-01-01", "2017-01-01"))

    def add_t(img):
        t = ee.Date(img.get("system:time_start")).difference(
            ee.Date("2000-01-01"), "year")
        return img.addBands(ee.Image(t).rename("t").float()).select(
            ["t", "lwe_thickness_jpl"])
    gw_trend = grace.map(add_t).reduce(ee.Reducer.linearFit()).select(
        "scale").rename("gw_trend")

    rain = ee.Image("WORLDCLIM/V1/BIO").select("bio12").rename("rain")
    ndvi = (ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate("2015-01-01", "2016-01-01").select("NDVI").mean()
            .multiply(0.0001).rename("ndvi"))
    elev = ee.Image("USGS/SRTMGL1_003").select("elevation")
    mcd = ee.Image(ee.ImageCollection("MODIS/061/MCD12Q1")
                   .filterDate("2015-01-01", "2016-01-01").first()).select("LC_Type1")
    cropland = mcd.eq(12).Or(mcd.eq(14)).rename("cropland")
    emb = (ee.ImageCollection(C.EMBED_COLLECTION)
           .filterDate("2021-01-01", "2022-01-01").filterBounds(india).mosaic())

    stacked = (gw_trend.addBands(rain).addBands(ndvi).addBands(elev)
               .addBands(cropland).addBands(emb))
    return india, stacked


def _sample_batched(ee, stacked, region, n_points, extra_cols=()):
    cols = ["gw_trend"] + FEATS + C.EMB_COLS + list(extra_cols) + ["lon", "lat"]

    def add_ll(f):
        c = f.geometry().coordinates()
        return f.set({"lon": c.get(0), "lat": c.get(1)})
    frames, done, i = [], 0, 0
    while done < n_points:
        npts = min(4500, n_points - done)
        pts = ee.FeatureCollection.randomPoints(region, npts, C.SEED + i)
        s = stacked.sampleRegions(collection=pts, scale=FINE_SCALE,
                                  geometries=True, tileScale=8)
        s = s.filter(ee.Filter.notNull(["gw_trend", "A00", "rain"])).map(add_ll)
        feats = s.getInfo()["features"]
        frames.append(pd.DataFrame.from_records(
            [{k: ft["properties"].get(k) for k in cols} for ft in feats]))
        print(f"[ds] batch {i+1}: +{len(frames[-1])} pts")
        done += npts; i += 1
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["lon", "lat"]).reset_index(drop=True)


def export_train(project=C.EE_PROJECT, n=7000, out_csv=None):
    ee = _init(project)
    india, stacked = _fine_stack(ee)
    print("[ds] sampling TRAIN points over India (fine features) ...")
    df = _sample_batched(ee, stacked, india, n)
    out_csv = out_csv or TRAIN_CSV
    df.to_csv(out_csv, index=False)
    print(f"[ds] train: {len(df):,} rows -> {out_csv}")
    return out_csv


def export_grid(project=C.EE_PROJECT, n=12000, out_csv=None):
    ee = _init(project)
    _, stacked = _fine_stack(ee)
    nw = (ee.FeatureCollection(C.GAUL1)
          .filter(ee.Filter.inList("ADM1_NAME", NW_STATES)))
    region = nw.geometry().simplify(maxError=2000)
    # district id band for aggregation
    gaul2 = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
        ee.Filter.inList("ADM1_NAME", NW_STATES))
    dist_img = ee.Image().int().paint(gaul2, "ADM2_CODE").rename("district")
    stacked = stacked.addBands(dist_img)
    print("[ds] sampling APPLY grid over Punjab+Haryana ...")
    df = _sample_batched(ee, stacked, region, n, extra_cols=["district"])
    out_csv = out_csv or GRID_CSV
    df.to_csv(out_csv, index=False)
    print(f"[ds] grid: {len(df):,} rows -> {out_csv}")
    return out_csv


def _district_names(project):
    ee = _init(project)
    gaul2 = ee.FeatureCollection("FAO/GAUL/2015/level2").filter(
        ee.Filter.inList("ADM1_NAME", NW_STATES))
    info = gaul2.select(["ADM2_CODE", "ADM2_NAME", "ADM1_NAME"]).getInfo()
    return {f["properties"]["ADM2_CODE"]:
            (f["properties"]["ADM2_NAME"], f["properties"]["ADM1_NAME"])
            for f in info["features"]}


def run_downscale(train_csv, grid_csv, project=C.EE_PROJECT, tag="gw_downscale_nw"):
    tr = pd.read_csv(train_csv).dropna(subset=["gw_trend"]).reset_index(drop=True)
    grid = pd.read_csv(grid_csv).dropna(subset=FEATS + ["A00"]).reset_index(drop=True)
    feat_cols = FEATS + list(C.EMB_COLS)

    # spatial-CV score of the downscaling model (held-out regions)
    m_tr, m_te = utils.spatial_block_split(tr, block_deg=2.0)
    reg = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                       n_jobs=-1, random_state=C.SEED)
    reg.fit(tr.loc[m_tr, feat_cols].to_numpy("float32"), tr.loc[m_tr, "gw_trend"].to_numpy("float32"))
    p_te = reg.predict(tr.loc[m_te, feat_cols].to_numpy("float32"))
    r2 = r2_score(tr.loc[m_te, "gw_trend"], p_te)
    mae = mean_absolute_error(tr.loc[m_te, "gw_trend"], p_te)
    print(f"[ds] downscaling model spatial-CV: R2 {r2:.3f}  MAE {mae:.3f} cm/yr")

    # refit on all train, predict the fine NW grid
    reg.fit(tr[feat_cols].to_numpy("float32"), tr["gw_trend"].to_numpy("float32"))
    grid["pred_trend"] = reg.predict(grid[feat_cols].to_numpy("float32"))
    print(f"[ds] downscaled grid: {len(grid):,} cells | "
          f"GRACE-native mean {grid.gw_trend.mean():.2f} vs downscaled mean "
          f"{grid.pred_trend.mean():.2f} cm/yr; range "
          f"[{grid.pred_trend.min():.2f}, {grid.pred_trend.max():.2f}]")

    # district ranking
    names = _district_names(project)
    g = grid.dropna(subset=["district"]).copy()
    g["district"] = g["district"].astype(int)
    agg = (g.groupby("district")["pred_trend"].agg(["mean", "count"])
           .reset_index().sort_values("mean"))
    agg["name"] = agg["district"].map(lambda c: names.get(c, ("?", "?"))[0])
    agg["state"] = agg["district"].map(lambda c: names.get(c, ("?", "?"))[1])
    agg = agg[agg["count"] >= 10]
    top = agg.head(15)
    print("\n=== Most groundwater-depleted districts (downscaled, cm/yr) ===")
    print(top[["name", "state", "mean", "count"]].to_string(
        index=False, float_format=lambda x: f"{x:.2f}"))

    # report
    md = ["# Groundwater downscaling -- Punjab + Haryana (GRACE -> ~1.5 km)", "",
          f"- Downscaled {len(grid):,} cells at ~{FINE_SCALE/1000:.1f} km "
          f"(GRACE native ~110 km)",
          f"- Downscaling model spatial-CV R2 = {r2:.3f}, MAE = {mae:.3f} cm/yr",
          "- Trend in cm water-equivalent/yr; negative = depletion",
          "- CAVEAT: unvalidated vs wells; scale-invariance assumed; no InSAR", "",
          "## Most-depleted districts (mean predicted trend)", "",
          "| Rank | District | State | Trend (cm/yr) | cells |", "|---|---|---|---|---|"]
    for i, r in top.reset_index(drop=True).iterrows():
        md.append(f"| {i+1} | {r['name']} | {r['state']} | {r['mean']:.2f} | {int(r['count'])} |")
    (C.OUT_REP / f"{tag}_report.md").write_text("\n".join(md), encoding="utf-8")
    agg.to_csv(C.OUT_REP / f"{tag}_districts.csv", index=False)
    _downscale_map(grid, C.OUT_FIG / f"{tag}_map.html")
    print(f"\n[ds] report -> {C.OUT_REP / (tag + '_report.md')}")
    return top


def _downscale_map(grid, path):
    try:
        import folium, branca.colormap as cm
    except Exception:
        return None
    lim = float(np.nanpercentile(np.abs(grid.pred_trend), 95))
    colmap = cm.LinearColormap(["#67000d", "#f7f7f7", "#08306b"],
                               vmin=-lim, vmax=lim,
                               caption="Downscaled groundwater trend cm/yr (red=depletion)")
    m = folium.Map([30.5, 75.8], zoom_start=7, tiles="CartoDB positron")
    for _, r in grid.iterrows():
        folium.CircleMarker([float(r.lat), float(r.lon)], radius=2.5, color=None,
                            fill=True, fill_color=colmap(float(r.pred_trend)),
                            fill_opacity=0.85,
                            popup=f"{r.pred_trend:.2f} cm/yr").add_to(m)
    colmap.add_to(m); m.save(str(path)); return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=C.EE_PROJECT)
    a = ap.parse_args()
    tc = export_train(a.project)
    gc = export_grid(a.project)
    run_downscale(tc, gc, a.project)
