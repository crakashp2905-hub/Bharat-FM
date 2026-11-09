"""Task: groundwater depletion over NW India (Punjab focus).

HONEST SCOPE: groundwater is sub-surface; the only satellite signal here is
GRACE (terrestrial water storage), which is very coarse (~1 deg / ~110 km).
The blueprint's real product fuses InSAR + GRACE + CGWB wells; this is the
tractable GRACE + surface-driver version:

  1. GRACE TWS linear trend (cm/yr, 2003-2016) -> map the depletion hotspot.
  2. Regression at ~50 km cells: do surface drivers (rainfall, greenness,
     cropland fraction, AlphaEarth embedding) explain the depletion trend?

This is coarse and correlational, NOT a field-scale groundwater map.
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

DRIVERS = ["rain", "ndvi", "elevation", "cropland_frac"]
GW_CSV = C.DATA_RAW / "india_groundwater_grace.csv"
SAMPLE_SCALE = 50000     # ~0.5 deg cells -> ~1300 over India


def _init(project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate(); ee.Initialize(project=project)
    return ee


def _build(ee):
    india = (ee.FeatureCollection(C.GAUL0)
             .filter(ee.Filter.eq("ADM0_NAME", C.COUNTRY))
             .geometry().simplify(maxError=5000))

    # --- GRACE TWS linear trend (cm / year) ---------------------------------
    grace = (ee.ImageCollection("NASA/GRACE/MASS_GRIDS/LAND")
             .select("lwe_thickness_jpl").filterDate("2003-01-01", "2017-01-01"))

    def add_t(img):
        t = ee.Date(img.get("system:time_start")).difference(
            ee.Date("2000-01-01"), "year")
        return img.addBands(ee.Image(t).rename("t").float()).select(
            ["t", "lwe_thickness_jpl"])
    fit = grace.map(add_t).reduce(ee.Reducer.linearFit())
    gw_trend = fit.select("scale").rename("gw_trend")     # cm of water / year

    # --- surface drivers (cheap, static climatologies) ----------------------
    rain = ee.Image("WORLDCLIM/V1/BIO").select("bio12").rename("rain")    # annual precip mm
    ndvi = (ee.ImageCollection("MODIS/061/MOD13Q1")
            .filterDate("2015-01-01", "2016-01-01").select("NDVI").mean()
            .multiply(0.0001).rename("ndvi"))
    elev = ee.Image("USGS/SRTMGL1_003").select("elevation")
    mcd = ee.Image(ee.ImageCollection("MODIS/061/MCD12Q1")
                   .filterDate("2015-01-01", "2016-01-01").first()).select("LC_Type1")
    crop = mcd.eq(12).Or(mcd.eq(14))                      # croplands / cropland mosaic
    crop_frac = (crop.reduceResolution(ee.Reducer.mean(), maxPixels=20000)
                 .reproject(crs="EPSG:4326", scale=SAMPLE_SCALE).rename("cropland_frac"))
    emb = (ee.ImageCollection(C.EMBED_COLLECTION)
           .filterDate("2021-01-01", "2022-01-01").filterBounds(india).mosaic())

    stacked = (gw_trend.addBands(rain).addBands(ndvi).addBands(elev)
               .addBands(crop_frac).addBands(emb))
    return india, stacked


def export_gw(project=C.EE_PROJECT, out_csv=None, n_cells=3000):
    ee = _init(project)
    india, stacked = _build(ee)
    print(f"[gw] sampling GRACE + drivers at ~{SAMPLE_SCALE/1000:.0f} km cells ...")
    fc = stacked.sample(region=india, scale=SAMPLE_SCALE, numPixels=n_cells,
                        seed=C.SEED, geometries=True, tileScale=8, dropNulls=True)

    def add_ll(f):
        c = f.geometry().coordinates()
        return f.set({"lon": c.get(0), "lat": c.get(1)})
    cols = ["gw_trend"] + DRIVERS + C.EMB_COLS + ["lon", "lat"]
    feats = fc.map(add_ll).getInfo()["features"]
    df = pd.DataFrame.from_records(
        [{k: ft["properties"].get(k) for k in cols} for ft in feats])
    df = df.dropna(subset=["gw_trend", "rain", "lon", "lat"]).reset_index(drop=True)
    out_csv = out_csv or GW_CSV
    df.to_csv(out_csv, index=False)
    print(f"[gw] wrote {len(df):,} cells -> {out_csv}")
    return out_csv


def punjab_trend(project=C.EE_PROJECT):
    ee = _init(project)
    _, stacked = _build(ee)
    punjab = (ee.FeatureCollection(C.GAUL1)
              .filter(ee.Filter.eq("ADM1_NAME", "Punjab")).geometry())
    val = stacked.select("gw_trend").reduceRegion(
        ee.Reducer.mean(), punjab, scale=SAMPLE_SCALE, maxPixels=1e9).getInfo()
    return val.get("gw_trend")


# ----------------------------- regression ----------------------------------
def _reg(df, feature_cols):
    tr, te = utils.spatial_block_split(df, block_deg=2.0)   # bigger blocks (coarse data)
    X = df[feature_cols].to_numpy("float32"); y = df["gw_trend"].to_numpy("float32")
    m = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                     n_jobs=-1, random_state=C.SEED)
    m.fit(X[tr], y[tr]); p = m.predict(X[te])
    return {"r2": float(r2_score(y[te], p)),
            "mae": float(mean_absolute_error(y[te], p)),
            "model": m, "features": feature_cols}


def run_gw(csv, project=C.EE_PROJECT, tag="india_groundwater"):
    df = pd.read_csv(csv)
    print(f"[gw] {len(df):,} cells | gw_trend cm/yr: "
          f"mean {df.gw_trend.mean():.2f}, min {df.gw_trend.min():.2f}, "
          f"max {df.gw_trend.max():.2f}")
    try:
        pj = punjab_trend(project)
        print(f"[gw] Punjab mean GRACE trend: {pj:.2f} cm water/yr "
              f"({'DEPLETION' if pj and pj < 0 else 'gain'})")
    except Exception as e:
        pj = None; print(f"[gw] (Punjab stat skipped: {e})")

    groups = {
        "G1  Rainfall only": ["rain"],
        "G2  Climate + agriculture": DRIVERS,
        "G3  + AlphaEarth embedding": DRIVERS + list(C.EMB_COLS),
    }
    rows, best = [], None
    for name, feats in groups.items():
        r = _reg(df, feats)
        rows.append((name, r["r2"], r["mae"], len(feats)))
        if name.startswith("G2"):
            best = r
    lb = pd.DataFrame(rows, columns=["model", "R2", "MAE_cm_yr", "n_features"])
    print("\n=== Groundwater (GRACE trend) regression -- spatial CV ===")
    print(lb.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # feature importance of the climate+agri model
    imp = sorted(zip(best["features"], best["model"].feature_importances_),
                 key=lambda x: -x[1])
    print("\nDrivers of depletion (G2 importance):")
    for k, v in imp:
        print(f"  {k:<16} {v:.3f}")

    md = ["# Groundwater depletion (GRACE) -- NW India / Punjab", "",
          f"- GRACE TWS linear trend 2003-2016 (cm water/yr); {len(df):,} ~50 km cells",
          f"- Punjab mean trend: {pj:.2f} cm/yr" if pj is not None else "- Punjab trend: n/a",
          "- COARSE & correlational (GRACE ~110 km); not a field-scale product", "",
          "| Model | R2 (spatial CV) | MAE (cm/yr) | #features |", "|---|---|---|---|"]
    for _, r in lb.iterrows():
        md.append(f"| {r['model']} | {r['R2']:.3f} | {r['MAE_cm_yr']:.3f} | {int(r['n_features'])} |")
    md += ["", "**Top drivers (G2):** " + ", ".join(f"{k} {v:.2f}" for k, v in imp)]
    (C.OUT_REP / f"{tag}_report.md").write_text("\n".join(md), encoding="utf-8")
    lb.to_csv(C.OUT_REP / f"{tag}_leaderboard.csv", index=False)
    _trend_map(df, C.OUT_FIG / f"{tag}_trend_map.html")
    print(f"[gw] report -> {C.OUT_REP / (tag + '_report.md')}")
    return lb


def _trend_map(df, path):
    try:
        import folium, branca.colormap as cm
    except Exception:
        return None
    vmin, vmax = float(df.gw_trend.quantile(0.05)), float(df.gw_trend.quantile(0.95))
    lim = max(abs(vmin), abs(vmax))
    colmap = cm.LinearColormap(["#b2182b", "#f7f7f7", "#2166ac"], vmin=-lim, vmax=lim,
                               caption="GRACE TWS trend (cm/yr): red = depletion")
    m = folium.Map([23.0, 80.0], zoom_start=5, tiles="CartoDB positron")
    for _, r in df.iterrows():
        folium.CircleMarker([float(r.lat), float(r.lon)], radius=4,
                            color=None, fill=True, fill_color=colmap(float(r.gw_trend)),
                            fill_opacity=0.85,
                            popup=f"{r.gw_trend:.2f} cm/yr").add_to(m)
    colmap.add_to(m); m.save(str(path)); return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=C.EE_PROJECT)
    a = ap.parse_args()
    csv = export_gw(a.project)
    run_gw(csv, a.project)
