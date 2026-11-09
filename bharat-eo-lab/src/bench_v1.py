"""Bharat-EO-Bench v1 construction: agro-climatic zonation + stratified,
time-series, geographically-split point sampling.  See docs/BENCHMARK_SPEC.md.
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from . import config as C
from .groundwater import _init

ZONE_NAMES = {  # rainfall(arid,semiarid,subhumid,humid,perhumid) x physio(plain,upland,hill)
    0: "arid-plain", 1: "arid-upland", 2: "arid-hill",
    3: "semiarid-plain", 4: "semiarid-upland", 5: "semiarid-hill",
    6: "subhumid-plain", 7: "subhumid-upland", 8: "subhumid-hill",
    9: "humid-plain", 10: "humid-upland", 11: "humid-hill",
    12: "perhumid-plain", 13: "perhumid-upland", 14: "perhumid-hill",
}


def agro_zone(ee):
    """15-zone agro-climatic layer = rainfall regime x physiography."""
    rain = ee.Image("WORLDCLIM/V1/BIO").select("bio12")           # annual precip mm
    elev = ee.Image("USGS/SRTMGL1_003").select("elevation")
    rcl = (ee.Image(0).where(rain.gte(500), 1).where(rain.gte(1000), 2)
           .where(rain.gte(1500), 3).where(rain.gte(2500), 4))
    ecl = ee.Image(0).where(elev.gte(300), 1).where(elev.gte(1000), 2)
    return rcl.multiply(3).add(ecl).rename("zone").toInt()


def india(ee):
    return (ee.FeatureCollection(C.GAUL0).filter(ee.Filter.eq("ADM0_NAME", C.COUNTRY))
            .geometry().simplify(5000))


def verify_zones(project=C.EE_PROJECT):
    ee = _init(project)
    zone = agro_zone(ee)
    hist = zone.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(), geometry=india(ee),
        scale=5000, maxPixels=1e10, bestEffort=True).getInfo()["zone"]
    total = sum(hist.values())
    print("=== agro-climatic zone distribution over India (approx area %) ===")
    for z in range(15):
        pct = 100 * hist.get(str(z), 0) / total
        bar = "#" * int(pct / 2)
        print(f"  {z:>2} {ZONE_NAMES[z]:<18} {pct:5.1f}%  {bar}")
    print(f"  zones with >1% area: {sum(1 for z in range(15) if 100*hist.get(str(z),0)/total > 1)}/15")
    return hist


QUARTERS = [("q1", 1, 4), ("q2", 4, 7), ("q3", 7, 10), ("q4", 10, 13)]  # rabi/zaid/kharif/post
TS_COLS = [f"{v}_{q}" for q, _, _ in QUARTERS for v in ("ndvi", "vv", "vh")]
T2_COLS = C.EMB_COLS + TS_COLS + ["elevation", "slope", "zone", "label", "lon", "lat"]
POOL_CSV = C.DATA_RAW / "bench_v1_t2_pool.csv"
BENCH_DIR = C.ROOT / "data" / "bench_v1"


def _t2_stack(ee, year=2021):
    g = india(ee)
    emb = (ee.ImageCollection(C.EMBED_COLLECTION)
           .filterDate(f"{year}-01-01", f"{year+1}-01-01").filterBounds(g).mosaic())
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(f"{year}-01-01", f"{year+1}-01-01")
          .map(lambda i: i.updateMask(i.select("SCL").neq(3).And(i.select("SCL").neq(8))
                                      .And(i.select("SCL").neq(9)).And(i.select("SCL").neq(10)))))
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    bands = []
    for q, m0, m1 in QUARTERS:
        d0, d1 = f"{year}-{m0:02d}-01", (f"{year}-{m1:02d}-01" if m1 <= 12 else f"{year+1}-01-01")
        s2q = s2.filterDate(d0, d1).median()
        bands.append(s2q.normalizedDifference(["B8", "B4"]).rename(f"ndvi_{q}"))
        s1q = s1.filterDate(d0, d1).median()
        bands.append(s1q.select("VV").rename(f"vv_{q}"))
        bands.append(s1q.select("VH").rename(f"vh_{q}"))
    elev = ee.Image("USGS/SRTMGL1_003").select("elevation")
    wc = ee.ImageCollection(C.WORLDCOVER).first().select("Map").rename("label")
    stacked = (emb.addBands(ee.Image.cat(bands)).addBands(elev)
               .addBands(ee.Terrain.slope(elev).rename("slope"))
               .addBands(agro_zone(ee)).addBands(wc))
    return g, stacked


def _fc_df(ee, fc):
    def add_ll(f):
        c = f.geometry().coordinates(); return f.set({"lon": c.get(0), "lat": c.get(1)})
    feats = fc.map(add_ll).getInfo()["features"]
    return pd.DataFrame.from_records(
        [{k: ft["properties"].get(k) for k in T2_COLS} for ft in feats])


def export_t2_pool(project=C.EE_PROJECT, n=32000, out_csv=None, year=2021):
    ee = _init(project)
    g, stacked = _t2_stack(ee, year)
    frames, done, i = [], 0, 0
    while done < n:                                   # random pool over all India
        npts = min(4500, n - done)
        print(f"[t2] random batch {i+1}: {npts} (seed {C.SEED+i})")
        for _try in range(3):
            try:
                pts = ee.FeatureCollection.randomPoints(g, npts, C.SEED + i)
                s = stacked.sampleRegions(pts, scale=C.SCALE, geometries=True, tileScale=8) \
                    .filter(ee.Filter.notNull(["label", "A00", "ndvi_q3", "zone"]))
                frames.append(_fc_df(ee, s)); break
            except Exception as e:
                print("   retry:", str(e).splitlines()[0][:60]); import time; time.sleep(6)
        done += npts; i += 1
    for cls, boxes in C.RARE_BOXES.items():           # boost rare WorldCover classes
        for bb in boxes:
            try:
                pts = ee.FeatureCollection.randomPoints(ee.Geometry.Rectangle(bb), 150, C.SEED)
                s = stacked.sampleRegions(pts, scale=C.SCALE, geometries=True, tileScale=8) \
                    .filter(ee.Filter.eq("label", cls)).filter(ee.Filter.notNull(["A00", "ndvi_q3", "zone"]))
                frames.append(_fc_df(ee, s))
            except Exception:
                continue
        print(f"[t2] boosted class {cls}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["lon", "lat"]).reset_index(drop=True)
    out_csv = out_csv or POOL_CSV
    df.to_csv(out_csv, index=False)
    print(f"[t2] pool: {len(df):,} rows -> {out_csv}")
    return out_csv


def build_t2(pool_csv=None, per_cell=90, version="1.0.0"):
    """Stratify pool by zone x class, assign 70/15/15 geographically-disjoint splits."""
    import hashlib, json
    df = pd.read_csv(pool_csv or POOL_CSV)
    df["label"] = df["label"].astype(int).map({10:0,20:1,30:2,40:3,50:4,60:5,70:6,80:7,90:8,95:9,100:10})
    df = df.dropna(subset=["label", "zone"]).reset_index(drop=True); df["label"] = df["label"].astype(int)
    df["zone"] = df["zone"].astype(int)
    # stratified draw: <= per_cell per (zone x class)
    keep = pd.concat([g.sample(min(len(g), per_cell), random_state=C.SEED)
                      for _, g in df.groupby(["zone", "label"])], ignore_index=True)
    # geographically-disjoint 70/15/15 by 1-deg block
    bx = np.floor(keep.lon).astype(int); by = np.floor(keep.lat).astype(int)
    blocks = (bx.astype(str) + "_" + by.astype(str)); u = blocks.unique().tolist()
    rng = np.random.default_rng(C.SEED); rng.shuffle(u)
    n = len(u); te = set(u[:int(n*.15)]); va = set(u[int(n*.15):int(n*.30)])
    keep = keep.copy()
    keep["split"] = ["test" if b in te else "val" if b in va else "train" for b in blocks]
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / f"t2_landcover_ts_v{version}.parquet"
    keep.to_parquet(out, index=False)
    man = {"version": version, "task": "T2_landcover_timeseries", "n": len(keep),
           "splits": keep.split.value_counts().to_dict(),
           "by_zone": keep.zone.value_counts().sort_index().to_dict(),
           "by_class": keep.label.value_counts().sort_index().to_dict(),
           "features": {"embedding": 64, "timeseries": TS_COLS, "static": ["elevation","slope","zone"]},
           "sha256": hashlib.sha256(out.read_bytes()).hexdigest()[:16]}
    (BENCH_DIR / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"[t2] benchmark: {len(keep):,} pts | splits {man['splits']} -> {out}")
    print(f"[t2] zones covered: {sum(1 for v in man['by_zone'].values() if v>0)}/15 | classes: {len(man['by_class'])}/11")
    return out


# ---- T3: real CGWB well ground-truth (India Data Portal CKAN datastore) -----
CKAN = "https://ckandev.indiadataportal.com/api/3/action/datastore_search"
CGWB_RID = "580a8f6e-3d86-4ca7-ac7d-cd5df12b443c"       # CGWB depth-to-water-level
NW_STATES = ["Punjab", "Haryana", "Rajasthan", "Uttar Pradesh", "Gujarat",
             "Nct Of Delhi", "Madhya Pradesh"]
WELLS_CSV = C.DATA_RAW / "cgwb_wells_nw.csv"


def pull_cgwb(states=NW_STATES, date_from="2005-01-01", max_per_state=12000, out_csv=None):
    import json
    import requests
    rows = []
    for st in states:
        got = 0
        while got < max_per_state:
            try:
                r = requests.get(CKAN, timeout=60, params={
                    "resource_id": CGWB_RID, "limit": 2000, "offset": got,
                    "filters": json.dumps({"state_name": st})}).json()
            except Exception as e:
                print("  retry", st, str(e)[:50]); continue
            recs = r.get("result", {}).get("records", [])
            if not recs:
                break
            rows += recs; got += len(recs)
            if len(recs) < 2000:
                break
        print(f"[cgwb] {st}: {got}")
    df = pd.DataFrame(rows)
    for c in ("latitude", "longitude", "currentlevel"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "currentlevel", "date"])
    out_csv = out_csv or WELLS_CSV
    df.to_csv(out_csv, index=False)
    print(f"[cgwb] {len(df):,} obs from {df.station_name.nunique()} wells -> {out_csv}")
    return out_csv


def validate_gw_groundtruth(wells_csv=None, grace_csv=None, min_obs=8):
    """Per-well water-table trend (m/yr; +ve = deepening = depletion) vs GRACE
    downscaled trend at the well. Real cross-sensor ground-truth validation."""
    from scipy.stats import pearsonr
    wells = pd.read_csv(wells_csv or WELLS_CSV, parse_dates=["date"])
    tr = []
    for stn, g in wells.groupby("station_name"):
        g = g.sort_values("date")
        if len(g) < min_obs:
            continue
        yr = (g.date - g.date.min()).dt.days / 365.25
        if yr.max() < 4:
            continue
        slope = np.polyfit(yr, g.currentlevel, 1)[0]        # m/yr depth change
        tr.append({"station": stn, "lat": g.latitude.iloc[0], "lon": g.longitude.iloc[0],
                   "well_trend_m_yr": slope})
    wt = pd.DataFrame(tr)
    grace = pd.read_csv(grace_csv or (C.DATA_RAW / "gw_validation_trends.csv"))
    gcol = "grace_03_16" if "grace_03_16" in grace.columns else "gw_trend"
    gl = grace[["lon", "lat", gcol]].to_numpy()
    def nearest(lon, lat):
        d = (gl[:, 0] - lon) ** 2 + (gl[:, 1] - lat) ** 2
        return gl[d.argmin(), 2]
    wt["grace_trend"] = [nearest(r.lon, r.lat) for r in wt.itertuples()]
    wt = wt[np.isfinite(wt.grace_trend) & np.isfinite(wt.well_trend_m_yr)]
    r, p = pearsonr(wt.well_trend_m_yr, wt.grace_trend)
    frac_dep = (wt.well_trend_m_yr > 0).mean()
    print(f"\n=== T3 GROUND-TRUTH VALIDATION (CGWB wells vs GRACE) ===")
    print(f"  wells with trend: {len(wt)} | mean well trend {wt.well_trend_m_yr.mean():+.3f} m/yr "
          f"({100*frac_dep:.0f}% deepening/depleting)")
    print(f"  Pearson r(well depth-trend, GRACE storage-trend) = {r:.3f} (p={p:.1e})")
    print(f"  (expect NEGATIVE: wells deepening <-> GRACE storage falling = agreement)")
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    wt.to_csv(BENCH_DIR / "t3_cgwb_grace_validation.csv", index=False)
    return r, len(wt)


# ---- T2 FIELD-TRUTH anchor: WorldCover vs real WorldCereal crop points -------
WCEREAL = [f"https://zenodo.org/api/records/7825628/files/{f}/content"
           for f in ("sv_worldcereal_validation.csv", "sv_croptype_validations.csv")]
NONCROP = {"Other", "Not cropland", "No cropland", "Unknown", "Grassland", "Trees",
           "Shrubland", "Bare", "Built up", "Water", "Barren"}


def lc_fieldtruth(project=C.EE_PROJECT):
    """Real ground truth: ESA WorldCover label vs expert-validated crop points
    (WorldCereal validation campaign, India subset). Field truth, not a model."""
    import io
    import requests
    ref = pd.concat([pd.read_csv(io.BytesIO(requests.get(u, timeout=60).content))
                     for u in WCEREAL], ignore_index=True)
    ref = ref[(ref.x.between(68, 97.5)) & (ref.y.between(8, 37))]
    ref = ref.drop_duplicates(subset=["x", "y"]).reset_index(drop=True)
    ee = _init(project)
    wc = ee.ImageCollection(C.WORLDCOVER).first().select("Map")
    feats = [ee.Feature(ee.Geometry.Point([float(r.x), float(r.y)]), {"i": int(i)})
             for i, r in ref.iterrows()]
    samp = wc.sampleRegions(collection=ee.FeatureCollection(feats), scale=10,
                            tileScale=4).getInfo()["features"]
    ref["wc"] = [({f["properties"]["i"]: f["properties"].get("Map")
                   for f in samp}).get(i) for i in range(len(ref))]
    ref = ref.dropna(subset=["wc"]); ref["wc"] = ref["wc"].astype(int)
    crop = ref[~ref.cropType.isin(NONCROP)]
    acc = float((crop.wc == 40).mean())
    print(f"\n=== T2 FIELD-TRUTH anchor (WorldCover vs real crop-validation points) ===")
    print(f"  {len(crop)} expert-validated India crop points | WorldCover cropland agreement: {100*acc:.1f}%")
    ref.to_csv(BENCH_DIR / "t2_fieldtruth_india.csv", index=False)
    return acc


# ---- LC label-reliability anchor: WorldCover vs Dynamic World ---------------
# harmonise to common legend: 0 water,1 tree,2 shrub/grass,3 crop,4 built,5 bare,6 snow
WC2COMMON = {0:1,1:2,2:2,3:3,4:4,5:5,6:6,7:0,8:0,9:1,10:5}
DW2COMMON = {0:0,1:1,2:2,3:0,4:3,5:2,6:4,7:5,8:6}
COMMON = {0:"water",1:"tree",2:"shrub/grass",3:"crop",4:"built",5:"bare",6:"snow"}


def lc_anchor(project=C.EE_PROJECT):
    ee = _init(project)
    df = pd.read_parquet(BENCH_DIR / "t2_landcover_ts_v1.0.0.parquet")
    sub = df[df.split.isin(["test", "val"])][["lon", "lat", "label"]].reset_index(drop=True)
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterDate("2021-01-01", "2022-01-01")
          .select("label").mode().rename("dw"))
    feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]), {"idx": int(i)})
             for i, r in sub.iterrows()]
    samp = dw.sampleRegions(collection=ee.FeatureCollection(feats), scale=10,
                            tileScale=4).getInfo()["features"]
    dwv = {f["properties"]["idx"]: f["properties"].get("dw") for f in samp}
    sub["dw"] = [dwv.get(i) for i in range(len(sub))]
    sub = sub.dropna(subset=["dw"])
    wc_c = sub["label"].astype(int).map(WC2COMMON)
    dw_c = sub["dw"].astype(int).map(DW2COMMON)
    agree = (wc_c == dw_c)
    print(f"\n=== T2 LC label-reliability anchor (WorldCover vs Dynamic World) ===")
    print(f"  {len(sub)} points | overall agreement: {100*agree.mean():.1f}%")
    print("  per common-class agreement:")
    for c in range(7):
        m = wc_c == c
        if m.sum() >= 10:
            print(f"    {COMMON[c]:<12} {100*(agree & m).sum()/m.sum():5.1f}%  (n={m.sum()})")
    sub.to_csv(BENCH_DIR / "t2_lc_anchor_agreement.csv", index=False)
    return float(agree.mean())


def compare_t2(parquet=None, n_boot=500):
    """Representation head-to-head on T2 (laptop): AlphaEarth FM embedding vs
    hand-crafted seasonal time-series vs combined, same fixed test + bootstrap CIs."""
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, f1_score
    df = pd.read_parquet(parquet or (BENCH_DIR / "t2_landcover_ts_v1.0.0.parquet"))
    allcols = C.EMB_COLS + TS_COLS + ["elevation", "slope"]
    df[allcols] = df[allcols].fillna(df[allcols].median())
    y = df["label"].to_numpy(int)
    tr = (df.split == "train").to_numpy(); te = (df.split == "test").to_numpy()
    sets = {"AlphaEarth embedding (global FM)": C.EMB_COLS,
            "Seasonal time-series (hand-crafted)": TS_COLS + ["elevation", "slope"],
            "Embedding + time-series (all)": C.EMB_COLS + TS_COLS + ["elevation", "slope"]}
    print("\n=== Bharat-EO-Bench v1 · T2 representation head-to-head (test n=%d) ===" % te.sum())
    print(f"  {'representation':<38} {'OA [95% CI]':<22} {'macro-F1 [95% CI]'}")
    rng = np.random.default_rng(C.SEED)
    for name, cols in sets.items():
        X = df[cols].to_numpy("float32")
        m = XGBClassifier(n_estimators=350, max_depth=8, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=C.SEED)
        m.fit(X[tr], y[tr]); p = m.predict(X[te]); yt = y[te]
        oa, f1 = [], []
        for _ in range(n_boot):
            b = rng.integers(0, len(yt), len(yt))
            oa.append(accuracy_score(yt[b], p[b])); f1.append(f1_score(yt[b], p[b], average="macro", zero_division=0))
        oa, f1 = np.array(oa), np.array(f1)
        print(f"  {name:<38} {oa.mean():.3f} [{np.percentile(oa,2.5):.3f},{np.percentile(oa,97.5):.3f}]  "
              f"{f1.mean():.3f} [{np.percentile(f1,2.5):.3f},{np.percentile(f1,97.5):.3f}]")


# ---- variance-reporting evaluation harness ----------------------------------
def _blocks(df):
    return (np.floor(df.lon).astype(int).astype(str) + "_" +
            np.floor(df.lat).astype(int).astype(str)).to_numpy()


def evaluate_t2(parquet=None, n_boot=1000, k_folds=5):
    """Rigorous T2 eval: bootstrap CIs on the fixed test set + k spatial folds."""
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, f1_score
    p = parquet or (BENCH_DIR / "t2_landcover_ts_v1.0.0.parquet")
    df = pd.read_parquet(p)
    cols = C.EMB_COLS + TS_COLS + ["elevation", "slope"]
    df[cols] = df[cols].fillna(df[cols].median())
    X = df[cols].to_numpy("float32"); y = df["label"].to_numpy(int)

    def fit_pred(tr_mask, te_mask):
        m = XGBClassifier(n_estimators=350, max_depth=8, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                          n_jobs=-1, random_state=C.SEED)
        m.fit(X[tr_mask], y[tr_mask]); return m.predict(X[te_mask]), y[te_mask]

    # (1) fixed test set + bootstrap CIs
    tr = (df.split == "train").to_numpy(); te = (df.split == "test").to_numpy()
    pred, yte = fit_pred(tr, te)
    rng = np.random.default_rng(C.SEED); oas, f1s = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yte), len(yte))
        oas.append(accuracy_score(yte[idx], pred[idx]))
        f1s.append(f1_score(yte[idx], pred[idx], average="macro", zero_division=0))
    oas, f1s = np.array(oas), np.array(f1s)

    # (2) k spatial folds (rotate held-out blocks)
    blk = _blocks(df); u = pd.unique(blk); rng.shuffle(u)
    fold_of = {b: i % k_folds for i, b in enumerate(u)}
    fold = np.array([fold_of[b] for b in blk])
    foa, ff1 = [], []
    for kf in range(k_folds):
        tem = fold == kf; trm = ~tem
        if tem.sum() == 0 or trm.sum() == 0:
            continue
        pr, yt = fit_pred(trm, tem)
        foa.append(accuracy_score(yt, pr)); ff1.append(f1_score(yt, pr, average="macro", zero_division=0))

    print("\n=== Bharat-EO-Bench v1 · T2 · AlphaEarth+seasonal-TS + XGBoost ===")
    print(f"  fixed test (n={te.sum()}), 1000x bootstrap:")
    print(f"     OA       {oas.mean():.3f}  95% CI [{np.percentile(oas,2.5):.3f}, {np.percentile(oas,97.5):.3f}]")
    print(f"     macro-F1 {f1s.mean():.3f}  95% CI [{np.percentile(f1s,2.5):.3f}, {np.percentile(f1s,97.5):.3f}]")
    print(f"  {k_folds}-fold spatial CV (generalisation across regions):")
    print(f"     OA       {np.mean(foa):.3f} +/- {np.std(foa):.3f}")
    print(f"     macro-F1 {np.mean(ff1):.3f} +/- {np.std(ff1):.3f}")
    res = {"test_oa_mean": float(oas.mean()), "test_oa_ci": [float(np.percentile(oas,2.5)), float(np.percentile(oas,97.5))],
           "test_macrof1_mean": float(f1s.mean()), "test_macrof1_ci": [float(np.percentile(f1s,2.5)), float(np.percentile(f1s,97.5))],
           "spatial_oa": [float(np.mean(foa)), float(np.std(foa))],
           "spatial_macrof1": [float(np.mean(ff1)), float(np.std(ff1))]}
    import json
    (BENCH_DIR / "t2_baseline_variance.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["verify-zones", "export-t2", "build-t2",
                                    "pull-cgwb", "validate-gw", "eval-t2", "lc-anchor", "lc-fieldtruth", "compare-t2"])
    ap.add_argument("--project", default=C.EE_PROJECT)
    ap.add_argument("--n", type=int, default=32000)
    a = ap.parse_args()
    if a.cmd == "verify-zones": verify_zones(a.project)
    elif a.cmd == "export-t2": export_t2_pool(a.project, a.n)
    elif a.cmd == "build-t2": build_t2()
    elif a.cmd == "pull-cgwb": pull_cgwb()
    elif a.cmd == "validate-gw": validate_gw_groundtruth()
    elif a.cmd == "eval-t2": evaluate_t2()
    elif a.cmd == "lc-anchor": lc_anchor()
    elif a.cmd == "lc-fieldtruth": lc_fieldtruth()
    elif a.cmd == "compare-t2": compare_t2()
