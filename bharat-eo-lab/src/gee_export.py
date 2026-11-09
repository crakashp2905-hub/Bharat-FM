"""Earth Engine acquisition: India-scale sample of AlphaEarth embeddings +
ESA WorldCover labels. All heavy work runs on Google's servers; only a small
table comes back to the laptop.

Robust design (avoids the interactive 'Computation timed out' limit):
  1. randomPoints(region, N)        -> cheap; no full-region raster scan
  2. emb + WorldCover .sampleRegions(points, scale=10)  -> point lookups only

Two acquisition modes:
  * export_local  -> random points over all India (natural class mix), batched
                     under the 5000-element getInfo cap.
  * export_boost  -> targeted random points inside known regions of rare classes
                     (mangroves, wetland, moss, snow) -> fixes class imbalance.

Prereq (one-time): `earthengine authenticate` + set EE_PROJECT in config.py.
"""
from __future__ import annotations
import argparse
import pandas as pd

from . import config as C


def _init(project):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)
    return ee


def _layers(ee, year):
    """Build the AOI geometry and the stacked (embedding + label + state) image."""
    india = (ee.FeatureCollection(C.GAUL0)
             .filter(ee.Filter.eq("ADM0_NAME", C.COUNTRY))
             .geometry().simplify(maxError=5000))
    emb = (ee.ImageCollection(C.EMBED_COLLECTION)
           .filterDate(f"{year}-01-01", f"{year+1}-01-01")
           .filterBounds(india).mosaic())
    wc = ee.ImageCollection(C.WORLDCOVER).first().select("Map").rename("label")
    states = ee.FeatureCollection(C.GAUL1).filter(
        ee.Filter.eq("ADM0_NAME", C.COUNTRY))
    state_img = ee.Image().byte().paint(states, "ADM1_CODE").rename("state_code")
    return india, emb.addBands(wc).addBands(state_img)


def _add_lonlat(f):
    coords = f.geometry().coordinates()
    return f.set({"lon": coords.get(0), "lat": coords.get(1)})


def _fc_to_df(fc):
    cols = C.EMB_COLS + ["label", "state_code", "lon", "lat"]
    feats = fc.getInfo()["features"]
    return pd.DataFrame.from_records(
        [{c: ft["properties"].get(c) for c in cols} for ft in feats])


def build_sample(ee, year=C.YEAR, n_points=C.N_POINTS_LOCAL, seed=C.SEED):
    india, stacked = _layers(ee, year)
    pts = ee.FeatureCollection.randomPoints(region=india, points=n_points, seed=seed)
    sampled = stacked.sampleRegions(collection=pts, scale=C.SCALE,
                                    geometries=True, tileScale=8)
    sampled = sampled.filter(ee.Filter.notNull(["label", "A00"]))
    return sampled.map(_add_lonlat)


def export_local(project=C.EE_PROJECT, year=C.YEAR,
                 n_points=C.N_POINTS_LOCAL, out_csv=None, batch=4500):
    """Random points over all India, batched under the 5000 getInfo cap."""
    ee = _init(project)
    frames, done, i = [], 0, 0
    while done < n_points:
        npts = min(batch, n_points - done)
        print(f"[gee] batch {i+1}: requesting {npts} points (seed {C.SEED+i}) ...")
        frames.append(_fc_to_df(build_sample(ee, year, npts, seed=C.SEED + i)))
        done += npts
        i += 1
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["lon", "lat"]).reset_index(drop=True)
    df["year"] = year
    out_csv = out_csv or C.DEFAULT_CSV
    df.to_csv(out_csv, index=False)
    print(f"[gee] wrote {len(df):,} valid rows ({i} batches) -> {out_csv}")
    return out_csv


def export_boost(project=C.EE_PROJECT, year=C.YEAR, per_box=C.PER_BOX_POINTS,
                 boxes=None, out_csv=None):
    """Targeted sampling of rare classes inside their known regions.

    For each (class -> [boxes]) in config.RARE_BOXES, draw random points inside
    each box and keep only points whose WorldCover label equals that class.
    """
    ee = _init(project)
    boxes = boxes or C.RARE_BOXES
    _, stacked = _layers(ee, year)
    frames = []
    for code, blist in boxes.items():
        name = C.CLASS_NAMES.get(code, str(code))
        for j, bb in enumerate(blist):
            region = ee.Geometry.Rectangle(bb)
            pts = ee.FeatureCollection.randomPoints(region=region, points=per_box,
                                                    seed=C.SEED + j)
            s = stacked.sampleRegions(collection=pts, scale=C.SCALE,
                                      geometries=True, tileScale=8)
            s = s.filter(ee.Filter.eq("label", code)).filter(
                ee.Filter.notNull(["A00"])).map(_add_lonlat)
            d = _fc_to_df(s)
            frames.append(d)
            print(f"[boost] {name} ({code}) box {j+1}/{len(blist)}: +{len(d)} pts")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["lon", "lat"]).reset_index(drop=True)
    df["year"] = year
    out_csv = out_csv or C.BOOST_CSV
    df.to_csv(out_csv, index=False)
    print(f"[boost] wrote {len(df):,} rare-class rows -> {out_csv}")
    return out_csv


def export_drive(project=C.EE_PROJECT, year=C.YEAR, n_points=C.N_POINTS_DRIVE,
                 folder="bharat_eo_lab", description=None):
    """Launch a large asynchronous Export to your Google Drive."""
    ee = _init(project)
    fc = build_sample(ee, year, n_points)
    description = description or f"india_landcover_{year}"
    selectors = C.EMB_COLS + ["label", "state_code", "lon", "lat"]
    task = ee.batch.Export.table.toDrive(
        collection=fc, description=description, folder=folder,
        fileNamePrefix=description, fileFormat="CSV", selectors=selectors)
    task.start()
    print(f"[gee] started Drive export '{description}' ({n_points} pts). "
          f"Monitor at https://code.earthengine.google.com/tasks ; "
          f"then download the CSV into {C.DATA_RAW}")
    return task


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "boost", "drive"], default="local")
    ap.add_argument("--project", default=C.EE_PROJECT)
    ap.add_argument("--year", type=int, default=C.YEAR)
    ap.add_argument("--points", type=int, default=None)
    args = ap.parse_args()
    if args.mode == "local":
        export_local(args.project, args.year, args.points or C.N_POINTS_LOCAL)
    elif args.mode == "boost":
        export_boost(args.project, args.year)
    else:
        export_drive(args.project, args.year, args.points or C.N_POINTS_DRIVE)
