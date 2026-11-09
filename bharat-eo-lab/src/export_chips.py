"""Export Sentinel-2 image chips + ESA WorldCover masks for fine-tuning a real
EO foundation model (Prithvi-EO-2.0) on a GPU.

Each chip = a 224x224, 6-band Sentinel-2 patch (B,G,R,NIR,SWIR1,SWIR2 — the
Prithvi/HLS band order) saved as int16 .npy, plus a uint8 WorldCover label mask
(remapped to 0..10; 255 = no-data). A manifest with a spatial-block train/val
split is written. Zip data/chips/ and upload it as a Kaggle Dataset.
"""
from __future__ import annotations
import argparse
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C
from .groundwater import _init

S2_BANDS = ["B2", "B3", "B4", "B8A", "B11", "B12"]   # B,G,R,NIR,SWIR1,SWIR2
S1_BANDS = ["VV", "VH"]                              # Sentinel-1 SAR (dB x100)
ALL_BANDS = S2_BANDS + S1_BANDS                      # 8-band chips
WC_REMAP = {10: 0, 20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 70: 6, 80: 7, 90: 8, 95: 9, 100: 10}
CHIP = 224
CHIPS_DIR = C.ROOT / "data" / "chips"


def _setup(ee, year=2021):
    india = (ee.FeatureCollection(C.GAUL0)
             .filter(ee.Filter.eq("ADM0_NAME", C.COUNTRY)).geometry().simplify(5000))
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(f"{year}-01-01", f"{year+1}-01-01"))   # NOT filtered to India
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filterDate(f"{year}-01-01", f"{year+1}-01-01").select(S1_BANDS))
    wc = ee.ImageCollection(C.WORLDCOVER).first().select("Map").rename("label")
    return india, s2, s1, wc


def _chip_stack(ee, s2, s1, wc, region):
    """Composite from ONLY the few scenes over this chip -> fast. 6 S2 + 2 S1 + label."""
    def mask(img):
        scl = img.select("SCL")
        keep = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(keep)
    s2c = s2.filterBounds(region).map(mask).select(S2_BANDS).mosaic().toInt16()
    s1c = s1.filterBounds(region).mosaic().select(S1_BANDS).multiply(100).toInt16()  # dB x100
    return s2c.addBands(s1c).addBands(wc).reproject(crs="EPSG:3857", scale=10)


def _fix(a, chip):
    """Centre-crop / pad an array to chip x chip (keeps trailing channels)."""
    H, W = a.shape[0], a.shape[1]
    if H < chip or W < chip:
        pad = [(0, max(0, chip - H)), (0, max(0, chip - W))]
        if a.ndim == 3:
            pad += [(0, 0)]
        a = np.pad(a, pad, constant_values=0)
        H, W = a.shape[0], a.shape[1]
    y0, x0 = (H - chip) // 2, (W - chip) // 2
    return a[y0:y0 + chip, x0:x0 + chip] if a.ndim == 2 else a[y0:y0 + chip, x0:x0 + chip, :]


def export_chips(project=C.EE_PROJECT, n=200, chip=CHIP, year=2021, out_dir=None):
    ee = _init(project)
    india, s2, s1, wc = _setup(ee, year)
    out = Path(out_dir or CHIPS_DIR)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "lbl").mkdir(parents=True, exist_ok=True)

    def _candidates():                       # batched to stay under 5000 getInfo cap
        bi = 0
        while bi < (n // 60 + 40):
            try:
                fc = ee.FeatureCollection.randomPoints(india, 4000, C.SEED + bi).getInfo()["features"]
            except Exception:
                bi += 1; continue
            bi += 1
            for f in fc:
                yield f
    rows, saved = [], 0
    for f in _candidates():
        if saved >= n:
            break
        lon, lat = f["geometry"]["coordinates"]
        region = ee.Geometry.Point([lon, lat]).buffer(chip / 2 * 10).bounds()
        try:
            stack = _chip_stack(ee, s2, s1, wc, region)
            r = stack.sampleRectangle(region=region, defaultValue=0).getInfo()["properties"]
            img = np.stack([np.array(r[b], dtype=np.int16) for b in ALL_BANDS], axis=-1)
            lab = np.array(r["label"], dtype=np.int32)
        except Exception:
            continue
        img, lab = _fix(img, chip), _fix(lab, chip)
        out_lab = np.full(lab.shape, 255, dtype=np.uint8)
        for k, v in WC_REMAP.items():
            out_lab[lab == k] = v
        if (out_lab != 255).mean() < 0.5:        # skip ocean / no-data chips
            continue
        np.save(out / "img" / f"chip_{saved:04d}.npy", img)
        np.save(out / "lbl" / f"chip_{saved:04d}.npy", out_lab)
        rows.append({"chip": f"chip_{saved:04d}", "lon": lon, "lat": lat})
        saved += 1
        if saved % 25 == 0:
            print(f"[chips] saved {saved}/{n}")

    df = pd.DataFrame(rows)
    bx = np.floor(df.lon / 1.0).astype(int); by = np.floor(df.lat / 1.0).astype(int)
    blocks = bx.astype(str) + "_" + by.astype(str)
    u = blocks.unique().tolist()
    rng = np.random.default_rng(C.SEED); rng.shuffle(u)
    val = set(u[:max(1, int(len(u) * 0.2))])
    df["split"] = np.where(blocks.isin(val), "val", "train")
    df["classes"] = ",".join(str(k) for k in WC_REMAP)
    df.to_csv(out / "manifest.csv", index=False)
    print(f"[chips] done: {saved} chips -> {out} "
          f"({(df.split=='train').sum()} train / {(df.split=='val').sum()} val)")
    return out


# ---- targeted regional sampling: chips centred where rare classes live -----
# keys = remapped class index (0..10); boxes = [W, S, E, N]
CHIP_TARGET_BOXES = {
    9: [  # mangrove
        [88.0, 21.5, 89.2, 22.6], [86.75, 20.55, 87.15, 20.95],
        [82.2, 16.2, 82.6, 16.95], [79.7, 11.35, 79.95, 11.55],
        [69.3, 22.2, 70.6, 23.0], [92.5, 11.4, 93.2, 13.6]],
    8: [  # herbaceous_wetland
        [85.0, 19.45, 85.75, 20.0], [68.5, 23.3, 71.8, 24.8],
        [93.7, 24.35, 93.95, 24.75], [81.05, 16.45, 81.45, 16.85],
        [91.5, 26.0, 94.0, 27.0]],
    6: [  # snow_ice (high Himalaya)
        [76.3, 33.9, 78.2, 35.6], [78.4, 30.7, 79.9, 31.7], [88.1, 27.75, 88.85, 28.25],
        [76.0, 32.5, 77.5, 33.6], [79.2, 30.2, 80.4, 31.0], [92.0, 27.8, 94.5, 28.6]],
    10: [  # moss_lichen (high Himalaya)
        [76.8, 33.4, 78.6, 34.8], [78.9, 30.45, 80.1, 31.4], [88.45, 27.65, 88.95, 28.2],
        [77.4, 31.9, 78.7, 32.7], [79.5, 30.0, 80.3, 30.7], [91.5, 27.6, 93.5, 28.4]],
    4: [  # built_up (major cities)
        [76.9, 28.4, 77.4, 28.8], [72.75, 18.9, 73.05, 19.25], [88.25, 22.45, 88.5, 22.7],
        [80.15, 12.9, 80.35, 13.2], [77.5, 12.85, 77.75, 13.1], [78.3, 17.3, 78.6, 17.55]],
    5: [  # bare_sparse (deserts)
        [70.5, 26.5, 72.5, 28.0], [77.3, 33.4, 78.6, 34.4], [70.0, 23.5, 71.5, 24.3]],
    7: [  # water (large reservoirs/lakes)
        [83.6, 21.45, 84.1, 21.8], [79.15, 16.45, 79.45, 16.75], [76.35, 31.25, 76.7, 31.55],
        [76.2, 22.1, 76.7, 22.6], [73.45, 21.15, 73.75, 21.45]],
}


def export_chips_targeted(project=C.EE_PROJECT, per_class=25, chip=CHIP, year=2021,
                          out_dir=None, min_frac=0.02):
    """Export chips centred in each weak class's known regions -> real coverage."""
    ee = _init(project)
    india, s2, s1, wc = _setup(ee, year)
    out = Path(out_dir or (C.ROOT / "data" / "chips_targeted"))
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "lbl").mkdir(parents=True, exist_ok=True)
    rows, saved = [], 0
    for cls, blist in CHIP_TARGET_BOXES.items():
        target_n = per_class.get(cls, 0) if isinstance(per_class, dict) else per_class
        if target_n <= 0:
            continue
        cands = []
        for bb in blist:
            for _t in range(3):                          # resilient to network blips
                try:
                    fc = ee.FeatureCollection.randomPoints(
                        ee.Geometry.Rectangle(bb), max(20, target_n), C.SEED).getInfo()["features"]
                    cands += [f["geometry"]["coordinates"] for f in fc]; break
                except Exception:
                    time.sleep(5)
        random.Random(C.SEED + cls).shuffle(cands)
        got = 0
        for lon, lat in cands:
            if got >= target_n:
                break
            region = ee.Geometry.Point([lon, lat]).buffer(chip / 2 * 10).bounds()
            r = None
            for _t in range(2):                          # retry transient failures
                try:
                    stack = _chip_stack(ee, s2, s1, wc, region)
                    r = stack.sampleRectangle(region=region, defaultValue=0).getInfo()["properties"]
                    break
                except Exception:
                    time.sleep(5)
            if r is None:
                continue
            try:
                img = np.stack([np.array(r[b], dtype=np.int16) for b in ALL_BANDS], axis=-1)
                lab = np.array(r["label"], dtype=np.int32)
            except Exception:
                continue
            img, lab = _fix(img, chip), _fix(lab, chip)
            out_lab = np.full(lab.shape, 255, dtype=np.uint8)
            for k, v in WC_REMAP.items():
                out_lab[lab == k] = v
            if (out_lab == cls).mean() < min_frac:        # require the target class present
                continue
            np.save(out / "img" / f"t{cls}_{saved:04d}.npy", img)
            np.save(out / "lbl" / f"t{cls}_{saved:04d}.npy", out_lab)
            rows.append({"chip": f"t{cls}_{saved:04d}", "lon": lon, "lat": lat, "target": cls})
            saved += 1; got += 1
        print(f"[target] class {cls}: +{got}")
    df = pd.DataFrame(rows)
    bx = np.floor(df.lon / 1.0).astype(int); by = np.floor(df.lat / 1.0).astype(int)
    blocks = bx.astype(str) + "_" + by.astype(str)
    u = blocks.unique().tolist(); rng = np.random.default_rng(C.SEED); rng.shuffle(u)
    val = set(u[:max(1, int(len(u) * 0.2))])
    df["split"] = np.where(blocks.isin(val), "val", "train")
    df.to_csv(out / "manifest.csv", index=False)
    print(f"[target] {saved} targeted chips -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=C.EE_PROJECT)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--targeted", action="store_true")
    ap.add_argument("--per-class", type=int, default=25)
    a = ap.parse_args()
    if a.targeted:
        export_chips_targeted(a.project, a.per_class)
    else:
        export_chips(a.project, a.n)
