"""Bharat-EO-Bench **T1 v2** — the improvised, richer, bigger chip dataset.

Per chip (224x224, int16), 15 input bands:
  optical(9): B2 B3 B4 B5 B6 B7 B8A B11 B12   (adds red-edge B5/B6/B7)
  SAR(2):     VV VH                            (annual median, dB x100)
  SAR-var(2): VVstd VHstd                      (yearly std x100 -> crops/water dynamics)
  terrain(2): elevation(m)  slope(deg x10)
+ label:  ESA WorldCover remapped 0..10 (255=nodata)
+ conf:   1 where Dynamic World (class-mapped) AGREES with WorldCover, else 0
          (rare classes 9/10 forced 1 — DW can't represent them) -> multi-source
          label fusion; training can ignore low-confidence pixels.
+ zone:   agro-climatic zone (0..14) for stratified sampling & evaluation.

Sampling is ZONE-STRATIFIED (per-zone cap) + targeted rare-class boxes, over one or
more YEARS. Resumable (existing chips skipped). getInfo is chunked to <=8 bands/call
to stay under the sampleRectangle payload cap.
"""
from __future__ import annotations
import argparse, random, time
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C
from .groundwater import _init
from .export_chips import _fix, CHIP, CHIP_TARGET_BOXES

OPT9 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]
IMG_BANDS = OPT9 + ["VV", "VH", "elevation", "slope"]   # 13 (dropped SAR-std: reducer too slow)
N_ZONES = 15
OUT_DIR = C.ROOT / "data" / "chips_v2"
# Dynamic World label (0..8) -> our WorldCover-remap index (0..10)
DW_TO_IDX = [7, 0, 2, 8, 3, 1, 4, 5, 6]   # water,tree,grass,floodveg,crop,shrub,built,bare,snow


def agro_zone(ee):
    rain = ee.Image("WORLDCLIM/V1/BIO").select("bio12")
    elev = ee.Image("USGS/SRTMGL1_003").select("elevation")
    rcl = (ee.Image(0).where(rain.gte(500), 1).where(rain.gte(1000), 2)
           .where(rain.gte(1500), 3).where(rain.gte(2500), 4))
    ecl = ee.Image(0).where(elev.gte(300), 1).where(elev.gte(1000), 2)
    return rcl.multiply(3).add(ecl).rename("zone").toInt()


def _india(ee):
    return (ee.FeatureCollection(C.GAUL0).filter(ee.Filter.eq("ADM0_NAME", C.COUNTRY))
            .geometry().simplify(5000))


def _stack(ee, region, year):
    def mask(img):
        scl = img.select("SCL")
        keep = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(keep)
    # cloud-masked MOSAIC (first clear pixel) — much faster than median, still cloud-free
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(f"{year}-01-01", f"{year+1}-01-01").filterBounds(region)
          .map(mask).select(OPT9).mosaic().toInt16())
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filterDate(f"{year}-01-01", f"{year+1}-01-01").filterBounds(region).select(["VV", "VH"])
          .mosaic().multiply(100).toInt16().rename(["VV", "VH"]))
    srtm = ee.Image("USGS/SRTMGL1_003")
    elev = srtm.select("elevation").toInt16()
    slope = ee.Terrain.slope(srtm).multiply(10).toInt16().rename("slope")
    wc_col = "ESA/WorldCover/v100" if year <= 2020 else "ESA/WorldCover/v200"
    wc = ee.ImageCollection(wc_col).first().select("Map")
    wcr = wc.remap([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
                   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).rename("label")
    # multi-source label fusion: Dynamic World monsoon-window mosaic (cheap, no reducer)
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
          .filterDate(f"{year}-06-01", f"{year}-11-30").filterBounds(region).select("label").mosaic())
    dwm = dw.remap(list(range(9)), DW_TO_IDX)
    conf = wcr.eq(dwm).Or(wcr.gte(9)).unmask(0).rename("conf").toUint8()
    img = (s2.addBands(s1).addBands(elev).addBands(slope)
           .addBands(wcr).addBands(conf).addBands(agro_zone(ee)))
    return img.reproject(crs="EPSG:3857", scale=10)


_BACKOFF = (5, 12, 25, 45)          # exponential backoff -> rides through intermittent throttle


def _gi(stack, bands, region, default):
    """sampleRectangle().getInfo() with backoff retries; returns properties or None."""
    for t in range(len(_BACKOFF)):
        try:
            return stack.select(bands).sampleRectangle(region=region, defaultValue=default).getInfo()["properties"]
        except Exception as e:
            if not hasattr(_gi, "_warned"):
                _gi._warned = True
                print(f"[_gi first-exception] bands={bands[:3]}..: {repr(e)[:200]}", flush=True)
            time.sleep(_BACKOFF[t])
    return None


def _sample(ee, region, year, chip):
    """chunked getInfo calls -> (img[224,224,13] int16, label u8, conf u8, zone int).
    Returns None on failure."""
    stack = _stack(ee, region, year)
    props = {}
    for i in range(0, len(IMG_BANDS), 7):                      # image bands in <=7-band chunks
        r = _gi(stack, IMG_BANDS[i:i + 7], region, 0)
        if r is None:
            return None
        props.update(r)
    r = _gi(stack, ["label", "conf", "zone"], region, 255)
    if r is None:
        return None
    try:
        # _fix each band to chip x chip BEFORE stacking: separate sampleRectangle calls
        # can return off-by-one pixel dims, so stacking raw arrays would raise.
        img = np.stack([_fix(np.array(props[b], dtype=np.int16), chip) for b in IMG_BANDS], -1)
        lab = _fix(np.array(r["label"], dtype=np.uint8), chip)
        conf = _fix(np.array(r["conf"], dtype=np.uint8), chip)
        zone = _fix(np.array(r["zone"], dtype=np.int16), chip)
    except Exception as e:
        if not hasattr(_sample, "_warned"):
            _sample._warned = True; print(f"[_sample assembly] {repr(e)[:140]}", flush=True)
        return None
    lab[lab > 10] = 255                                        # remap defaultValue/oob -> ignore
    zv = zone[(zone >= 0) & (zone < N_ZONES)]                  # drop masked(255)/oob before bincount
    zc = int(np.bincount(zv.ravel(), minlength=N_ZONES).argmax()) if zv.size else -1
    return img, lab, conf, zc


def _split(df):
    bx = np.floor(df.lon).astype(int); by = np.floor(df.lat).astype(int)
    blocks = bx.astype(str) + "_" + by.astype(str)            # by LOCATION only (years share a split)
    u = blocks.unique().tolist(); rng = np.random.default_rng(C.SEED); rng.shuffle(u)
    val = set(u[:max(1, int(len(u) * 0.15))]); test = set(u[int(len(u) * 0.15):int(len(u) * 0.30)])
    df["split"] = np.where(blocks.isin(val), "val", np.where(blocks.isin(test), "test", "train"))
    return df


def export_v2(project=C.EE_PROJECT, per_zone=140, per_class=25, years=(2021,), chip=CHIP, out_dir=None):
    ee = _init(project); india = _india(ee)
    out = Path(out_dir or OUT_DIR); (out / "img").mkdir(parents=True, exist_ok=True)
    for sub in ("lbl", "conf"): (out / sub).mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in (out / "img").glob("*.npy")}
    rows = []; t0 = time.time()
    for year in years:
        zc_count = {z: 0 for z in range(N_ZONES)}; saved = 0; fails = 0; tried = 0
        print(f"[v2 {year}] starting (per_zone={per_zone}) ...", flush=True)
        # ---- zone-stratified random chips ----
        def cands():
            bi = 0
            while bi < (per_zone * N_ZONES // 50 + 40):
                try:
                    fc = ee.FeatureCollection.randomPoints(india, 4000, C.SEED + bi + year).getInfo()["features"]
                except Exception:
                    bi += 1; continue
                bi += 1
                for f in fc: yield f["geometry"]["coordinates"]
        for lon, lat in cands():
            if all(v >= per_zone for v in zc_count.values()): break
            name = f"y{year}_r_{saved:04d}"
            if name in existing: saved += 1; continue
            region = ee.Geometry.Point([lon, lat]).buffer(chip / 2 * 10).bounds()
            tried += 1
            got = _sample(ee, region, year, chip)
            if got is None:
                fails += 1
                if fails % 5 == 0: print(f"[v2 {year}] {fails} getInfo fails (throttling?) | {saved} saved", flush=True)
                continue
            img, lab, conf, zc = got
            if (lab != 255).mean() < 0.5: continue            # skip ocean / no-data
            if zc >= 0 and zc_count.get(zc, 0) >= per_zone: continue  # zone already full -> stratify
            np.save(out / "img" / f"{name}.npy", img); np.save(out / "lbl" / f"{name}.npy", lab)
            np.save(out / "conf" / f"{name}.npy", conf)
            rows.append({"chip": name, "lon": lon, "lat": lat, "year": year, "zone": zc, "target": -1})
            saved += 1
            if zc >= 0: zc_count[zc] = zc_count.get(zc, 0) + 1
            if saved <= 3 or saved % 10 == 0:
                el = time.time() - t0; nz = sum(v > 0 for v in zc_count.values())
                print(f"[v2 {year}] {saved} chips | {nz}/{N_ZONES} zones | {el/max(tried,1):.0f}s/try | {el/60:.1f}m", flush=True)
        # ---- targeted rare-class chips ----
        for cls, blist in CHIP_TARGET_BOXES.items():
            tgt = per_class.get(cls, 0) if isinstance(per_class, dict) else per_class
            if tgt <= 0: continue
            cc = []
            for bb in blist:
                for _t in range(3):
                    try:
                        fc = ee.FeatureCollection.randomPoints(ee.Geometry.Rectangle(bb), max(20, tgt), C.SEED).getInfo()["features"]
                        cc += [f["geometry"]["coordinates"] for f in fc]; break
                    except Exception:
                        time.sleep(5)
            random.Random(C.SEED + cls).shuffle(cc); got_n = len([e for e in existing if e.startswith(f"y{year}_t{cls}_")])
            for lon, lat in cc:
                if got_n >= tgt: break
                name = f"y{year}_t{cls}_{got_n:04d}"
                if name in existing: got_n += 1; continue
                region = ee.Geometry.Point([lon, lat]).buffer(chip / 2 * 10).bounds()
                got = _sample(ee, region, year, chip)
                if got is None or (got[1] == cls).mean() < 0.02: continue
                img, lab, conf, zc = got
                np.save(out / "img" / f"{name}.npy", img); np.save(out / "lbl" / f"{name}.npy", lab)
                np.save(out / "conf" / f"{name}.npy", conf)
                rows.append({"chip": name, "lon": lon, "lat": lat, "year": year, "zone": zc, "target": cls})
                got_n += 1
            print(f"[v2 {year}] class {cls}: {got_n}", flush=True)
    # ---- manifest (merge + spatial split by location) ----
    mp = out / "manifest.csv"
    df = pd.concat([pd.read_csv(mp), pd.DataFrame(rows)], ignore_index=True).drop_duplicates("chip") \
         if mp.exists() and rows else (pd.read_csv(mp) if mp.exists() else pd.DataFrame(rows))
    if len(df):
        df = _split(df); df["bands"] = "|".join(IMG_BANDS); df.to_csv(mp, index=False)
        print(f"[v2] DONE {len(df)} chips -> {out} | splits {df.split.value_counts().to_dict()}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=C.EE_PROJECT)
    ap.add_argument("--per-zone", type=int, default=140)
    ap.add_argument("--per-class", type=int, default=25)
    ap.add_argument("--years", default="2021")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    yrs = tuple(int(y) for y in a.years.split(","))
    if a.test:
        ee = _init(a.project); india = _india(ee)
        ok = 0
        for i in range(8):
            f = ee.FeatureCollection.randomPoints(india, 3, C.SEED + i).getInfo()["features"]
            lon, lat = f[0]["geometry"]["coordinates"]
            region = ee.Geometry.Point([lon, lat]).buffer(CHIP / 2 * 10).bounds()
            t = time.time(); got = _sample(ee, region, yrs[0], CHIP)
            if got is None: print(f"  chip {i}: FAILED"); continue
            img, lab, conf, zc = got
            print(f"  chip {i}: img {img.shape}{img.dtype} | lab valid {100*(lab!=255).mean():.0f}% "
                  f"| conf-agree {100*(conf==1).mean():.0f}% | zone {zc} | {time.time()-t:.0f}s")
            ok += 1
            if ok >= 3: break
    else:
        export_v2(a.project, a.per_zone, a.per_class, yrs)
