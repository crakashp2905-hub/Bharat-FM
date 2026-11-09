"""Multi-temporal (monsoon-phase) + SAR chip export for Prithvi multi-frame input
and the MPATE (monsoon-phase-aware temporal encoding) experiments.

Each chip = P monsoon phases x (6 S2 optical + 2 S1 SAR) bands + a WorldCover label:
  premon (Feb-May) | monsoon (Jun-Sep) | postmon (Oct-Jan of next year)
Saved as [224, 224, P*8] int16 .npy + uint8 label .npy. Cloud-masked MEDIAN per
phase (monsoon optical is cloudy, so Sentinel-1 SAR carries the signal). Sampling
is done ONE PHASE AT A TIME to stay under GEE's sampleRectangle payload limit, and
the export is RESUMABLE (existing chips are skipped).
"""
from __future__ import annotations
import argparse, random, time
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C
from .groundwater import _init
from .export_chips import S2_BANDS, S1_BANDS, WC_REMAP, CHIP, _fix, CHIP_TARGET_BOXES

# (name, start_MMDD, end_MMDD, wraps_to_next_year)
PHASES = [
    ("premon",  "02-01", "05-31", False),
    ("monsoon", "06-01", "09-30", False),
    ("postmon", "10-01", "12-31", False),
]
PBANDS = S2_BANDS + S1_BANDS                                  # 8 per phase
TCOLS = [f"{nm}_{b}" for nm, *_ in PHASES for b in PBANDS]    # 24-band layout, in order
OUT_DIR = C.ROOT / "data" / "chips_temporal"


def _india(ee):
    return (ee.FeatureCollection(C.GAUL0)
            .filter(ee.Filter.eq("ADM0_NAME", C.COUNTRY)).geometry().simplify(5000))


def _phase_image(ee, region, name, a, b, year):
    """Cloud-masked S2 median (6 band) + S1 median (2 band) for one monsoon phase."""
    def mask(img):
        scl = img.select("SCL")
        keep = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return img.updateMask(keep)
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(f"{year}-{a}", f"{year}-{b}").filterBounds(region)
          .map(mask).select(S2_BANDS).median().toInt16())
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filterDate(f"{year}-{a}", f"{year}-{b}").filterBounds(region)
          .select(S1_BANDS).median().multiply(100).toInt16())
    img = s2.addBands(s1).rename([f"{name}_{x}" for x in PBANDS])
    return img.reproject(crs="EPSG:3857", scale=10)


def _sample_chip(ee, wc, region, chip, year):
    """One temporal chip: sample each phase separately, then the label. Returns
    (img[224,224,24] int16, lab[224,224] uint8) or None on failure/ocean."""
    phase_arrs = []
    for nm, a, b, _ in PHASES:
        r = None
        for _t in range(2):                                  # retry transient blips
            try:
                pi = _phase_image(ee, region, nm, a, b, year)
                r = pi.sampleRectangle(region=region, defaultValue=0).getInfo()["properties"]
                break
            except Exception:
                time.sleep(4)
        if r is None:
            return None
        try:
            arr = np.stack([np.array(r[f"{nm}_{x}"], dtype=np.int16) for x in PBANDS], axis=-1)
        except Exception:
            return None
        phase_arrs.append(_fix(arr, chip))
    try:
        lr = wc.sampleRectangle(region=region, defaultValue=0).getInfo()["properties"]
        lab = _fix(np.array(lr["label"], dtype=np.int32), chip)
    except Exception:
        return None
    img = np.concatenate(phase_arrs, axis=-1)                 # [224,224,24]
    out_lab = np.full(lab.shape, 255, dtype=np.uint8)
    for k, v in WC_REMAP.items():
        out_lab[lab == k] = v
    return img, out_lab


def _write(out, tag, saved, lon, lat, img, lab, rows, target=-1):
    np.save(out / "img" / f"{tag}_{saved:04d}.npy", img)
    np.save(out / "lbl" / f"{tag}_{saved:04d}.npy", lab)
    rows.append({"chip": f"{tag}_{saved:04d}", "lon": lon, "lat": lat, "target": target})


def export_temporal(project=C.EE_PROJECT, n=400, per_class=15, chip=CHIP,
                    year=2021, out_dir=None):
    """n random chips across India + per_class targeted chips in each rare-class box.
    Resumable: skips chips whose .npy already exists; merges into manifest.csv."""
    ee = _init(project)
    india = _india(ee)
    wc = ee.ImageCollection(C.WORLDCOVER).first().select("Map").rename("label")
    out = Path(out_dir or OUT_DIR)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "lbl").mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in (out / "img").glob("*.npy")}
    rows = []
    t0 = time.time()

    # ---- random chips across India -----------------------------------------
    def _cands():
        bi = 0
        while bi < (n // 60 + 30):
            try:
                fc = ee.FeatureCollection.randomPoints(india, 4000, C.SEED + bi).getInfo()["features"]
            except Exception:
                bi += 1; continue
            bi += 1
            for f in fc:
                yield f["geometry"]["coordinates"]
    saved = len([e for e in existing if e.startswith("r_")])
    for lon, lat in _cands():
        if saved >= n:
            break
        tag = "r"
        if f"{tag}_{saved:04d}" in existing:
            saved += 1; continue
        region = ee.Geometry.Point([lon, lat]).buffer(chip / 2 * 10).bounds()
        got = _sample_chip(ee, wc, region, chip, year)
        if got is None or (got[1] != 255).mean() < 0.5:      # skip ocean / no-data
            continue
        _write(out, tag, saved, lon, lat, got[0], got[1], rows)
        saved += 1
        if saved % 10 == 0:
            el = time.time() - t0
            print(f"[temporal] random {saved}/{n}  ({el/max(saved,1):.0f}s/chip, {el/60:.1f}m elapsed)", flush=True)

    # ---- targeted rare-class chips -----------------------------------------
    for cls, blist in CHIP_TARGET_BOXES.items():
        tgt = per_class.get(cls, 0) if isinstance(per_class, dict) else per_class
        if tgt <= 0:
            continue
        cands = []
        for bb in blist:
            for _t in range(3):
                try:
                    fc = ee.FeatureCollection.randomPoints(
                        ee.Geometry.Rectangle(bb), max(20, tgt), C.SEED).getInfo()["features"]
                    cands += [f["geometry"]["coordinates"] for f in fc]; break
                except Exception:
                    time.sleep(5)
        random.Random(C.SEED + cls).shuffle(cands)
        base = len([e for e in existing if e.startswith(f"t{cls}_")])
        got_n = base
        for lon, lat in cands:
            if got_n >= tgt:
                break
            tag = f"t{cls}"
            if f"{tag}_{got_n:04d}" in existing:
                got_n += 1; continue
            region = ee.Geometry.Point([lon, lat]).buffer(chip / 2 * 10).bounds()
            got = _sample_chip(ee, wc, region, chip, year)
            if got is None or (got[1] == cls).mean() < 0.02:
                continue
            _write(out, tag, got_n, lon, lat, got[0], got[1], rows, target=cls)
            got_n += 1
        print(f"[temporal] class {cls}: {got_n} total", flush=True)

    # ---- manifest (merge with any prior) + spatial split -------------------
    df_new = pd.DataFrame(rows)
    mpath = out / "manifest.csv"
    if mpath.exists() and len(df_new):
        df = pd.concat([pd.read_csv(mpath), df_new], ignore_index=True).drop_duplicates("chip")
    elif mpath.exists():
        df = pd.read_csv(mpath)
    else:
        df = df_new
    if len(df):
        bx = np.floor(df.lon / 1.0).astype(int); by = np.floor(df.lat / 1.0).astype(int)
        blocks = bx.astype(str) + "_" + by.astype(str)
        u = blocks.unique().tolist(); rng = np.random.default_rng(C.SEED); rng.shuffle(u)
        val = set(u[:max(1, int(len(u) * 0.2))])
        df["split"] = np.where(blocks.isin(val), "val", "train")
        df["layout"] = "|".join(TCOLS)
        df.to_csv(mpath, index=False)
    print(f"[temporal] DONE: {len(df)} chips -> {out} "
          f"({(df.split=='train').sum()} train / {(df.split=='val').sum()} val)" if len(df) else "[temporal] no chips", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=C.EE_PROJECT)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--per-class", type=int, default=15)
    ap.add_argument("--test", action="store_true", help="sample 3 chips and print shapes, then exit")
    a = ap.parse_args()
    if a.test:
        ee = _init(a.project); india = _india(ee)
        wc = ee.ImageCollection(C.WORLDCOVER).first().select("Map").rename("label")
        ok = 0
        for i in range(6):
            f = ee.FeatureCollection.randomPoints(india, 3, C.SEED + i).getInfo()["features"]
            lon, lat = f[0]["geometry"]["coordinates"]
            region = ee.Geometry.Point([lon, lat]).buffer(CHIP / 2 * 10).bounds()
            t = time.time(); got = _sample_chip(ee, wc, region, CHIP, 2021)
            if got is None:
                print(f"  chip {i}: FAILED"); continue
            print(f"  chip {i}: img {got[0].shape} {got[0].dtype} | lab {got[1].shape} "
                  f"| valid {100*(got[1]!=255).mean():.0f}% | {time.time()-t:.0f}s")
            ok += 1
            if ok >= 3:
                break
    else:
        export_temporal(a.project, a.n, a.per_class)
