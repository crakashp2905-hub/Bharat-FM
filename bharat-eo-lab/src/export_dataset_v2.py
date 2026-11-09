"""Bharat-EO-Bench **Dataset v2** — the fully-improvised local chip dataset.

Everything the model-side v2 has on the data side, via the LOCAL getInfo path
(no Export-to-Drive). Over the 13-band chips_v2 exporter this adds:

INPUT bands (context; all cheap / static -> no slow reducers):
  optical(9) B2 B3 B4 B5 B6 B7 B8A B11 B12 | SAR(2) VV VH
  terrain(3) elevation slope aspect
  climate(2) rain(WorldClim bio12) temp(WorldClim bio1)
  night(1)   VIIRS annual nighttime lights
LABEL / SOFT-LABEL bands (multi-source fusion -> cleaner + soft targets):
  label      ESA WorldCover  (remap 0..10, 255=nodata)   [primary]
  label_esri ESRI/IO 10m LULC (remap to our 0..10)        [2nd source]
  label_dw   Dynamic World monsoon mode (remap)            [3rd source]
  dwp0..dwp8 Dynamic World mean class PROBABILITIES x255   [SOFT labels]
  wc_crop    ESA WorldCereal temporary-crops (0/1)         [crop-type refine, 2021]
  conf       # of sources agreeing w/ WorldCover (1..3; rare classes forced 3)
  zone       agro-climatic zone (0..14)
META (per-chip manifest): lon lat year zone dominant_class n_classes valid_frac
  conf_frac cloudmask_frac target split  -> enables sliced (per-zone/season) eval.

QC: reject ocean/nodata (<50% valid), near-duplicate locations, and cap the
dominant common classes (class-balanced sampling) so crop/tree don't swamp it.
Optional sources are PROBED at startup; any that error are dropped, not fatal.
Resumable. getInfo chunked <=7 bands/call. Backoff retries ride through throttle.

Deferred / opt-in (documented, not built here):
  * AlphaEarth-as-bands (--ae-dims K): the 64-d FM embedding per pixel. OFF by
    default -- redundant with a pretrained encoder and ~64 extra bands would make
    the local getInfo export ~3x longer. Enable with --ae-dims 16 to try a subset.
  * ISRO Bhuvan sovereign labels: not in the GEE catalog (needs Bhuvan WMS/portal
    ingest) -> future work; ESRI+DW+WorldCereal are the in-GEE multi-source stand-in.
"""
from __future__ import annotations
import argparse, atexit, os, random, time
from concurrent import futures
from pathlib import Path
import numpy as np
import pandas as pd

from . import config as C
from .groundwater import _init
from .export_chips import _fix, CHIP, CHIP_TARGET_BOXES
from .export_chips_v2 import agro_zone, _india, _gi, _split, N_ZONES, DW_TO_IDX

OPT9 = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]
CORE_IMG = OPT9 + ["VV", "VH", "elevation", "slope", "aspect", "rain", "temp", "ntl"]   # 17
DWP = [f"dwp{i}" for i in range(9)]                                                       # soft
DW_BANDS = ["water", "trees", "grass", "flooded_vegetation", "crops",
            "shrub_and_scrub", "built", "bare", "snow_and_ice"]
# ESRI/IO 10m classes -> our index (0..10); 3/6/10 (unused/clouds) -> 255
ESRI_FROM = [1, 2, 4, 5, 7, 8, 9, 11]
ESRI_TO   = [7, 0, 8, 3, 4, 5, 6, 2]     # water,tree,floodveg,crop,built,bare,snow,rangeland->grass
OUT_DIR = C.ROOT / "data" / "chips_dsv2"
AE_COLL = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


def _base_stack(ee, region, year, have):
    """Input + label + soft + zone image graph. `have` gates optional sources."""
    def mask(img):
        scl = img.select("SCL")
        return img.updateMask(scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterDate(f"{year}-01-01", f"{year+1}-01-01").filterBounds(region)
          .map(mask).select(OPT9).mosaic().toInt16())
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
          .filterDate(f"{year}-01-01", f"{year+1}-01-01").filterBounds(region)
          .select(["VV", "VH"]).mosaic().multiply(100).toInt16().rename(["VV", "VH"]))
    srtm = ee.Image("USGS/SRTMGL1_003")
    elev = srtm.select("elevation").toInt16()
    slope = ee.Terrain.slope(srtm).multiply(10).toInt16().rename("slope")
    aspect = ee.Terrain.aspect(srtm).toInt16().rename("aspect")
    wcbio = ee.Image("WORLDCLIM/V1/BIO")
    rain = wcbio.select("bio12").toInt16().rename("rain")                 # annual precip (mm)
    temp = wcbio.select("bio01").toInt16().rename("temp")                 # mean temp (x10 degC)
    yy = min(year, 2021)                                                  # V21 covers 2012..2021
    ntl_col = (ee.ImageCollection("NOAA/VIIRS/DNB/ANNUAL_V21")
               .filterDate(f"{yy}-01-01", f"{yy+1}-01-01").select("average"))
    ntl = (ee.Image(ee.Algorithms.If(ntl_col.size().gt(0), ntl_col.first(), ee.Image(0)))
           .multiply(100).toInt16().unmask(0).rename("ntl"))              # null-guarded
    img = s2.addBands(s1).addBands(elev).addBands(slope).addBands(aspect).addBands(rain).addBands(temp).addBands(ntl)

    wc_col = "ESA/WorldCover/v100" if year <= 2020 else "ESA/WorldCover/v200"
    wcr = (ee.ImageCollection(wc_col).first().select("Map")
           .remap([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100], list(range(11))).rename("label"))
    dw_ic = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
             .filterDate(f"{year}-06-01", f"{year}-11-30").filterBounds(region))
    dwm = dw_ic.select("label").mosaic().remap(list(range(9)), DW_TO_IDX).rename("label_dw")
    agree = wcr.eq(dwm).unmask(0)
    lab_img = wcr.addBands(dwm)

    if have.get("esri"):
        esri = (ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
                .filterDate(f"{min(year,2023)}-01-01", f"{min(year,2023)+1}-01-01").mosaic()
                .remap(ESRI_FROM, ESRI_TO, 255).rename("label_esri"))
        lab_img = lab_img.addBands(esri); agree = agree.add(wcr.eq(esri).unmask(0))
    if have.get("dwprobs"):
        dwp = dw_ic.select(DW_BANDS).mean().multiply(255).toUint8().rename(DWP)
        lab_img = lab_img.addBands(dwp)
    if have.get("worldcereal") and year == 2021:
        wcrop = (ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
                 .filter(ee.Filter.eq("product", "temporarycrops")).mosaic()
                 .select("classification").gt(0).unmask(0).toUint8().rename("wc_crop"))
        lab_img = lab_img.addBands(wcrop)
    conf = agree.add(1).min(3).where(wcr.gte(9), 3).rename("conf")             # 1..3; rare forced 3
    # cast the whole label/aux stack to Int16 so sampleRectangle's 255 nodata default
    # is valid for every band (remap yields int8, whose max is 127 -> 255 rejected).
    lab_img = lab_img.addBands(conf).addBands(agro_zone(ee)).toInt16()
    if have.get("ae") and have["ae"] > 0:
        ae = (ee.ImageCollection(AE_COLL).filterDate(f"{year}-01-01", f"{year+1}-01-01")
              .filterBounds(region).first().select([f"A{i:02d}" for i in range(have["ae"])])
              .multiply(1000).toInt16())
        img = img.addBands(ae)
    return img.addBands(lab_img).reproject(crs="EPSG:3857", scale=10)


def _bandlists(have):
    img = list(CORE_IMG)
    if have.get("ae"): img += [f"A{i:02d}" for i in range(have["ae"])]
    lab = ["label", "label_dw"]
    if have.get("esri"): lab += ["label_esri"]
    if have.get("dwprobs"): lab += DWP
    if have.get("worldcereal"): lab += ["wc_crop"]
    lab += ["conf", "zone"]
    return img, lab


def _sample(ee, region, year, chip, have):
    stack = _base_stack(ee, region, year, have)
    img_bands, lab_bands = _bandlists(have)
    props = {}
    for i in range(0, len(img_bands), 7):
        r = _gi(stack, img_bands[i:i + 7], region, 0)
        if r is None: return None
        props.update(r)
    for i in range(0, len(lab_bands), 7):
        r = _gi(stack, lab_bands[i:i + 7], region, 255)
        if r is None: return None
        props.update(r)
    try:
        img = np.stack([_fix(np.array(props[b], dtype=np.int16), chip) for b in img_bands], -1)
        lab = _fix(np.array(props["label"], dtype=np.uint8), chip)
        # soft/aux label stack (everything except the primary label) in declared order
        aux_bands = [b for b in lab_bands if b != "label"]
        aux = np.stack([_fix(np.array(props[b], dtype=np.int16), chip) for b in aux_bands], -1).astype(np.int16)
        conf = _fix(np.array(props["conf"], dtype=np.uint8), chip)
        zone = _fix(np.array(props["zone"], dtype=np.int16), chip)
    except Exception as e:
        if not hasattr(_sample, "_warned"):
            _sample._warned = True; print(f"[assembly] {repr(e)[:150]}", flush=True)
        return None
    lab[lab > 10] = 255
    zv = zone[(zone >= 0) & (zone < N_ZONES)]
    zc = int(np.bincount(zv.ravel(), minlength=N_ZONES).argmax()) if zv.size else -1
    return dict(img=img, lab=lab, aux=aux, aux_bands=aux_bands, conf=conf, zone=zc)


def _probe_sources(ee, ae_dims):
    """Return which optional layers actually resolve (so a dead asset isn't fatal)."""
    india = _india(ee)
    pt = ee.FeatureCollection.randomPoints(india, 1, 1).getInfo()["features"][0]["geometry"]["coordinates"]
    region = ee.Geometry.Point(pt).buffer(300).bounds()
    have = {"dwprobs": True, "ae": ae_dims}
    tests = {
        "esri": lambda: ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS")
                 .filterDate("2021-01-01", "2022-01-01").mosaic().remap(ESRI_FROM, ESRI_TO, 255)
                 .rename("x").sampleRectangle(region, defaultValue=255).getInfo(),
        "worldcereal": lambda: ee.ImageCollection("ESA/WorldCereal/2021/MODELS/v100")
                 .filter(ee.Filter.eq("product", "temporarycrops")).mosaic().select("classification")
                 .gt(0).rename("x").sampleRectangle(region, defaultValue=0).getInfo(),
    }
    for k, fn in tests.items():
        try:
            fn(); have[k] = True
        except Exception as e:
            have[k] = False; print(f"  [source {k}] unavailable -> dropped ({repr(e)[:70]})", flush=True)
    print(f"  sources: {', '.join(k for k,v in have.items() if v and k!='ae')}"
          f"{f' +AE{ae_dims}' if ae_dims else ''}", flush=True)
    return have


class _LockBusy(RuntimeError):
    pass


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                    # exists, just not signalable
    except OSError:
        return False
    return True


def _acquire_lock(out):
    """Refuse to start when another exporter already owns this chip dir.

    Two exporters writing one directory race on both the y2021_r_NNNN name counter
    and the _rows.csv append, so they can hand the same chip name to different
    coordinates. Whichever .npy wins the race then has a lon/lat that does not
    describe its pixels -- the chip lands in the wrong geographic block and
    silently reintroduces the spatial leakage this dataset exists to remove.
    That corruption is invisible without the per-chip coords, so: one writer only.
    """
    lk = out / ".export.lock"
    for _ in range(2):
        try:
            fd = os.open(str(lk), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            atexit.register(lambda: lk.unlink(missing_ok=True))
            return lk
        except FileExistsError:
            try:
                old = int(lk.read_text().strip())
            except Exception:
                old = -1
            if old > 0 and _pid_alive(old):
                raise _LockBusy(f"exporter pid {old} already owns {out} -- refusing to start")
            print(f"[dsv2] clearing stale lock (pid {old} is gone)", flush=True)
            lk.unlink(missing_ok=True)
    raise _LockBusy(f"could not acquire {lk}")


class _DiskFull(RuntimeError):
    pass


def _free_gb(path):
    import shutil
    return shutil.disk_usage(str(path)).free / 1e9


def _dupe_key(lon, lat): return (round(lon, 3), round(lat, 3))   # ~100 m near-duplicate guard


def _zone_points(ee, india, need, seed, scale=2000, over=3):
    """Random points drawn INSIDE specific agro-zones.

    Uniform points over India land in zone 14 only ~1% of the time, so filling a
    rare zone by rejection sampling costs ~100 GEE calls per chip. agro_zone is a
    deterministic image, so stratifiedSample can draw straight from the zone mask
    instead -> ~1 call per chip. Returns {zone: [(lon, lat), ...]}.
    """
    zimg = agro_zone(ee)
    cls = sorted(need)
    pts = [int(min(max(need[c] * over, 30), 300)) for c in cls]     # oversample: gates reject many
    try:
        fc = zimg.stratifiedSample(numPoints=1, classBand="zone", region=india, scale=scale,
                                   classValues=cls, classPoints=pts, seed=seed,
                                   dropNulls=True, tileScale=4, geometries=True
                                   ).getInfo()["features"]
    except Exception as e:
        print(f"  [zone-fill] stratifiedSample failed: {repr(e)[:90]}", flush=True)
        return {c: [] for c in cls}
    out = {c: [] for c in cls}
    for f in fc:
        c = int(f["properties"]["zone"])
        if c in out: out[c].append(tuple(f["geometry"]["coordinates"]))
    for c in out: random.Random(seed + c).shuffle(out[c])
    return out


def _log_row(out, row):
    """append one chip's metadata (incl lon/lat) immediately -> crash-safe coords."""
    p = out / "_rows.csv"
    pd.DataFrame([row]).to_csv(p, mode="a", header=not p.exists(), index=False)


def export_dataset_v2(project=C.EE_PROJECT, per_zone=80, per_class=20, years=(2021,),
                      chip=CHIP, out_dir=None, ae_dims=0, class_cap_frac=0.35,
                      targeted_only=False, zone_fill_rounds=6, min_free_gb=1.5, workers=8,
                      zone_fill_only=False):
    out = Path(out_dir or OUT_DIR)
    for sub in ("img", "lbl", "auxb", "conf"): (out / sub).mkdir(parents=True, exist_ok=True)
    _acquire_lock(out)                 # before any slow EE work, so a clash fails fast
    # Startup touches oauth2.googleapis.com for a token; a transient DNS blip there
    # kills the whole run before the sampling loop (which tolerates failures) is
    # ever reached. Retry with backoff so a hours-long export survives a hiccup.
    for att in range(8):
        try:
            ee = _init(project); india = _india(ee)
            have = _probe_sources(ee, ae_dims)
            break
        except Exception as e:
            wait = min(60, 5 * 2 ** att)
            print(f"[dsv2] startup attempt {att+1}/8 failed ({repr(e)[:80]}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    else:
        raise RuntimeError("EE startup failed after 8 attempts -- check network")
    existing = {p.stem for p in (out / "img").glob("*.npy")}
    rows, seen = [], set(); t0 = time.time()
    class_cap = int(per_zone * N_ZONES * class_cap_frac)          # max chips dominated by one class
    # --- RESUME: rebuild the quota counters from _rows.csv. Without this a resumed
    # run restarts every counter at 0, so saturated zones collect another full
    # `per_zone` each (skewing the distribution) while the rare zones that actually
    # stalled the run stay unfilled, and the dupe guard forgets every prior coord.
    prior = pd.read_csv(out / "_rows.csv") if (out / "_rows.csv").exists() else pd.DataFrame()
    prior = prior[prior.chip.isin(existing)] if len(prior) else prior     # ignore torn tail rows
    for year in years:
        zc_count = {z: 0 for z in range(N_ZONES)}; cls_count = {c: 0 for c in range(11)}
        if len(prior):
            py = prior[prior.year == year] if "year" in prior else prior
            for z, n in py.zone.value_counts().items():
                if 0 <= int(z) < N_ZONES: zc_count[int(z)] = int(n)
            for c, n in py.dominant.value_counts().items():
                if 0 <= int(c) < 11: cls_count[int(c)] = int(n)
            seen.update(_dupe_key(r.lon, r.lat) for r in py.itertuples())
            print(f"[dsv2 {year}] resume: {len(py)} prior chips | "
                  f"zones {dict(sorted(zc_count.items()))}", flush=True)
        saved = len([e for e in existing if e.startswith(f"y{year}_r_")]); fails = 0; tried = 0
        print(f"[dsv2 {year}] starting (per_zone={per_zone}) ...", flush=True)

        def cands():
            bi = 0
            while bi < (per_zone * N_ZONES // 40 + 50):
                try:
                    fc = ee.FeatureCollection.randomPoints(india, 4000, C.SEED + bi + year).getInfo()["features"]
                except Exception:
                    bi += 1; continue
                bi += 1
                for f in fc: yield f["geometry"]["coordinates"]

        def _commit(lon, lat, g):
            """Apply the quality/quota gates and write one chip. Runs on the calling
            thread only, so the shared counters need no lock."""
            nonlocal saved, fails
            if g is None:
                fails += 1
                if fails % 5 == 0: print(f"[dsv2 {year}] {fails} getInfo fails | {len(rows)} new", flush=True)
                return False
            lab = g["lab"]; valid = (lab != 255).mean()
            if valid < 0.5: return False
            zc = g["zone"]
            if zc >= 0 and zc_count.get(zc, 0) >= per_zone: return False
            vals, counts = np.unique(lab[lab != 255], return_counts=True)
            dom = int(vals[counts.argmax()]) if len(vals) else -1
            if dom >= 0 and cls_count.get(dom, 0) >= class_cap and counts.max() / max(counts.sum(), 1) > 0.9:
                return False                                      # class-balance: cap uniform common-class chips
            if _free_gb(out) < min_free_gb:      # stop BEFORE a half-written chip: a torn
                raise _DiskFull(                 # quadruple survives the img-keyed resume
                    f"only {_free_gb(out):.1f} GB free (< {min_free_gb} GB) after {saved} chips")
            name = f"y{year}_r_{saved:04d}"
            while name in existing:            # repaired/torn runs leave gaps -> walk to a free
                saved += 1                     # slot instead of burning a candidate point
                name = f"y{year}_r_{saved:04d}"
            np.save(out / "img" / f"{name}.npy", g["img"]); np.save(out / "lbl" / f"{name}.npy", lab)
            np.save(out / "auxb" / f"{name}.npy", g["aux"]); np.save(out / "conf" / f"{name}.npy", g["conf"])
            rows.append(dict(chip=name, lon=lon, lat=lat, year=year, zone=zc, target=-1,
                             dominant=dom, n_classes=int(len(vals)), valid_frac=round(float(valid), 3),
                             conf_frac=round(float((g["conf"] >= 2).mean()), 3)))
            _log_row(out, rows[-1])                              # crash-safe: coords logged now
            saved += 1; existing.add(name)
            if zc >= 0: zc_count[zc] += 1
            if dom >= 0: cls_count[dom] = cls_count.get(dom, 0) + 1
            if len(rows) <= 3 or len(rows) % 10 == 0:
                el = time.time() - t0; nz = sum(v > 0 for v in zc_count.values())
                on_disk = len(existing)
                print(f"[dsv2 {year}] {on_disk} chips ({len(rows)} new) | {nz}/{N_ZONES} zones | "
                      f"{el/max(tried,1):.1f}s/try | {60*len(rows)/max(el,1):.1f} chips/min | {el/60:.1f}m",
                      flush=True)
            return True

        def pump(coord_iter, done_check):
            """Sample candidates concurrently, commit them serially.

            Each _sample() is a single GEE getInfo that blocks ~30 s on the network,
            so a sequential crawl costs ~2 chips/min. Only the sampling is threaded;
            every mutation of saved/seen/zc_count happens back on this thread, so
            there is no shared-state race and no lock.
            """
            nonlocal tried
            n, it, inflight = 0, iter(coord_iter), {}
            with futures.ThreadPoolExecutor(max_workers=workers) as ex:
                while True:
                    while len(inflight) < workers and not done_check():
                        try: lon, lat = next(it)
                        except StopIteration: break
                        if _dupe_key(lon, lat) in seen: continue
                        seen.add(_dupe_key(lon, lat))   # reserve now: two workers must not
                        tried += 1                      # chase the same point concurrently
                        region = ee.Geometry.Point([lon, lat]).buffer(chip / 2 * 10).bounds()
                        inflight[ex.submit(_sample, ee, region, year, chip, have)] = (lon, lat)
                    if not inflight: break
                    done, _ = futures.wait(inflight, return_when=futures.FIRST_COMPLETED)
                    for fut in done:
                        lon, lat = inflight.pop(fut)
                        try: g = fut.result()
                        except Exception: g = None
                        n += bool(_commit(lon, lat, g))
                    if done_check(): break
            return n

        try:
            # The uniform pass only exits once EVERY zone is full, which rare zones
            # never allow -- so on a resume it burns its whole 400k-point candidate
            # budget (~2 days) re-sampling saturated zones. zone_fill_only skips
            # straight to targeted sampling, which serves common and rare zones alike.
            if not targeted_only and not zone_fill_only:
                pump(cands(), lambda: all(v >= per_zone for v in zc_count.values()))

            # --- zone-fill: rare agro-zones are effectively unreachable by uniform
            # sampling, so draw candidates from inside each under-quota zone instead.
            for rnd in range(zone_fill_rounds if not targeted_only else 0):
                need = {z: per_zone - n for z, n in zc_count.items() if n < per_zone}
                if not need: break
                print(f"[dsv2 {year}] zone-fill round {rnd + 1}: need {dict(sorted(need.items()))}", flush=True)
                pts = _zone_points(ee, india, need, C.SEED + year + rnd * 97)
                zs = set(need)
                got = pump((c for z in sorted(pts, key=lambda k: -need[k]) for c in pts[z]),
                           lambda: all(zc_count.get(z, 0) >= per_zone for z in zs))
                print(f"[dsv2 {year}] zone-fill round {rnd + 1}: +{got} chips | {saved} total", flush=True)
                if got == 0:
                    print(f"[dsv2 {year}] zone-fill stalled -> stopping", flush=True)
                    break
        except _DiskFull as e:
            print(f"[dsv2 {year}] DISK GUARD: {e} -- stopping cleanly, "
                  f"chips on disk stay consistent", flush=True)

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
            random.Random(C.SEED + cls).shuffle(cc)
            got_n = len([e for e in existing if e.startswith(f"y{year}_t{cls}_")])
            for lon, lat in cc:
                if got_n >= tgt: break
                if _dupe_key(lon, lat) in seen: continue
                name = f"y{year}_t{cls}_{got_n:04d}"
                if name in existing: got_n += 1; continue
                region = ee.Geometry.Point([lon, lat]).buffer(chip / 2 * 10).bounds()
                g = _sample(ee, region, year, chip, have)
                if g is None or (g["lab"] == cls).mean() < 0.02: continue
                seen.add(_dupe_key(lon, lat))
                np.save(out / "img" / f"{name}.npy", g["img"]); np.save(out / "lbl" / f"{name}.npy", g["lab"])
                np.save(out / "auxb" / f"{name}.npy", g["aux"]); np.save(out / "conf" / f"{name}.npy", g["conf"])
                rows.append(dict(chip=name, lon=lon, lat=lat, year=year, zone=g["zone"], target=cls,
                                 dominant=cls, n_classes=-1, valid_frac=-1, conf_frac=-1))
                _log_row(out, rows[-1])                          # crash-safe: coords logged now
                got_n += 1
            print(f"[dsv2 {year}] class {cls}: {got_n}", flush=True)

    if targeted_only:                                   # finalize_dsv2 rebuilds the manifest from disk
        print(f"[dsv2] targeted pass done ({len(rows)} new chips) -> run `python -m src.finalize_dsv2`", flush=True)
        return out
    img_bands, lab_bands = _bandlists(have)
    mp = out / "manifest.csv"
    df = pd.concat([pd.read_csv(mp), pd.DataFrame(rows)], ignore_index=True).drop_duplicates("chip") \
         if mp.exists() and rows else (pd.read_csv(mp) if mp.exists() else pd.DataFrame(rows))
    if len(df):
        df = _split(df)
        df["img_bands"] = "|".join(img_bands); df["aux_bands"] = "|".join([b for b in lab_bands if b != "label"])
        df.to_csv(mp, index=False)
        print(f"[dsv2] DONE {len(df)} chips -> {out} | splits {df.split.value_counts().to_dict()}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=C.EE_PROJECT)
    ap.add_argument("--per-zone", type=int, default=80)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--years", default="2021")
    ap.add_argument("--ae-dims", type=int, default=0)
    ap.add_argument("--targeted-only", action="store_true", help="skip random loop; only rare-class boxes")
    ap.add_argument("--out-dir", default=None, help="export to a fresh dir (e.g. for a coord-complete spatial set)")
    ap.add_argument("--zone-fill-rounds", type=int, default=6,
                    help="targeted passes sampling inside under-quota agro-zones (0 = off)")
    ap.add_argument("--zone-fill-only", action="store_true",
                    help="skip the uniform pass; go straight to targeted zone sampling")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent GEE getInfo calls (each blocks ~30 s on the network)")
    ap.add_argument("--min-free-gb", type=float, default=1.5,
                    help="stop cleanly when free disk drops below this")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    yrs = tuple(int(y) for y in a.years.split(","))
    if a.test:
        ee = _init(a.project); india = _india(ee)
        have = _probe_sources(ee, a.ae_dims)
        ib, lb = _bandlists(have); print(f"  img_bands({len(ib)}) + label/aux({len(lb)})")
        ok = 0
        for i in range(8):
            f = ee.FeatureCollection.randomPoints(india, 3, C.SEED + i).getInfo()["features"]
            lon, lat = f[0]["geometry"]["coordinates"]
            region = ee.Geometry.Point([lon, lat]).buffer(CHIP / 2 * 10).bounds()
            t = time.time(); g = _sample(ee, region, yrs[0], CHIP, have)
            if g is None: print(f"  chip {i}: FAILED"); continue
            print(f"  chip {i}: img {g['img'].shape} | aux {g['aux'].shape} {g['aux_bands']} "
                  f"| valid {100*(g['lab']!=255).mean():.0f}% | conf>=2 {100*(g['conf']>=2).mean():.0f}% "
                  f"| zone {g['zone']} | {time.time()-t:.0f}s", flush=True)
            ok += 1
            if ok >= 3: break
    else:
        export_dataset_v2(a.project, a.per_zone, a.per_class, yrs, ae_dims=a.ae_dims,
                          out_dir=a.out_dir, targeted_only=a.targeted_only,
                          zone_fill_rounds=a.zone_fill_rounds, min_free_gb=a.min_free_gb,
                          workers=a.workers, zone_fill_only=a.zone_fill_only)
