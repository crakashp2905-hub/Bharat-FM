"""Dataset-integrity, spatial-distance and leakage checks for Bharat-EO-Bench.

Self-contained on purpose (numpy + pandas only, no package-relative imports): the Kaggle
training kernels inline this file verbatim, so the exact code the tests exercise is the code
that guards every training run.

Every ``check_*`` function returns a list of human-readable problems (empty = pass); the
``assert_*`` / ``require_*`` helpers raise ``IntegrityError`` and are what the kernels call.
"""
from __future__ import annotations
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd

R_EARTH_KM = 6371.0088
SPLITS = ("train", "val", "test")
REQUIRED_COLS = ("chip", "lon", "lat", "split", "split_type")
N_CLASSES = 11
IGNORE = 255
IMG_BANDS, AUX_BANDS, CHIP_PX = 17, 14, 224
CONF_AUX_IDX = 12                      # aux channel holding "# label sources agreeing"
# class index -> name (single source of truth for docs/tests/kernels)
CLASS_NAMES = ["tree", "shrub", "grass", "crop", "built", "bare",
               "snow", "water", "wetland", "mangrove", "moss"]
RARE_CLASSES = (6, 8, 9, 10)           # snow, wetland, mangrove, moss


class IntegrityError(AssertionError):
    pass


# ----------------------------------------------------------------------------- geometry
def haversine_km(lon1, lat1, lon2, lat2):
    """Pairwise great-circle distance in km, shape (len(1), len(2))."""
    la1 = np.radians(np.asarray(lat1, float))[:, None]
    la2 = np.radians(np.asarray(lat2, float))[None, :]
    dla = la2 - la1
    dlo = np.radians(np.asarray(lon2, float))[None, :] - np.radians(np.asarray(lon1, float))[:, None]
    h = np.sin(dla / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlo / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def min_distance_km(man, split_a, split_b, chunk=512):
    """Smallest great-circle distance between any chip in split_a and any chip in split_b."""
    a = man[man.split == split_a]; b = man[man.split == split_b]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    best = np.inf
    for i in range(0, len(a), chunk):
        d = haversine_km(a.lon.values[i:i + chunk], a.lat.values[i:i + chunk], b.lon.values, b.lat.values)
        best = min(best, float(d.min()))
    return best


# ----------------------------------------------------------------------------- manifest
def load_manifest(root):
    return pd.read_csv(Path(root) / "manifest.csv")


def check_manifest(man, block_deg=1.0):
    p = []
    miss = [c for c in REQUIRED_COLS if c not in man.columns]
    if miss:
        return [f"manifest missing columns {miss}"]
    if man.chip.duplicated().any():
        p.append(f"{int(man.chip.duplicated().sum())} duplicated chip ids")
    bad = sorted(set(man.split.unique()) - set(SPLITS))
    if bad:
        p.append(f"unknown split labels {bad}")
    for s in SPLITS:
        if not (man.split == s).any():
            p.append(f"split '{s}' is empty")
    if man[["lon", "lat"]].isna().any().any():
        p.append("chips with missing lon/lat")
    st = str(man.split_type.iloc[0])
    if not st.startswith("spatial_block"):
        p.append(f"split_type is {st!r}, not a spatial_block split")
    if man.split_type.nunique() != 1:
        p.append("mixed split_type values")
    # blocks must not straddle splits (recomputed from coordinates, not trusted from the file)
    blk = (np.floor(man.lon / block_deg).astype(int).astype(str) + "_"
           + np.floor(man.lat / block_deg).astype(int).astype(str))
    n_splits = man.assign(_b=blk).groupby("_b").split.nunique()
    if (n_splits > 1).any():
        p.append(f"{int((n_splits > 1).sum())} {block_deg}-degree blocks straddle more than one split")
    if "block" in man.columns:
        # NB: the exporter's `block` id is only informative; compare equivalence classes
        if man.assign(_b=blk).groupby("block")._b.nunique().max() > 1:
            p.append("manifest `block` column disagrees with lon/lat-derived blocks")
    return p


def check_spatial(man, buffer_km):
    """Train must be >= buffer_km from every val/test chip (the leakage guarantee)."""
    p = []
    for ev in ("val", "test"):
        d = min_distance_km(man, "train", ev)
        if not np.isfinite(d):
            p.append(f"cannot measure train->{ev} distance (empty split)")
        elif d < buffer_km:
            p.append(f"min train->{ev} distance {d:.1f} km < required {buffer_km:.1f} km")
    return p


def parse_buffer_km(split_type):
    """'spatial_block_1.0deg_buf25.0km' -> 25.0"""
    try:
        return float(str(split_type).split("buf")[1].replace("km", ""))
    except Exception as e:                                         # pragma: no cover
        raise IntegrityError(f"cannot parse buffer from split_type={split_type!r}") from e


# ----------------------------------------------------------------------------- files
def _p(root, sub, chip):
    return Path(root) / sub / f"{chip}.npy"


def aux_dir(root):
    """`aux` is a reserved device name on Windows; local copies use `auxb`."""
    return "auxb" if (Path(root) / "auxb").is_dir() else "aux"


def check_files(root, man, sample=None, seed=0):
    """Every manifest chip has img/lbl/aux of the right shape, dtype and value range."""
    p = []
    ad = aux_dir(root)
    chips = man.chip.tolist()
    if sample and sample < len(chips):
        chips = list(np.random.RandomState(seed).choice(chips, sample, replace=False))
    for c in chips:
        for sub in ("img", "lbl", ad):
            if not _p(root, sub, c).exists():
                p.append(f"{c}: missing {sub}/")
        if p and p[-1].startswith(c):
            continue
        img = np.load(_p(root, "img", c), mmap_mode="r")
        lbl = np.load(_p(root, "lbl", c), mmap_mode="r")
        aux = np.load(_p(root, ad, c), mmap_mode="r")
        if img.shape != (CHIP_PX, CHIP_PX, IMG_BANDS):
            p.append(f"{c}: img shape {img.shape}")
        if aux.shape != (CHIP_PX, CHIP_PX, AUX_BANDS):
            p.append(f"{c}: aux shape {aux.shape}")
        if lbl.shape != (CHIP_PX, CHIP_PX):
            p.append(f"{c}: lbl shape {lbl.shape}")
        v = np.unique(lbl)
        if ((v >= N_CLASSES) & (v != IGNORE)).any():
            p.append(f"{c}: label values outside 0..{N_CLASSES - 1}/255: {v[(v >= N_CLASSES) & (v != IGNORE)].tolist()}")
        if (lbl == IGNORE).all():
            p.append(f"{c}: label is entirely ignore")
        cf = np.asarray(aux[:, :, CONF_AUX_IDX])
        if cf.min() < 0 or cf.max() > 3:
            p.append(f"{c}: confidence band outside 0..3 ({cf.min()}..{cf.max()})")
        if not np.isfinite(np.asarray(img[:, :, :6], dtype=np.float32)).all():
            p.append(f"{c}: non-finite optical values")
    return p


def check_no_orphans_in_manifest(root, man):
    """Manifest rows must exist on disk (extra files on disk are allowed: buffer-dropped chips)."""
    on_disk = {q.stem for q in (Path(root) / "img").glob("*.npy")}
    gone = sorted(set(man.chip) - on_disk)
    return [f"{len(gone)} manifest chips have no image on disk, e.g. {gone[:3]}"] if gone else []


def check_cross_split_duplicates(root, man, hash_bytes=65536):
    """No byte-identical imagery (hash of the first `hash_bytes` + label) shared across splits."""
    seen, p = {}, []
    for c, s in zip(man.chip, man.split):
        h = hashlib.md5(_p(root, "img", c).read_bytes()[:hash_bytes]
                        + _p(root, "lbl", c).read_bytes()).hexdigest()
        if h in seen and seen[h][1] != s:
            p.append(f"identical content shared by {seen[h][0]} ({seen[h][1]}) and {c} ({s})")
        seen.setdefault(h, (c, s))
    return p


# ----------------------------------------------------------------------------- class support
def class_support(root, man, classes=range(N_CLASSES)):
    """pixels and #chips per (split, class), counted from the label files themselves."""
    rows = []
    for s in SPLITS:
        px = np.zeros(N_CLASSES, np.int64); ch = np.zeros(N_CLASSES, np.int64)
        for c in man.loc[man.split == s, "chip"]:
            b = np.bincount(np.load(_p(root, "lbl", c)).ravel(), minlength=256)[:N_CLASSES]
            px += b; ch += (b > 0)
        for k in classes:
            rows.append(dict(split=s, cls=int(k), name=CLASS_NAMES[k], px=int(px[k]), chips=int(ch[k])))
    return pd.DataFrame(rows)


def unmeasurable_classes(support, split="test", min_chips=5):
    """Classes present in fewer than `min_chips` chips of an evaluation split."""
    s = support[support.split == split]
    return s.loc[s.chips < min_chips, "name"].tolist()


# ----------------------------------------------------------------------------- rare-class pool
def build_rare_pool(root, man, classes=RARE_CLASSES, min_px=200, split="train"):
    """Chips of `split` ONLY whose label contains >= min_px pixels of a rare class.

    Derived from the label files, never from manifest['target'] (which is -1 for randomly
    sampled chips, so a `target`-based pool is silently empty).
    """
    pool = {}
    for c in man.loc[man.split == split, "chip"]:
        b = np.bincount(np.load(_p(root, "lbl", c)).ravel(), minlength=256)
        got = [k for k in classes if b[k] >= min_px]
        if got:
            pool[c] = got
    return pool


def assert_pool_train_only(pool_chips, man):
    chips = set(pool_chips)
    not_train = set(man.loc[man.split != "train", "chip"]) & chips
    if not_train:
        raise IntegrityError(f"{len(not_train)} non-train chips in the augmentation pool, e.g. {sorted(not_train)[:3]}")
    unknown = chips - set(man.chip)
    if unknown:
        raise IntegrityError(f"{len(unknown)} pool chips are not in the manifest (dropped/buffered): {sorted(unknown)[:3]}")


def check_pool_covers_rare(pool, man_root_support, classes=RARE_CLASSES):
    """A pool that cannot serve a rare class is a silent no-op: report which classes it misses."""
    have = {k for v in pool.values() for k in v}
    return [f"copy-paste pool has no chip for class {CLASS_NAMES[k]}" for k in classes if k not in have]


# ----------------------------------------------------------------------------- one-call guards
def run_all(root, sample_files=100, full_duplicates=False):
    """Return {check_name: [problems]} for a dataset directory."""
    man = load_manifest(root)
    out = {"manifest": check_manifest(man)}
    out["spatial"] = check_spatial(man, parse_buffer_km(man.split_type.iloc[0]))
    out["on_disk"] = check_no_orphans_in_manifest(root, man)
    out["files"] = check_files(root, man, sample=sample_files)
    if full_duplicates:
        out["duplicates"] = check_cross_split_duplicates(root, man)
    return out


def require_clean(root, **kw):
    res = run_all(root, **kw)
    bad = {k: v for k, v in res.items() if v}
    if bad:
        raise IntegrityError("dataset integrity failed: " + "; ".join(f"{k}: {v[:3]}" for k, v in bad.items()))
    return res
