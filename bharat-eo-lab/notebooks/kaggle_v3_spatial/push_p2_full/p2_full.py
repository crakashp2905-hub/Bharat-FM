# %% [markdown]
# # Bharat-EO-Bench · Prithvi-Seg **v3 ABLATION** — what each improvisation is worth
# 
# Runs the proven v2 model (UPerNet-on-ViT + Lovász + LLRD + TTA → 0.663) on the
# **Dataset v2** chips (`bharat-eo-chips-dsv2`), consuming everything the richer data adds:
# - **17 input bands**: Prithvi's native 6 optical feed the ViT; the extra 11 (red-edge
#   B5/B6/B7, SAR VV/VH, elevation/slope/aspect, rain/temp/nightlights) feed an **aux stem**
#   fused as a 5th decoder pyramid level.
# - **Confidence-masking** (`USE_CONF`): ignore pixels where <2 label sources agree.
# - **Soft labels** (`USE_SOFT`): distil from Dynamic World class probabilities (aux `dwp*`).
# - **Copy-paste rare-class aug** (`USE_COPYPASTE`): paste snow/wetland/mangrove/moss instances.
# - **Class-balanced sampler** (`USE_BALANCED`): resample by dominant class.
# Every improvisation is an ablation flag → measure each contribution.
# 
# Kaggle: + Add Input `bharat-eo-chips-dsv2`, GPU T4, Internet ON, Run All. ~2–3 h.

# %% [code]
import importlib.metadata as md, subprocess, sys
KEEP = {p: md.version(p) for p in ("numpy","pandas","scipy","scikit-learn")}
# subprocess rather than IPython bang-syntax: this file may load in Kaggle's Script
# editor, where a leading-bang line is a SyntaxError. Identical in Notebook mode.
def _pip(*a): subprocess.run([sys.executable,"-m","pip","install","-q",*a], check=False)
_pip("segmentation-models-pytorch","timm")
_pip("terratorch")
pins=[f"{k}=={v}" for k,v in KEEP.items()]
_pip("--no-deps","--force-reinstall",*pins)
# fresh interpreter: numpy already imported here reports the pre-reinstall version
subprocess.run([sys.executable,"-c","import numpy;print('numpy',numpy.__version__)"])

# %% [code]
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"   # cut allocator fragmentation
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import glob, random, json, gc, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, jaccard_score, accuracy_score
dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev)
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
torch.use_deterministic_algorithms(True, warn_only=True)
PROTOCOL="p2: train-only label-derived rare pool; val/test scored separately; final-epoch; deterministic"
ARM_TAG='p2-full'; SEEDS=(0, 1, 2)

def purge():
    """Hard-release GPU memory. Jupyter pins the failed model inside the last
    traceback after an OOM, so every later attempt OOMs too -> clear that first."""
    for a in ("last_traceback","last_value","last_type"):
        if hasattr(sys,a): setattr(sys,a,None)
    try: __import__("IPython").get_ipython().user_ns.pop("_i",None)
    except Exception: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.ipc_collect()
        free,total=torch.cuda.mem_get_info()
        print(f"  [gpu] {free/1e9:.1f} GB free / {total/1e9:.1f} GB total", flush=True)
purge()

K, EPOCHS, WARMUP, SEED, MODE = 11, 35, 3, 0, "finetune"      # MODE: 'finetune' | 'frozen'
BS_FT = 2                                                      # fine-tune batch (2 fits a 16GB T4)
# dsv2 img band order: B2 B3 B4 B5 B6 B7 B8A B11 B12 VV VH elev slope aspect rain temp ntl
PRITHVI_IDX = [0,1,2,6,7,8]                                   # Prithvi native 6 optical
AUX_IDX     = [3,4,5,9,10,11,12,13,14,15,16]                  # the 11 extra context bands
DW_TO_IDX   = [7,0,2,8,3,1,4,5,6]                             # Dynamic World class i -> our class
RARE_CP     = [6,8,9,10]                                      # snow, wetland, mangrove, moss
# --- ablation flags (turn each improvisation on/off) ---
# best config: 3/3 seeds showed the class-balanced sampler HURTS (mean dmIoU +0.0111)
USE_AUXSTEM, USE_CONF, USE_SOFT, USE_COPYPASTE, USE_BALANCED = True, True, True, True, False
LAM_SOFT, CONF_MIN, P_CP = 0.3, 2, 0.4

# %% [code]
# --- locate dsv2 chips (img[17] + aux[14] + lbl) -----------------------------
import zipfile
def _find():
    for pat in ("/kaggle/input/**/img/*.npy","/kaggle/working/**/img/*.npy","data/chips_dsv2/img/*.npy"):
        h=glob.glob(pat,recursive=True)
        if h: return h
    return []
imgs=_find()
if not imgs:
    for z in glob.glob("/kaggle/input/**/*.zip",recursive=True):
        try:
            with zipfile.ZipFile(z) as zf:
                if any(n.endswith(".npy") for n in zf.namelist()[:3000]): zf.extractall("/kaggle/working/chips")
        except Exception: pass
    imgs=_find()
if not imgs:
    print("!! No dsv2 chips. Attach 'bharat-eo-chips-dsv2'."); raise SystemExit
ROOT=os.path.dirname(os.path.dirname(imgs[0])); man=pd.read_csv(os.path.join(ROOT,"manifest.csv"))
if "split" not in man.columns: man["split"]="train"
_st=str(man["split_type"].iloc[0]) if "split_type" in man.columns else "MISSING"
assert _st.startswith("spatial_block"), f"refusing to run: split_type={_st!r}; attach bharat-eo-chips-dsv2-sp"
print("split_type OK:",_st,flush=True)
print("root",ROOT,"| n=",len(imgs),"| splits",man.split.value_counts().to_dict())
# ===== inlined: src/bench_integrity.py =====
"""Dataset-integrity, spatial-distance and leakage checks for Bharat-EO-Bench.

Self-contained on purpose (numpy + pandas only, no package-relative imports): the Kaggle
training kernels inline this file verbatim, so the exact code the tests exercise is the code
that guards every training run.

Every ``check_*`` function returns a list of human-readable problems (empty = pass); the
``assert_*`` / ``require_*`` helpers raise ``IntegrityError`` and are what the kernels call.
"""
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

# ===== end inlined =====
_integ=require_clean(ROOT,sample_files=200)
print("integrity OK:",{k:len(v) for k,v in _integ.items()},"| min train->val %.1f km, train->test %.1f km" % (min_distance_km(man,'train','val'),min_distance_km(man,'train','test')),flush=True)
def load_img(c): return np.transpose(np.load(f"{ROOT}/img/{c}.npy").astype(np.float32),(2,0,1))       # [17,H,W]
# `aux` is a reserved name on Windows (os.walk skips it silently), so local exports
# use `auxb`; datasets already on Kaggle still carry `aux`. Resolve either.
AUXD="auxb" if os.path.isdir(f"{ROOT}/auxb") else "aux"
def load_aux(c): return np.transpose(np.load(f"{ROOT}/{AUXD}/{c}.npy").astype(np.float32),(2,0,1))   # [14,H,W]
def load_lbl(c): return np.load(f"{ROOT}/lbl/{c}.npy").astype(np.uint8)
CIN=len(load_img(man.chip.iloc[0])); print("input channels:",CIN,"| Prithvi",len(PRITHVI_IDX),"| aux",len(AUX_IDX))
tr=man.loc[man.split=="train","chip"].tolist()
va_chips=man.loc[man.split=="val","chip"].tolist(); te_chips=man.loc[man.split=="test","chip"].tolist()
assert not (set(tr)&set(va_chips)) and not (set(tr)&set(te_chips)) and not (set(va_chips)&set(te_chips))
tr_eval=sorted(np.random.RandomState(0).choice(tr,min(150,len(tr)),replace=False).tolist())   # fixed train subset (over-fit diagnostic)
va=va_chips+te_chips
st=np.stack([load_img(c) for c in tr[:200]]); MU=st.mean((0,2,3),keepdims=True)[0]; SD=st.std((0,2,3),keepdims=True)[0]+1e-6
cnt=np.zeros(K)
for c in tr:
    v=load_lbl(c).ravel(); v=v[v!=255]; cnt+=np.bincount(v,minlength=K)
CW=torch.tensor(np.clip(cnt.sum()/(K*np.clip(cnt,1,None)),0.5,8.0).astype(np.float32))
POOL=build_rare_pool(ROOT,man,RARE_CP,min_px=200,split="train"); assert_pool_train_only(list(POOL),man)
POOL_BY_CLS={k:[c for c,v in POOL.items() if k in v] for k in RARE_CP}; POOL_BY_CLS={k:v for k,v in POOL_BY_CLS.items() if v}
POOL_CLS=sorted(POOL_BY_CLS); CP_POOL=list(POOL)
print("rare pool (train only):",{CLASS_NAMES[k]:len(v) for k,v in POOL_BY_CLS.items()},"| unserved:",check_pool_covers_rare(POOL,None),flush=True)
if USE_COPYPASTE: assert CP_POOL, "copy-paste is ON but the train-only pool is empty (silent no-op)"
print("copy-paste rare pool:",len(CP_POOL),"chips | class pixel counts:",dict(enumerate(cnt.astype(int))))

def soft_from_aux(aux):                                       # aux[2:11]=dwp0..8 -> 11-class soft target
    dwp=aux[2:11]; soft=np.zeros((K,)+dwp.shape[1:],np.float32)
    for i,c in enumerate(DW_TO_IDX): soft[c]+=dwp[i]
    return soft/(soft.sum(0,keepdims=True)+1e-6)

class DS(torch.utils.data.Dataset):
    def __init__(s,ch,tr=False,ev=False): s.ch,s.tr,s.ev=ch,tr,ev
    def __len__(s): return len(s.ch)
    def __getitem__(s,i):
        c=s.ch[i]; raw=load_img(c); x=(raw-MU)/SD; aux=load_aux(c); y=load_lbl(c).astype(np.int64)
        soft=soft_from_aux(aux) if USE_SOFT else np.zeros((K,)+y.shape,np.float32)
        lowc=aux[12]<CONF_MIN                       # <2 label sources agree
        if USE_CONF and not s.ev: y=np.where(lowc,255,y).astype(np.int64)   # mask TRAIN only
        if s.ev: y=np.where(lowc&(y!=255),y+20,y).astype(np.int64)         # eval: keep every pixel, tag low-agreement (+20)
        if s.tr and USE_COPYPASTE and random.random()<P_CP:
            k_=random.choice(POOL_CLS); sc=random.choice(POOL_BY_CLS[k_]); sx=(load_img(sc)-MU)/SD; sl=load_lbl(sc)
            m=np.isin(sl,POOL_CLS)
            if m.any():
                x[:,m]=sx[:,m]; y[m]=sl[m]; soft[:,m]=0                       # pasted -> hard label only
        if s.tr:
            if random.random()<.5: x=x[:,:,::-1]; y=y[:,::-1]; soft=soft[:,:,::-1]
            if random.random()<.5: x=x[:,::-1,:]; y=y[::-1,:]; soft=soft[:,::-1,:]
            k=random.randint(0,3); x=np.rot90(x,k,(1,2)); y=np.rot90(y,k); soft=np.rot90(soft,k,(1,2))
        return (torch.from_numpy(x.copy()).float(), torch.from_numpy(soft.copy()).float(),
                torch.from_numpy(y.copy()).long())

def make_sampler():
    if not USE_BALANCED or "dominant" not in man.columns: return None
    dom=man.set_index("chip").loc[tr,"dominant"].to_numpy()
    freq=np.bincount(dom[dom>=0],minlength=K).astype(np.float32); freq[freq==0]=1
    w=np.array([1.0/freq[d] if d>=0 else 1.0/freq.mean() for d in dom])
    return torch.utils.data.WeightedRandomSampler(torch.from_numpy(w).double(),len(w),replacement=True)

# %% [code]
# --- decoder + losses (from v2) + aux stem -----------------------------------
class PPM(nn.Module):
    def __init__(s,inc,outc,scales=(1,2,3,6)):
        super().__init__()
        s.st=nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(sc),nn.Conv2d(inc,outc,1),nn.GroupNorm(8,outc),nn.GELU()) for sc in scales])
        s.out=nn.Sequential(nn.Conv2d(inc+len(scales)*outc,outc,3,padding=1),nn.GroupNorm(8,outc),nn.GELU())
    def forward(s,x):
        h,w=x.shape[-2:]; o=[x]
        for st in s.st: o.append(F.interpolate(st(x),(h,w),mode="bilinear",align_corners=False))
        return s.out(torch.cat(o,1))
class UPerHead(nn.Module):
    def __init__(s,dims,k,ch=256):
        super().__init__(); s.L=len(dims)
        s.lat=nn.ModuleList([nn.Conv2d(d,ch,1) for d in dims]); s.ppm=PPM(ch,ch)
        s.fpn=nn.ModuleList([nn.Sequential(nn.Conv2d(ch,ch,3,padding=1),nn.GroupNorm(8,ch),nn.GELU()) for _ in range(s.L)])
        s.fuse=nn.Sequential(nn.Conv2d(s.L*ch,ch,3,padding=1),nn.GroupNorm(8,ch),nn.GELU()); s.cls=nn.Conv2d(ch,k,1)
        s.scales=([4,2,1,0.5,0.25]+[0.25]*8)[:s.L]            # supports up to 5+ levels (aux = 5th)
    def forward(s,feats,out_hw=224):
        base=feats[0].shape[-2]; lats=[]
        for i,f in enumerate(feats):
            th=max(1,int(round(base*s.scales[i]))); lats.append(F.interpolate(s.lat[i](f),(th,th),mode="bilinear",align_corners=False))
        lats[-1]=s.ppm(lats[-1])
        for i in range(s.L-2,-1,-1): lats[i]=lats[i]+F.interpolate(lats[i+1],lats[i].shape[-2:],mode="bilinear",align_corners=False)
        lats=[s.fpn[i](lats[i]) for i in range(s.L)]; tgt=lats[0].shape[-2:]
        x=s.fuse(torch.cat([F.interpolate(l,tgt,mode="bilinear",align_corners=False) for l in lats],1))
        return F.interpolate(s.cls(x),(out_hw,out_hw),mode="bilinear",align_corners=False)
class AuxStem(nn.Module):                                     # 11 context bands 224->14
    def __init__(s,inc,outc=128):
        super().__init__(); s.net=nn.Sequential(
            nn.Conv2d(inc,64,3,2,1),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv2d(64,outc,3,2,1),nn.GroupNorm(8,outc),nn.GELU(),
            nn.Conv2d(outc,outc,3,2,1),nn.GroupNorm(8,outc),nn.GELU(),
            nn.Conv2d(outc,outc,3,2,1),nn.GroupNorm(8,outc),nn.GELU()); s.outc=outc
    def forward(s,x): return s.net(x)
def _lg(g):
    p=len(g); gts=g.sum(); inter=gts-g.float().cumsum(0); union=gts+(1-g).float().cumsum(0); jac=1.-inter/union
    if p>1: jac[1:p]=jac[1:p]-jac[0:-1]
    return jac
def lovasz(logits,labels,ig=255):
    pr=logits.softmax(1); B,C,H,W=pr.shape; pr=pr.permute(0,2,3,1).reshape(-1,C); lb=labels.reshape(-1); m=lb!=ig
    pr,lb=pr[m],lb[m]
    if lb.numel()==0: return logits.sum()*0.
    L=[]
    for c in range(C):
        fg=(lb==c).float()
        if fg.sum()==0: continue
        err=(fg-pr[:,c]).abs(); es,pm=torch.sort(err,0,descending=True); L.append(torch.dot(es,_lg(fg[pm])))
    return torch.stack(L).mean() if L else logits.sum()*0.
def dice(lg,t,ig=255,eps=1.):
    m=(t!=ig); tt=t.clone(); tt[~m]=0; oh=F.one_hot(tt,lg.shape[1]).permute(0,3,1,2).float()*m.unsqueeze(1)
    p=lg.softmax(1)*m.unsqueeze(1); i=(p*oh).sum((0,2,3)); u=p.sum((0,2,3))+oh.sum((0,2,3)); return 1-((2*i+eps)/(u+eps)).mean()
def soft_ce(logits,soft,valid):                               # DW soft-label distillation, masked
    lp=F.log_softmax(logits,1); has=(soft.sum(1)>0.5)&valid
    if has.sum()==0: return logits.sum()*0.
    return -(soft*lp).sum(1)[has].mean()

# %% [code]
# --- encoder (Prithvi/ViT multi-level via hooks, from v2) ---------------------
def _find_blocks(enc):
    best=None
    for _,m in enc.named_modules():
        if isinstance(m,nn.ModuleList) and len(m)>=8 and (best is None or len(m)>len(best)): best=m
    return best
def _to_map(t):
    if t.dim()==4: return t
    B,N,D=t.shape; hw=int(round(N**.5))
    if hw*hw!=N: t=t[:,1:,:]; N=t.shape[1]; hw=int(round(N**.5))
    return t.transpose(1,2).reshape(B,D,hw,hw)
def make_encoder(name="prithvi_eo_v2_300"):
    from terratorch import BACKBONE_REGISTRY
    enc=BACKBONE_REGISTRY.build(name,pretrained=True,num_frames=1,
        bands=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]).to(dev)
    def run(x):
        try: return enc(x)
        except Exception: return enc(x.unsqueeze(2))
    blocks=_find_blocks(enc); cap={}
    idx=[int(round(f*(len(blocks)-1))) for f in (0.25,0.5,0.75,1.0)]
    handles=[]
    def mk(j):
        def h(m,i,o): cap[j]=o
        return h
    for j,bi in enumerate(idx): handles.append(blocks[bi].register_forward_hook(mk(j)))
    def feats_fn(x):
        cap.clear(); _=run(x); out=[_to_map(cap[j]) for j in range(4)]; cap.clear(); return out
    enc.eval()
    with torch.no_grad(): dims=[m.shape[1] for m in feats_fn(torch.zeros(1,len(PRITHVI_IDX),224,224,device=dev))]
    enc._hook_handles=handles; enc._cap=cap          # kept so cleanup can release them
    return enc, feats_fn, dims

class V3(nn.Module):
    def __init__(s,enc,feats_fn,dims):
        super().__init__(); s.enc,s.feats_fn=enc,feats_fn
        s.aux=AuxStem(len(AUX_IDX)) if USE_AUXSTEM else None
        s.head=UPerHead(dims+([s.aux.outc] if USE_AUXSTEM else []),K)
    def forward(s,x,ft):
        pin=x[:,PRITHVI_IDX]
        f=s.feats_fn(pin) if ft else [t.detach() for t in s.feats_fn(pin)]
        if USE_AUXSTEM: f=f+[s.aux(x[:,AUX_IDX])]
        return s.head(f)

def build_optim(enc,model,mode,base=1.5e-4,dcy=0.85):
    head_aux=[p for n,p in model.named_parameters() if not n.startswith("enc.")]
    if mode=="frozen":
        for p in enc.parameters(): p.requires_grad=False
        return torch.optim.AdamW([p for p in head_aux if p.requires_grad],lr=1e-3,weight_decay=1e-4)
    blocks=_find_blocks(enc); nb=len(blocks); groups=[{"params":head_aux,"lr":1e-3}]; seen=set()
    for i,blk in enumerate(blocks):
        ps=[p for p in blk.parameters() if p.requires_grad]
        for p in ps: seen.add(id(p))
        groups.append({"params":ps,"lr":base*(dcy**(nb-1-i))})
    rest=[p for p in enc.parameters() if p.requires_grad and id(p) not in seen]
    if rest: groups.append({"params":rest,"lr":base*(dcy**nb)})
    return torch.optim.AdamW(groups,weight_decay=1e-4)

@torch.no_grad()
def tta(fwd,x):
    ts=[lambda a:a, lambda a:torch.flip(a,[3]), lambda a:torch.flip(a,[2]), lambda a:torch.flip(a,[2,3])]
    return sum(t(fwd(t(x))).softmax(1) for t in ts)/len(ts)

# %% [code]
# --- train v3 ----------------------------------------------------------------
def _collect(model,dl,tta_on=True):
    P,T,C=[],[],[]
    with torch.no_grad():
        for x,soft,y in dl:
            xx=x.to(dev); prob=tta(lambda a: model(a,False), xx) if tta_on else model(xx,False)
            p=prob.argmax(1).cpu().numpy().ravel(); t=y.numpy().ravel(); k=t!=255
            lowc=(t>=20)&k; t=np.where(lowc,t-20,t)
            P.append(p[k]); T.append(t[k]); C.append(~lowc[k])
    return np.concatenate(P),np.concatenate(T),np.concatenate(C)

def _score(P,T,C):
    def m(P_,T_):
        iou=jaccard_score(T_,P_,average=None,zero_division=0,labels=list(range(K)))
        return dict(oa=accuracy_score(T_,P_),mf1=f1_score(T_,P_,average="macro",zero_division=0),
                    miou=jaccard_score(T_,P_,average="macro",zero_division=0),
                    per_class_iou=[round(float(v),3) for v in iou],px=np.bincount(T_,minlength=K).tolist())
    return dict(all_px=m(P,T),conf_px=m(P[C],T[C]),frac_confident=float(C.mean()))

def run_v3(name="prithvi_eo_v2_300",seed=SEED):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    enc,feats_fn,dims=make_encoder(name); model=V3(enc,feats_fn,dims).to(dev)
    ft=(MODE=="finetune"); opt=build_optim(enc,model,MODE); bs=BS_FT if ft else 8; accum=max(1,8//bs)   # effective batch stays ~8 via accumulation
    warm=torch.optim.lr_scheduler.LinearLR(opt,0.1,1.0,total_iters=WARMUP)
    cos=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS-WARMUP)
    sch=torch.optim.lr_scheduler.SequentialLR(opt,[warm,cos],milestones=[WARMUP])
    sc=torch.amp.GradScaler("cuda",enabled=(dev=="cuda")); ce=nn.CrossEntropyLoss(weight=CW.to(dev),ignore_index=255)
    smp=make_sampler()
    def _winit(w): random.seed(seed*1000+w); np.random.seed(seed*1000+w)
    _g=torch.Generator(); _g.manual_seed(seed)
    trdl=torch.utils.data.DataLoader(DS(tr,True),batch_size=bs,sampler=smp,shuffle=(smp is None),num_workers=2,drop_last=True,worker_init_fn=_winit,generator=_g)
    def _dl(ch): return torch.utils.data.DataLoader(DS(ch,ev=True),batch_size=8,num_workers=2)
    vadl=_dl(va_chips)                                         # monitoring only; never used to pick a checkpoint
    for ep in range(EPOCHS):
        if ft: model.train()
        else:
            model.eval(); model.head.train()
            if USE_AUXSTEM: model.aux.train()
        opt.zero_grad()
        for j,(x,soft,y) in enumerate(trdl):
            x,soft,y=x.to(dev),soft.to(dev),y.to(dev)
            with torch.amp.autocast("cuda",enabled=(dev=="cuda")):
                lg=model(x,ft); loss=ce(lg,y)+dice(lg,y)+lovasz(lg,y)
                if USE_SOFT: loss=loss+LAM_SOFT*soft_ce(lg,soft,y!=255)
                loss=loss/accum
            sc.scale(loss).backward()
            if (j+1)%accum==0: sc.step(opt); sc.update(); opt.zero_grad()
        sch.step()
        if (ep+1)%5==0 or ep==EPOCHS-1:
            model.eval(); _m=_collect(model,vadl,tta_on=False); model.train() if ft else None
            print(f'    ep {ep+1:>2}/{EPOCHS} val(conf-px mIoU, no TTA) {_score(*_m)["conf_px"]["miou"]:.4f}',flush=True)
    model.eval(); out={}
    cols={"train":_collect(model,_dl(tr_eval)),"val":_collect(model,_dl(va_chips)),"test":_collect(model,_dl(te_chips))}
    for nm,(P_,T_,C_) in cols.items(): out[nm]=_score(P_,T_,C_)
    P_,T_,C_=[np.concatenate([cols["val"][i_],cols["test"][i_]]) for i_ in range(3)]
    out["val+test"]=_score(P_,T_,C_)                               # protocol-1 comparable pooled number
    # --- free EVERYTHING before returning (else each ablation config leaks a 300M
    #     model + optimizer states on the GPU -> CUDA OOM after a few configs) ---
    for h in getattr(enc,"_hook_handles",[]): h.remove()
    getattr(enc,"_cap",{}).clear()
    opt.zero_grad(set_to_none=True)
    del model, enc, feats_fn, opt, sc, sch, trdl, vadl, smp
    gc.collect(); torch.cuda.empty_cache()
    if dev=="cuda":
        print(f"    [gpu freed -> {torch.cuda.memory_allocated()/1e9:.2f} GB allocated]", flush=True)
    return out

t0=time.time(); res={}
import hashlib, platform
def _md5(p):
    try: return hashlib.md5(open(p,"rb").read()).hexdigest()
    except Exception: return None
META=dict(protocol=PROTOCOL,arm=ARM_TAG,seeds=list(SEEDS),flags=dict(auxstem=USE_AUXSTEM,conf=USE_CONF,soft=USE_SOFT,copypaste=USE_COPYPASTE,balanced=USE_BALANCED),
          epochs=EPOCHS,manifest_md5=_md5(os.path.join(ROOT,"manifest.csv")),split_type=_st,
          n_train=len(tr),n_val=len(va_chips),n_test=len(te_chips),pool={CLASS_NAMES[k]:len(v) for k,v in POOL_BY_CLS.items()},
          torch=torch.__version__,python=platform.python_version(),gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
try:
    import terratorch; META["terratorch"]=getattr(terratorch,"__version__",None)
except Exception: pass
CKPT="seg_p2_p2-full.json"
for _i,sd in enumerate(SEEDS):
    try: r=run_v3(seed=sd); oom=False
    except torch.cuda.OutOfMemoryError: oom=True
    if oom: purge(); r=run_v3(seed=sd)
    res[f"s{sd}#{_i}"]=r; json.dump(dict(meta=META,runs=res),open(CKPT,"w"),indent=2); purge()
    print(f"seed {sd} (#{_i}) done | TRAIN conf mIoU {r['train']['conf_px']['miou']:.4f} | VAL {r['val']['conf_px']['miou']:.4f}/{r['val']['all_px']['miou']:.4f} | TEST {r['test']['conf_px']['miou']:.4f}/{r['test']['all_px']['miou']:.4f} (conf/all)",flush=True)
print("=== p2-full: all runs done ===",flush=True)
