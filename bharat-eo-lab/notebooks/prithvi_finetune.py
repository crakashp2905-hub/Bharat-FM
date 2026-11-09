# %% [markdown]
# # Bharat-EO-Lab -- fine-tune Prithvi-EO-2.0 on S2(+S1) chips (Kaggle GPU)
#
# Upgraded: **class-balanced loss + augmentation + more epochs**, a **progressive
# learned decoder** (vs the toy head), a **600M backbone** option, and **8-band
# Sentinel-1+2** chips. Robust defaults; risky options behind flags. Falls back
# to a segmentation-models-pytorch U-Net if TerraTorch fails.
#
# **Kaggle:** upload the chips zip as a Dataset; Import this .ipynb; Accelerator
# **GPU T4 x2**, **Internet ON**; + Add Input -> chips dataset; Run All.

# %%
# --- installs (Internet ON) --------------------------------------------------
# terratorch's deps try to UPGRADE numpy (-> 2.5.x), which corrupts Kaggle's
# preinstalled compiled stack (ImportError: cannot import name '_center' from
# numpy._core.umath). Fix: capture preinstalled versions, install, restore.
import importlib.metadata as md
KEEP = {p: md.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")}
print("preinstalled:", KEEP)
!pip install -q segmentation-models-pytorch
!pip install -q terratorch
pins = " ".join(f"{k}=={v}" for k, v in KEEP.items())
!pip install -q --no-deps --force-reinstall {pins}
# verify the on-disk stack imports cleanly (subprocess, before kernel imports it)
!python -c "import numpy, pandas, scipy, sklearn; print('core stack OK | numpy', numpy.__version__, '| pandas', pandas.__version__)"

# %%
# --- imports + robust device ------------------------------------------------
import os, glob, random, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, jaccard_score, accuracy_score
print("torch", torch.__version__, "| CUDA build", torch.version.cuda)

def pick_device():
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(8, device="cuda") + 1).sum().item()
            print("GPU:", torch.cuda.get_device_name(0)); return "cuda"
        except Exception as e:
            print("[warn] GPU unusable:", str(e).splitlines()[0], "-> CPU"); return "cpu"
    print("[info] no GPU -> CPU"); return "cpu"
dev = pick_device()

# %%
# --- config (edit these) ----------------------------------------------------
BACKBONE = "prithvi_eo_v2_300"   # proven. Heavier option: "prithvi_eo_v2_600"
IN_BANDS = 6                     # 6 = S2 optical (Prithvi-native). 8 = + Sentinel-1 (experimental)
EPOCHS   = 60
BATCH    = 8 if "300" in BACKBONE else 4
print(f"config: {BACKBONE} | in_bands={IN_BANDS} | epochs={EPOCHS} | batch={BATCH}")

# %%
# --- locate chips (robust) + datasets with augmentation ---------------------
imgs = glob.glob("/kaggle/input/**/img/*.npy", recursive=True) or glob.glob("data/chips/img/*.npy")
if not imgs:
    print("Contents of /kaggle/input:")
    for p in glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/**/*", recursive=True)[:40]:
        print("   ", p)
    raise FileNotFoundError("Add the chips Dataset via '+ Add Input' (needs img/ and lbl/).")
ROOT = os.path.dirname(os.path.dirname(imgs[0])); print("chips root:", ROOT)

mp = os.path.join(ROOT, "manifest.csv")
if os.path.exists(mp):
    man = pd.read_csv(mp)
    if "split" not in man.columns:
        man["split"] = np.where(np.arange(len(man)) % 5 == 0, "val", "train")
else:
    chips = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ROOT}/img/*.npy"))
    man = pd.DataFrame({"chip": chips})
    man["split"] = np.where(np.arange(len(man)) % 5 == 0, "val", "train")
    print(f"[warn] no manifest -> reconstructed {len(man)} chips, random 80/20 split")
K = 11
AVAIL = np.load(f"{ROOT}/img/{man.chip.iloc[0]}.npy").shape[-1]
IN_BANDS = min(IN_BANDS, AVAIL)
print(f"{len(man)} chips | {AVAIL}-band data | using {IN_BANDS} bands | "
      f"{(man.split=='train').sum()} train / {(man.split=='val').sum()} val")

def load_img(c):
    a = np.load(f"{ROOT}/img/{c}.npy").astype(np.float32)      # H,W,AVAIL
    return np.transpose(a, (2, 0, 1))[:IN_BANDS]               # IN_BANDS,H,W

tr_chips = man.loc[man.split == "train", "chip"].tolist()
st = np.stack([load_img(c) for c in tr_chips[:150]])
MU = st.mean((0, 2, 3), keepdims=True)[0]; SD = st.std((0, 2, 3), keepdims=True)[0] + 1e-6

# class-balanced weights from train-pixel frequencies
cnt = np.zeros(K)
for c in tr_chips:
    l = np.load(f"{ROOT}/lbl/{c}.npy").ravel(); l = l[l != 255]
    cnt += np.bincount(l, minlength=K)
# capped "balanced" weights -- raw inverse-freq (250x) over-chases rare classes
# and tanks the common ones; cap at 8x.
w = np.clip(cnt.sum() / (K * np.clip(cnt, 1, None)), 0.5, 8.0)
CW = torch.tensor(w.astype(np.float32))
print("class pixel counts:", cnt.astype(int).tolist())
print("class weights (capped 8x):", np.round(w, 2).tolist())

class ChipDS(torch.utils.data.Dataset):
    def __init__(self, chips, train=False): self.chips, self.train = chips, train
    def __len__(self): return len(self.chips)
    def __getitem__(self, i):
        c = self.chips[i]
        x = (load_img(c) - MU) / SD
        y = np.load(f"{ROOT}/lbl/{c}.npy").astype(np.int64)
        if self.train:                                          # augment
            if random.random() < 0.5: x = x[:, :, ::-1]; y = y[:, ::-1]
            if random.random() < 0.5: x = x[:, ::-1, :]; y = y[::-1, :]
            k = random.randint(0, 3); x = np.rot90(x, k, (1, 2)); y = np.rot90(y, k)
        return torch.from_numpy(x.copy()).float(), torch.from_numpy(y.copy()).long()

tr_dl = torch.utils.data.DataLoader(ChipDS(tr_chips, train=True), batch_size=BATCH,
                                    shuffle=True, num_workers=2, drop_last=True)
va_dl = torch.utils.data.DataLoader(ChipDS(man.loc[man.split=="val","chip"].tolist()),
                                    batch_size=BATCH, num_workers=2)

# %%
# --- model: Prithvi + progressive decoder (smp U-Net fallback) --------------
class UpDecoder(nn.Module):                 # 14x14 -> 224x224, learned upsampling
    def __init__(self, D, k):
        super().__init__(); chs = [D, 256, 128, 64, 32]; self.blocks = nn.ModuleList()
        for i in range(4):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(chs[i], chs[i+1], 3, padding=1), nn.GroupNorm(8, chs[i+1]),
                nn.GELU(), nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)))
        self.cls = nn.Conv2d(chs[4], k, 1)
    def forward(self, f):
        for b in self.blocks: f = b(f)
        return self.cls(f)

class PrithviSeg(nn.Module):
    def __init__(self, enc, k, img=224):
        super().__init__(); self.enc = enc; self.k = k; self.img = img; self.dec = None
    def _featmap(self, x):
        try: f = self.enc(x)
        except Exception: f = self.enc(x.unsqueeze(2))
        if isinstance(f, (list, tuple)): f = f[-1]
        if f.dim() == 3:
            B, N, D = f.shape; hw = int(round(N ** 0.5))
            if hw * hw != N: f = f[:, 1:, :]; N = f.shape[1]; hw = int(round(N ** 0.5))
            f = f.transpose(1, 2).reshape(B, D, hw, hw)
        return f
    def forward(self, x):
        f = self._featmap(x)
        if self.dec is None: self.dec = UpDecoder(f.shape[1], self.k).to(f.device)
        y = self.dec(f)
        return y if y.shape[-1] == self.img else F.interpolate(
            y, (self.img, self.img), mode="bilinear", align_corners=False)

def inflate_stem(enc, in_ch):               # best-effort: expand 6-ch patch embed
    for nm, mod in enc.named_modules():
        if isinstance(mod, (nn.Conv2d, nn.Conv3d)) and mod.in_channels == 6:
            wt = mod.weight.data; sh = list(wt.shape); sh[1] = in_ch
            nw = torch.zeros(sh, device=wt.device, dtype=wt.dtype)
            nw[:, :6] = wt; nw[:, 6:] = wt[:, :in_ch - 6]
            mod.weight = nn.Parameter(nw); mod.in_channels = in_ch
            print(f"  inflated stem {nm}: 6->{in_ch}"); return True
    return False

def build_model():
    try:
        from terratorch import BACKBONE_REGISTRY
        enc = BACKBONE_REGISTRY.build(BACKBONE, pretrained=True, num_frames=1,
              bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"])
        if IN_BANDS > 6: inflate_stem(enc, IN_BANDS)
        m = PrithviSeg(enc, K).to(dev)
        with torch.no_grad(): m(torch.zeros(1, IN_BANDS, 224, 224, device=dev))
        print(f"MODEL = {BACKBONE} (TerraTorch) + progressive decoder | {IN_BANDS}-band")
        return m, "prithvi"
    except Exception as e:
        print("[warn] Prithvi unavailable:", str(e).splitlines()[0])
    import segmentation_models_pytorch as smp
    m = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=IN_BANDS, classes=K).to(dev)
    print(f"MODEL = smp U-Net | {IN_BANDS}-band fallback"); return m, "smp"

model, kind = build_model()
opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

# loss = capped-weighted CE + Dice (Dice is robust to class imbalance)
_ce = nn.CrossEntropyLoss(weight=CW.to(dev), ignore_index=255)

def _dice(logits, target, ignore=255, eps=1.0):
    m = (target != ignore)
    t = target.clone(); t[~m] = 0
    oh = F.one_hot(t, logits.shape[1]).permute(0, 3, 1, 2).float() * m.unsqueeze(1)
    p = logits.softmax(1) * m.unsqueeze(1)
    inter = (p * oh).sum((0, 2, 3)); union = p.sum((0, 2, 3)) + oh.sum((0, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()

def lossf(logits, target):
    return _ce(logits, target) + _dice(logits, target)

# %%
# --- fine-tune --------------------------------------------------------------
def evaluate():
    model.eval(); P, T = [], []
    with torch.no_grad():
        for x, y in va_dl:
            p = model(x.to(dev)).argmax(1).cpu().numpy().ravel()
            t = y.numpy().ravel(); k = t != 255; P.append(p[k]); T.append(t[k])
    P, T = np.concatenate(P), np.concatenate(T)
    return (accuracy_score(T, P), f1_score(T, P, average="macro", zero_division=0),
            jaccard_score(T, P, average="macro", zero_division=0), P, T)

for ep in range(EPOCHS):
    model.train(); tl = 0
    for x, y in tr_dl:
        x, y = x.to(dev), y.to(dev); opt.zero_grad()
        with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
            loss = lossf(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); tl += loss.item()
    sched.step()
    if (ep + 1) % 5 == 0 or ep == 0:
        oa, mf1, miou, _, _ = evaluate()
        print(f"epoch {ep+1:2d} | loss {tl/len(tr_dl):.3f} | val OA {oa:.3f} "
              f"| macro-F1 {mf1:.3f} | mIoU {miou:.3f}")

# %%
# --- final per-class IoU ----------------------------------------------------
NAMES = ["tree", "shrub", "grass", "crop", "built", "bare", "snow", "water",
         "wetland", "mangrove", "moss"]
oa, mf1, miou, P, T = evaluate()
print(f"\nFINAL ({kind}, {BACKBONE}, {IN_BANDS}-band): "
      f"OA {oa:.3f} | macro-F1 {mf1:.3f} | mIoU {miou:.3f}")
iou = jaccard_score(T, P, average=None, labels=list(range(K)), zero_division=0)
for i in range(K):
    print(f"  {NAMES[i]:<9} IoU {iou[i]:.2f}")

# %% [markdown]
# ## Notes
# - **MODEL = Prithvi** => a fine-tuned EO foundation model; smp = robust fallback.
# - Improvements over v1: class-balanced loss, flips/rot90 augmentation, 60 epochs,
#   a learned progressive-upsampling decoder, optional 600M backbone, optional
#   Sentinel-1 bands (`IN_BANDS=8`, experimental stem inflation).
# - To push further: `IN_BANDS=8`, `BACKBONE="prithvi_eo_v2_600"`, more chips
#   (`export-chips --n 1000`), longer training.
