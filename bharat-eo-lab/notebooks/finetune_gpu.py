# %% [markdown]
# # Bharat-EO-Lab -- Cloud GPU bridge (Kaggle / Colab)
#
# Trains a deeper PyTorch model on the **AlphaEarth embeddings** (the same India
# table from the laptop) and compares it to XGBoost. Robust against Kaggle's
# common GPU-kernel mismatch: it tests the GPU and falls back to CPU if needed
# (this model is tiny, so CPU is fine).
#
# **How to use on Kaggle**
# 1. Datasets -> New Dataset -> upload `data/raw/india_landcover_2021_boosted.csv`.
# 2. New Notebook -> File -> Import Notebook -> this .ipynb.
# 3. Settings -> Accelerator -> **GPU T4 x2** (optional; CPU works too).
# 4. + Add Input -> add your dataset. Then Run All.

# %%
# --- imports + environment diagnostics --------------------------------------
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
print("torch", torch.__version__, "| built for CUDA", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0),
          "| device sm_%d%d" % torch.cuda.get_device_capability(0))
    print("torch has kernels for:", torch.cuda.get_arch_list())

def pick_device():
    """torch.cuda.is_available() can be True yet no kernel image exists for the
    device (Kaggle 'no kernel image' error). So actually launch a kernel."""
    if torch.cuda.is_available():
        try:
            _ = (torch.zeros(16, device="cuda") + 1).sum().item()   # real launch
            return "cuda"
        except Exception as e:
            print("[warn] GPU present but unusable:", str(e).splitlines()[0])
            print("       -> using CPU (model is tiny; this is fine).")
            print("       For GPU: Settings -> Accelerator -> GPU T4 x2, then restart.")
            return "cpu"
    print("[info] no GPU selected -> CPU (Settings -> Accelerator -> GPU to change).")
    return "cpu"

dev = pick_device(); print("device:", dev)

# %%
# --- load the India sample (auto-found under /kaggle/input) ------------------
import glob
_cands = (glob.glob("/kaggle/input/**/*landcover*boosted*.csv", recursive=True)
          or glob.glob("/kaggle/input/**/*.csv", recursive=True)
          or ["india_landcover_2021_boosted.csv"])   # local fallback
assert _cands, ("No CSV found. On Kaggle: + Add Input -> add the dataset that "
                "contains india_landcover_2021_boosted.csv.")
CSV = _cands[0]; print("using:", CSV)
df = pd.read_csv(CSV).dropna(subset=["label", "lon", "lat"]).reset_index(drop=True)
EMB = [f"A{i:02d}" for i in range(64)]
missing = [c for c in EMB if c not in df.columns]
assert not missing, f"CSV is missing embedding columns {missing[:3]}... wrong file?"
df["label"] = df["label"].astype(int)
print(df.shape, "| classes:", sorted(df["label"].unique().tolist()))

# %%
# --- leakage-free spatial block split (same logic as the laptop) ------------
def spatial_block_split(df, block_deg=1.0, test_frac=0.2, seed=42):
    bx = np.floor(df["lon"].to_numpy() / block_deg).astype(int)
    by = np.floor(df["lat"].to_numpy() / block_deg).astype(int)
    blocks = np.char.add(np.char.add(bx.astype(str), "_"), by.astype(str))
    uniq = np.unique(blocks); rng = np.random.default_rng(seed); rng.shuffle(uniq)
    test = set(uniq[:max(1, int(len(uniq) * test_frac))].tolist())
    m = np.array([b in test for b in blocks])
    return ~m, m

tr, te = spatial_block_split(df)
if tr.sum() == 0 or te.sum() == 0:                       # guard degenerate split
    print("[warn] spatial split degenerate -> random 80/20")
    m = np.random.default_rng(42).random(len(df)) < 0.2
    tr, te = ~m, m

classes = sorted(df["label"].unique().tolist())
to_idx = {c: i for i, c in enumerate(classes)}
X = df[EMB].to_numpy("float32")
y = np.array([to_idx[v] for v in df["label"]], dtype=np.int64)
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]

mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6                  # standardise on train
Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
print(f"train {len(ytr)} | test {len(yte)} | classes {len(classes)}")

# %%
# --- a small MLP head on the embeddings -------------------------------------
K = len(classes)

class MLP(nn.Module):
    def __init__(self, d=64, k=K):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, k))
    def forward(self, x): return self.net(x)

model = MLP().to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

counts = np.bincount(ytr, minlength=K).astype(np.float64)   # class-balanced weights
inv = 1.0 / np.clip(counts, 1, None)
cw = torch.tensor((inv / inv.sum() * K).astype(np.float32), device=dev)
lossf = nn.CrossEntropyLoss(weight=cw)

Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=dev)
ytr_t = torch.tensor(ytr, dtype=torch.long, device=dev)
Xte_t = torch.tensor(Xte, dtype=torch.float32, device=dev)

for epoch in range(300):
    model.train(); opt.zero_grad()
    loss = lossf(model(Xtr_t), ytr_t); loss.backward(); opt.step()
    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            p = model(Xte_t).argmax(1).cpu().numpy()
        print(f"epoch {epoch+1:3d} | loss {loss.item():.3f} "
              f"| OA {accuracy_score(yte, p):.3f} "
              f"| macroF1 {f1_score(yte, p, average='macro', zero_division=0):.3f}")

# %%
# --- compare against XGBoost on the SAME split ------------------------------
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                    n_jobs=-1, random_state=42)        # num_class/objective auto-inferred
xgb.fit(X[tr], y[tr]); xp = xgb.predict(X[te])
print("XGBoost  OA", round(accuracy_score(yte, xp), 3),
      "| macroF1", round(f1_score(yte, xp, average="macro", zero_division=0), 3))

# %% [markdown]
# ## Next: full foundation-model fine-tuning (the real GPU payoff)
# The step above trains a head on *frozen* AlphaEarth embeddings. To fine-tune a
# real EO foundation model you need **image patches**, not point embeddings:
# 1. Export Sentinel-1/2 chips + WorldCover masks (GeoTIFF) from Earth Engine.
# 2. `pip install terratorch`; use a `SemanticSegmentationTask` with a
#    Prithvi-EO-2.0 backbone + decoder; fine-tune on the GPU.
# 3. Score on the SAME spatial-block split to compare with Bharat-EO-Bench.
#
# Ask the laptop assistant to build the GEE image-chip exporter + TerraTorch
# notebook for that step.
