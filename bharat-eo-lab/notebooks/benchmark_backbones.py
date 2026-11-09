# %% [markdown]
# # Bharat-EO-Bench v1 · T1 — head-to-head backbone benchmark (Kaggle GPU)
#
# Fine-tunes multiple backbones on the **same T1 chips + same splits** and
# tabulates macro-F1 / mIoU: **Prithvi-EO-2.0 300M vs 600M vs an ImageNet U-Net
# baseline** (+ a Clay attempt). One row (Prithvi-300) is the 5k Prithvi re-run.
#
# **Kaggle:** upload `bharat-eo-chips-t1.zip` as a Dataset; Import this .ipynb;
# GPU T4 x2, Internet ON; + Add Input; Run All. (~a few hours for all backbones.)

# %%
# --- installs (numpy-safe: terratorch bumps numpy and breaks the compiled stack)
import importlib.metadata as md
KEEP = {p: md.version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn")}
print("preinstalled:", KEEP)
!pip install -q segmentation-models-pytorch
!pip install -q terratorch
pins = " ".join(f"{k}=={v}" for k, v in KEEP.items())
!pip install -q --no-deps --force-reinstall {pins}
!python -c "import numpy,pandas,scipy,sklearn;print('core stack OK | numpy',numpy.__version__)"

# %%
import os, glob, random, json, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, jaccard_score, accuracy_score
print("torch", torch.__version__)
def pick_device():
    if torch.cuda.is_available():
        try: _=(torch.zeros(8,device="cuda")+1).sum().item(); print("GPU:",torch.cuda.get_device_name(0)); return "cuda"
        except Exception as e: print("[warn] GPU unusable:",str(e).splitlines()[0],"-> CPU"); return "cpu"
    return "cpu"
dev = pick_device()

# config
BACKBONES = ["prithvi_eo_v2_300", "prithvi_eo_v2_600", "smp_resnet34"]
IN_BANDS, EPOCHS, K = 6, 35, 11          # 35 epochs; 600M uses grad-accum to fit T4

# %%
# --- T1 chips (robust loader + manifest split + augmentation) ----------------
imgs = glob.glob("/kaggle/input/**/img/*.npy", recursive=True) or glob.glob("data/chips_v1/img/*.npy")
ROOT = os.path.dirname(os.path.dirname(imgs[0])); print("chips root:", ROOT, "| n=", len(imgs))
mp = os.path.join(ROOT, "manifest.csv")
man = pd.read_csv(mp) if os.path.exists(mp) else pd.DataFrame(
    {"chip":[os.path.basename(p)[:-4] for p in imgs],
     "split":np.where(np.arange(len(imgs))%5==0,"val","train")})
if "split" not in man: man["split"]="train"
print("splits:", man.split.value_counts().to_dict())

def load_img(c):
    a = np.load(f"{ROOT}/img/{c}.npy").astype(np.float32)
    return np.transpose(a,(2,0,1))[:IN_BANDS]
tr_chips = man.loc[man.split=="train","chip"].tolist()
va_chips = man.loc[man.split.isin(["val","test"]),"chip"].tolist()
st = np.stack([load_img(c) for c in tr_chips[:200]])
MU = st.mean((0,2,3),keepdims=True)[0]; SD = st.std((0,2,3),keepdims=True)[0]+1e-6
cnt = np.zeros(K)
for c in tr_chips:
    l=np.load(f"{ROOT}/lbl/{c}.npy").ravel(); l=l[l!=255]; cnt+=np.bincount(l,minlength=K)
CW = torch.tensor(np.clip(cnt.sum()/(K*np.clip(cnt,1,None)),0.5,8.0).astype(np.float32))

class DS(torch.utils.data.Dataset):
    def __init__(s,ch,tr=False): s.ch,s.tr=ch,tr
    def __len__(s): return len(s.ch)
    def __getitem__(s,i):
        c=s.ch[i]; x=(load_img(c)-MU)/SD; y=np.load(f"{ROOT}/lbl/{c}.npy").astype(np.int64)
        if s.tr:
            if random.random()<.5: x=x[:,:,::-1]; y=y[:,::-1]
            if random.random()<.5: x=x[:,::-1,:]; y=y[::-1,:]
            k=random.randint(0,3); x=np.rot90(x,k,(1,2)); y=np.rot90(y,k)
        return torch.from_numpy(x.copy()).float(), torch.from_numpy(y.copy()).long()

# %%
# --- shared decoder + Prithvi wrapper + trainer ------------------------------
class UpDecoder(nn.Module):
    def __init__(s,D,k):
        super().__init__(); ch=[D,256,128,64,32]; s.b=nn.ModuleList()
        for i in range(4): s.b.append(nn.Sequential(nn.Conv2d(ch[i],ch[i+1],3,padding=1),
            nn.GroupNorm(8,ch[i+1]),nn.GELU(),nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)))
        s.c=nn.Conv2d(ch[4],k,1)
    def forward(s,f):
        for b in s.b: f=b(f)
        return s.c(f)
class FMSeg(nn.Module):
    def __init__(s,enc,k,img=224): super().__init__(); s.enc,s.k,s.img,s.dec=enc,k,img,None
    def _fm(s,x):
        try: f=s.enc(x)
        except Exception: f=s.enc(x.unsqueeze(2))
        if isinstance(f,(list,tuple)): f=f[-1]
        if f.dim()==3:
            B,N,D=f.shape; hw=int(round(N**.5))
            if hw*hw!=N: f=f[:,1:,:]; N=f.shape[1]; hw=int(round(N**.5))
            f=f.transpose(1,2).reshape(B,D,hw,hw)
        return f
    def forward(s,x):
        f=s._fm(x)
        if s.dec is None: s.dec=UpDecoder(f.shape[1],s.k).to(f.device)
        y=s.dec(f)
        return y if y.shape[-1]==s.img else F.interpolate(y,(s.img,s.img),mode="bilinear",align_corners=False)

def build(name):
    if name=="smp_resnet34":
        import segmentation_models_pytorch as smp
        return smp.Unet("resnet34",encoder_weights="imagenet",in_channels=IN_BANDS,classes=K).to(dev)
    from terratorch import BACKBONE_REGISTRY
    enc=BACKBONE_REGISTRY.build(name,pretrained=True,num_frames=1,
        bands=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"])
    try: enc.set_grad_checkpointing(True)          # cut activation memory (helps 600M)
    except Exception:
        try: enc.gradient_checkpointing_enable()
        except Exception: pass
    m=FMSeg(enc,K).to(dev)
    with torch.no_grad(): m(torch.zeros(1,IN_BANDS,224,224,device=dev))
    return m

def dice(lg,t,ig=255,eps=1.):
    msk=(t!=ig); tt=t.clone(); tt[~msk]=0
    oh=F.one_hot(tt,lg.shape[1]).permute(0,3,1,2).float()*msk.unsqueeze(1)
    p=lg.softmax(1)*msk.unsqueeze(1); inter=(p*oh).sum((0,2,3)); un=p.sum((0,2,3))+oh.sum((0,2,3))
    return 1-((2*inter+eps)/(un+eps)).mean()

def run(name):
    bs = 1 if "600" in name else 8          # 600M barely fits 16GB T4 even at batch 1
    accum = 8 // bs                          # keep effective batch ~8 via grad accumulation
    trdl=torch.utils.data.DataLoader(DS(tr_chips,True),batch_size=bs,shuffle=True,num_workers=2,drop_last=True)
    vadl=torch.utils.data.DataLoader(DS(va_chips),batch_size=8,num_workers=2)
    m=build(name)
    opt=torch.optim.AdamW(m.parameters(),lr=1.5e-4,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    sc=torch.amp.GradScaler("cuda",enabled=(dev=="cuda")); ce=nn.CrossEntropyLoss(weight=CW.to(dev),ignore_index=255)
    for ep in range(EPOCHS):
        m.train(); opt.zero_grad()
        for j,(x,y) in enumerate(trdl):
            x,y=x.to(dev),y.to(dev)
            with torch.amp.autocast("cuda",enabled=(dev=="cuda")): loss=(ce(m(x),y)+dice(m(x),y))/accum
            sc.scale(loss).backward()
            if (j+1)%accum==0: sc.step(opt); sc.update(); opt.zero_grad()
        sch.step()
    m.eval(); P,T=[],[]
    with torch.no_grad():
        for x,y in vadl:
            p=m(x.to(dev)).argmax(1).cpu().numpy().ravel(); t=y.numpy().ravel(); k=t!=255
            P.append(p[k]); T.append(t[k])
    P,T=np.concatenate(P),np.concatenate(T)
    return dict(oa=accuracy_score(T,P), mf1=f1_score(T,P,average="macro",zero_division=0),
               miou=jaccard_score(T,P,average="macro",zero_division=0))

# %%
# --- HEAD-TO-HEAD: train every backbone on the same T1 splits ----------------
results={}
for name in BACKBONES:
    try:
        print(f"\n>>> training {name} ...")
        results[name]=run(name); print("   ", results[name])
    except Exception as e:
        print(f"   [skip] {name}: {str(e).splitlines()[0][:80]}")
print("\n=== Bharat-EO-Bench v1 · T1 backbone head-to-head ===")
print(f"  {'backbone':<22} {'OA':>6} {'macro-F1':>9} {'mIoU':>6}")
for n,r in sorted(results.items(), key=lambda x:-x[1]['mf1']):
    print(f"  {n:<22} {r['oa']:.3f}  {r['mf1']:.3f}  {r['miou']:.3f}")
json.dump(results, open("backbone_headtohead.json","w"), indent=2)

# %% [markdown]
# ## Reading it (honest)
# - At **35 epochs Prithvi-300 ~= ImageNet U-Net** (macro-F1 0.593 vs 0.589) — a tie.
#   The apparent FM win at 25 epochs was a **convergence-speed** effect (Prithvi's
#   pretraining converges faster; the ImageNet model catches up given more epochs).
# - **Decoder confound:** smp uses a full U-Net decoder (skip connections); the Prithvi
#   wrapper uses a simple progressive decoder (no skips). A fair encoder-vs-encoder test
#   needs the SAME decoder on both — the next step for a clean claim.
# - Prithvi-600 (batch 1 + grad-checkpointing) tests whether scale helps on a T4.
