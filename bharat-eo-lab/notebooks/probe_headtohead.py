# %% [markdown]
# # Bharat-EO-Bench v1 · T1 — FROZEN-ENCODER PROBE (fast, cached)
#
# The clean head-to-head: **is an EO-pretrained representation better than an
# ImageNet one?** Freeze each encoder, train the SAME decoder on all -> no decoder
# confound; 600M fits a T4 (no encoder optimizer states); same-arch ViTs
# (Prithvi-300, Prithvi-600, ImageNet ViT); **3 seeds -> mean ± std**.
#
# **Speed:** a frozen encoder's output never changes, so we **encode each dataset
# ONCE** and train the decoder on cached features (~25x faster than re-encoding
# every epoch) -> all 3 encoders x 3 seeds finish well inside a Kaggle session.
#
# Kaggle: + Add Input `bharat-eo-chips-t1`, GPU T4, Internet ON, Run All.

# %%
import importlib.metadata as md
KEEP = {p: md.version(p) for p in ("numpy","pandas","scipy","scikit-learn")}
!pip install -q segmentation-models-pytorch timm
!pip install -q terratorch
pins=" ".join(f"{k}=={v}" for k,v in KEEP.items())
!pip install -q --no-deps --force-reinstall {pins}
!python -c "import numpy;print('numpy',numpy.__version__)"

# %%
import os, glob, random, json, gc, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, jaccard_score
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev)
ENCODERS = ["prithvi_eo_v2_300", "prithvi_eo_v2_600", "imagenet_vit"]
IN_BANDS, EPOCHS, K, SEEDS = 6, 30, 11, [0, 1, 2]

# %%
# --- robust chip locate: extracted dir OR auto-extract a .zip OR find manifest -
import zipfile
def _find_npy():
    for pat in ("/kaggle/input/**/img/*.npy","/kaggle/working/**/img/*.npy","data/chips_v1/img/*.npy","data/chips/img/*.npy"):
        h=glob.glob(pat,recursive=True)
        if h: return h
    return []
imgs=_find_npy()
if not imgs:
    for z in glob.glob("/kaggle/input/**/*.zip",recursive=True):
        try:
            with zipfile.ZipFile(z) as zf:
                if any(n.endswith(".npy") for n in zf.namelist()[:3000]):
                    print("extracting",z,"..."); zf.extractall("/kaggle/working/chips")
        except Exception as e: print("  zip skip",os.path.basename(z),str(e)[:50])
    imgs=_find_npy()
if not imgs:
    for m in glob.glob("/kaggle/input/**/manifest.csv",recursive=True)+glob.glob("/kaggle/working/**/manifest.csv",recursive=True):
        r=os.path.dirname(m); h=glob.glob(f"{r}/img/*.npy")
        if h: imgs=h; break
if not imgs:
    print("!! No chips found. Contents of /kaggle/input:")
    for p in glob.glob("/kaggle/input/*"): print("   ",p,"->",[os.path.basename(x) for x in glob.glob(p+'/*')][:8])
    raise SystemExit("Attach the chips dataset: Notebook -> '+ Add Input' -> add 'bharat-eo-chips-t1', then Run All.")
ROOT=os.path.dirname(os.path.dirname(imgs[0])); print("root",ROOT,"| n=",len(imgs))
mp=os.path.join(ROOT,"manifest.csv")
if os.path.exists(mp):
    man=pd.read_csv(mp)
else:
    ch=sorted(os.path.basename(p)[:-4] for p in imgs); r=np.random.RandomState(0).permutation(len(ch))
    man=pd.DataFrame({"chip":ch,"split":np.where(r<0.7*len(ch),"train",np.where(r<0.85*len(ch),"val","test"))})
    print("  [no manifest] random 70/15/15 split")
if "split" not in man.columns: man["split"]="train"
print("  splits:",man.split.value_counts().to_dict())

def load_img(c): return np.transpose(np.load(f"{ROOT}/img/{c}.npy").astype(np.float32),(2,0,1))[:IN_BANDS]
def load_lbl(c): return np.load(f"{ROOT}/lbl/{c}.npy").astype(np.uint8)   # 0..10, 255=ignore (fits uint8)
tr_chips = man.loc[man.split=="train","chip"].tolist()
va_chips = man.loc[man.split.isin(["val","test"]),"chip"].tolist()
st=np.stack([load_img(c) for c in tr_chips[:200]]); MU=st.mean((0,2,3),keepdims=True)[0]; SD=st.std((0,2,3),keepdims=True)[0]+1e-6
trL=[load_lbl(c) for c in tr_chips]; vaL=[load_lbl(c) for c in va_chips]   # preload labels once (~260MB)
cnt=np.zeros(K)
for y in trL:
    v=y.ravel(); v=v[v!=255]; cnt+=np.bincount(v,minlength=K)
CW=torch.tensor(np.clip(cnt.sum()/(K*np.clip(cnt,1,None)),0.5,8.0).astype(np.float32))
print("  train",len(tr_chips),"| val+test",len(va_chips))

# %%
# --- decoder + frozen encoders + feature cache + cached-decoder trainer -------
class UpDecoder(nn.Module):
    def __init__(s,D,k):
        super().__init__(); ch=[D,256,128,64,32]; s.b=nn.ModuleList()
        for i in range(4): s.b.append(nn.Sequential(nn.Conv2d(ch[i],ch[i+1],3,padding=1),
            nn.GroupNorm(8,ch[i+1]),nn.GELU(),nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False)))
        s.c=nn.Conv2d(ch[4],k,1)
    def forward(s,f):
        for b in s.b: f=b(f)
        return s.c(f)

def make_encoder(name):
    """returns (encoder, feat_fn) with feat_fn(x)->[B,D,h,w]; encoder frozen+eval."""
    if name=="imagenet_vit":
        import timm
        enc=timm.create_model("vit_base_patch16_224",pretrained=True,in_chans=IN_BANDS,num_classes=0).to(dev)
        def ff(x):
            f=enc.forward_features(x)
            if f.dim()==3:
                f=f[:,1:,:]; B,N,D=f.shape; hw=int(round(N**.5)); f=f.transpose(1,2).reshape(B,D,hw,hw)
            return f
    else:
        from terratorch import BACKBONE_REGISTRY
        enc=BACKBONE_REGISTRY.build(name,pretrained=True,num_frames=1,
            bands=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]).to(dev)
        def ff(x):
            try: f=enc(x)
            except Exception: f=enc(x.unsqueeze(2))
            if isinstance(f,(list,tuple)): f=f[-1]
            if f.dim()==3:
                B,N,Dd=f.shape; hw=int(round(N**.5))
                if hw*hw!=N: f=f[:,1:,:]; N=f.shape[1]; hw=int(round(N**.5))
                f=f.transpose(1,2).reshape(B,Dd,hw,hw)
            return f
    for p in enc.parameters(): p.requires_grad=False
    enc.eval()
    with torch.no_grad(): D=ff(torch.zeros(1,IN_BANDS,224,224,device=dev)).shape[1]
    return enc, ff, D

@torch.no_grad()
def cache_features(ff, chips, bs=16):
    """encode every chip ONCE through the frozen encoder -> [N,D,h,w] fp16 on CPU."""
    out=[]
    for i in range(0,len(chips),bs):
        xb=torch.stack([torch.from_numpy((load_img(c)-MU)/SD).float() for c in chips[i:i+bs]]).to(dev)
        out.append(ff(xb).half().cpu())
    return torch.cat(out)

class FeatDS(torch.utils.data.Dataset):     # trains on cached features; aug in feature+label space
    def __init__(s,feat,lab,tr=False): s.f=feat; s.y=lab; s.tr=tr
    def __len__(s): return len(s.f)
    def __getitem__(s,i):
        f=s.f[i].float(); y=s.y[i]
        if s.tr:
            if random.random()<.5: f=torch.flip(f,[2]); y=np.ascontiguousarray(y[:,::-1])
            k=random.randint(0,3)
            if k: f=torch.rot90(f,k,[1,2]); y=np.ascontiguousarray(np.rot90(y,k))
        return f, torch.from_numpy(y.astype(np.int64))

def dice(lg,t,ig=255,eps=1.):
    m=(t!=ig); tt=t.clone(); tt[~m]=0
    oh=F.one_hot(tt,lg.shape[1]).permute(0,3,1,2).float()*m.unsqueeze(1)
    p=lg.softmax(1)*m.unsqueeze(1); inter=(p*oh).sum((0,2,3)); un=p.sum((0,2,3))+oh.sum((0,2,3))
    return 1-((2*inter+eps)/(un+eps)).mean()

def probe_cached(seed, trF, vaF, D):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    dec=UpDecoder(D,K).to(dev)
    opt=torch.optim.AdamW(dec.parameters(),lr=1e-3,weight_decay=1e-4)
    sc=torch.amp.GradScaler("cuda",enabled=(dev=="cuda")); ce=nn.CrossEntropyLoss(weight=CW.to(dev),ignore_index=255)
    trdl=torch.utils.data.DataLoader(FeatDS(trF,trL,True),batch_size=32,shuffle=True,num_workers=2,drop_last=True)
    vadl=torch.utils.data.DataLoader(FeatDS(vaF,vaL,False),batch_size=32,num_workers=2)
    for ep in range(EPOCHS):
        dec.train()
        for f,y in trdl:
            f,y=f.to(dev),y.to(dev); opt.zero_grad()
            with torch.amp.autocast("cuda",enabled=(dev=="cuda")):
                lg=F.interpolate(dec(f),(224,224),mode="bilinear",align_corners=False); loss=ce(lg,y)+dice(lg,y)
            sc.scale(loss).backward(); sc.step(opt); sc.update()
    dec.eval(); P,T=[],[]
    with torch.no_grad():
        for f,y in vadl:
            lg=F.interpolate(dec(f.to(dev)),(224,224),mode="bilinear",align_corners=False)
            p=lg.argmax(1).cpu().numpy().ravel(); t=y.numpy().ravel(); k=t!=255; P.append(p[k]); T.append(t[k])
    P,T=np.concatenate(P),np.concatenate(T); del dec; torch.cuda.empty_cache()
    return f1_score(T,P,average="macro",zero_division=0), jaccard_score(T,P,average="macro",zero_division=0)

# %%
# --- run: encode each frozen encoder ONCE, then 3 fast decoder seeds ----------
import time
def _fmt(sec): return f"{sec/60:.1f}m" if sec>=60 else f"{sec:.0f}s"
t0=time.time(); res={}; seed_t=[]; cache_t=[]; NE,NS=len(ENCODERS),len(SEEDS)
print(f"plan: {NE} encoders x {NS} seeds = {NE*NS} probes | ETA updates after the 1st seed\n")
for ei,name in enumerate(ENCODERS):
    try:
        enc, ff, D = make_encoder(name)
        tc=time.time(); print(f">>> {name}: caching frozen features (D={D}) ...")
        trF=cache_features(ff,tr_chips); vaF=cache_features(ff,va_chips)
        del enc, ff; gc.collect(); torch.cuda.empty_cache()
        cache_t.append(time.time()-tc); print(f"    cached in {_fmt(cache_t[-1])}")
    except Exception as e:
        print(f"  [skip encode] {name}: {str(e).splitlines()[0][:80]}"); continue
    fs=[]
    for si,s in enumerate(SEEDS):
        ts=time.time()
        try:
            mf1,miou=probe_cached(s,trF,vaF,D); fs.append((mf1,miou)); seed_t.append(time.time()-ts)
            enc_left=NE-ei-1; seeds_left=(NS-si-1)+enc_left*NS                      # remaining work
            eta=seeds_left*np.mean(seed_t)+enc_left*(np.mean(cache_t) if cache_t else 0)
            print(f"  {name} seed{s}: macroF1 {mf1:.3f} mIoU {miou:.3f}   "
                  f"[{_fmt(seed_t[-1])}/seed | elapsed {_fmt(time.time()-t0)} | ETA ~{_fmt(eta)}]")
        except Exception as e:
            print(f"  [skip] {name} seed{s}: {str(e).splitlines()[0][:70]}")
    del trF,vaF; gc.collect(); torch.cuda.empty_cache()
    if fs:
        a=np.array(fs); res[name]={"mf1":[float(a[:,0].mean()),float(a[:,0].std())],"miou":[float(a[:,1].mean()),float(a[:,1].std())]}
        json.dump(res, open("probe_headtohead.json","w"), indent=2)   # checkpoint after each encoder

print(f"\n=== FROZEN-PROBE head-to-head (encoder representation quality) ===  [total {_fmt(time.time()-t0)}]")
print(f"  {'encoder':<20} {'macro-F1 (mean±std)':<24} {'mIoU (mean±std)'}")
for n,r in sorted(res.items(),key=lambda x:-x[1]['mf1'][0]):
    print(f"  {n:<20} {r['mf1'][0]:.3f} ± {r['mf1'][1]:.3f}          {r['miou'][0]:.3f} ± {r['miou'][1]:.3f}")

# %% [markdown]
# ## Reading it
# - Same frozen protocol + same decoder on all -> the ONLY difference is pretraining.
# - Prithvi (EO) vs ImageNet ViT: a gap **bigger than ±std** = EO pretraining helps;
#   **within ±std** = a tie (consistent with the full-finetune result).
# - Prithvi-600 vs 300 (both fit now) = does FM scale help on India LC?
# - `probe_headtohead.json` is checkpointed after each encoder, so partial runs persist.
