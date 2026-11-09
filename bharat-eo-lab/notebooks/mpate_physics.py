# %% [markdown]
# # Bharat-EarthFM prototype — **MPATE + physics head** (sovereign innovations)
#
# Moves from *using* Prithvi to *extending* it with two patented ideas, on the
# multi-temporal + SAR chips (`data/chips_temporal`, 3 monsoon phases × [6 S2 + 2 S1]):
#
# 1. **MPATE — Monsoon-Phase-Aware Temporal Encoding (P2).** Prithvi (shared weights)
#    encodes each phase; we add a **learned monsoon-phase embedding** (pre / monsoon /
#    post) so the model knows *which* phase each frame is, then fuse phases with a
#    phase-aware temporal attention. Ordinary temporal pos-embeddings only know
#    *order*; MPATE injects monsoon *semantics*.
# 2. **SAR fusion** — a small Sentinel-1 stem (cloud-proof) fused with the optical tokens.
# 3. **Physics-informed head (P5-flavoured).** An auxiliary head regresses per-pixel
#    **NDVI seasonal amplitude** (max−min NDVI across phases), a phenology/water proxy
#    computed from the bands themselves — a self-supervised physics constraint that
#    forces the representation to encode monsoon dynamics, not just a static snapshot.
#
# Ready to run the moment the temporal export lands. Kaggle: + Add Input
# `bharat-eo-chips-temporal`, GPU T4, Internet ON, Run All.

# %%
import importlib.metadata as md
KEEP = {p: md.version(p) for p in ("numpy","pandas","scipy","scikit-learn")}
!pip install -q segmentation-models-pytorch timm
!pip install -q terratorch
pins=" ".join(f"{k}=={v}" for k,v in KEEP.items())
!pip install -q --no-deps --force-reinstall {pins}
!python -c "import numpy;print('numpy',numpy.__version__)"

# %%
import os, glob, random, json, gc, time, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, jaccard_score, accuracy_score
dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev)
P_PHASES, OPT, SAR, K = 3, 6, 2, 11          # 3 monsoon phases; 6 optical + 2 SAR per phase
BANDS_PER = OPT+SAR                            # 8
EPOCHS, MODE, LAM_PHYS = 40, "finetune", 0.3  # MODE: 'finetune' | 'frozen'; physics-loss weight

# %%
# --- locate temporal chips (24-band) -----------------------------------------
import zipfile
def _find():
    for pat in ("/kaggle/input/**/img/*.npy","/kaggle/working/**/img/*.npy","data/chips_temporal/img/*.npy"):
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
    print("!! No temporal chips. Attach 'bharat-eo-chips-temporal' once the export finishes."); raise SystemExit
ROOT=os.path.dirname(os.path.dirname(imgs[0])); man=pd.read_csv(os.path.join(ROOT,"manifest.csv"))
if "split" not in man.columns: man["split"]="train"
print("root",ROOT,"| n=",len(imgs),"| splits",man.split.value_counts().to_dict())
def load_img(c): return np.transpose(np.load(f"{ROOT}/img/{c}.npy").astype(np.float32),(2,0,1))  # [24,H,W]
def load_lbl(c): return np.load(f"{ROOT}/lbl/{c}.npy").astype(np.uint8)
tr=man.loc[man.split=="train","chip"].tolist(); va=man.loc[man.split.isin(["val","test"]),"chip"].tolist()
CB=len(load_img(tr[0]))                                      # actual channel count (expect 24)
assert CB==P_PHASES*BANDS_PER, f"expected {P_PHASES*BANDS_PER} bands, got {CB}"
st=np.stack([load_img(c) for c in tr[:150]]); MU=st.mean((0,2,3),keepdims=True)[0]; SD=st.std((0,2,3),keepdims=True)[0]+1e-6
cnt=np.zeros(K)
for c in tr:
    v=load_lbl(c).ravel(); v=v[v!=255]; cnt+=np.bincount(v,minlength=K)
CW=torch.tensor(np.clip(cnt.sum()/(K*np.clip(cnt,1,None)),0.5,8.0).astype(np.float32))

def phase_slices(x):                                         # x:[B,24,H,W] -> optical,sar per phase
    opt=torch.stack([x[:,p*BANDS_PER:p*BANDS_PER+OPT] for p in range(P_PHASES)],1)   # [B,P,6,H,W]
    sar=torch.cat([x[:,p*BANDS_PER+OPT:(p+1)*BANDS_PER] for p in range(P_PHASES)],1) # [B,P*2,H,W]
    return opt,sar
def ndvi_amp(x):                                             # physics target from RAW (un-norm) bands
    # optical order [B2,B3,B4,B8A,B11,B12]; NIR=idx3, RED=idx2
    nd=[]
    for p in range(P_PHASES):
        nir=x[:,p*BANDS_PER+3]; red=x[:,p*BANDS_PER+2]; nd.append((nir-red)/(nir+red+1e-4))
    nd=torch.stack(nd,1)                                     # [B,P,H,W]
    return (nd.amax(1)-nd.amin(1)).clamp(-1,2)               # seasonal amplitude [B,H,W]

class DS(torch.utils.data.Dataset):
    def __init__(s,ch,tr=False): s.ch,s.tr=ch,tr
    def __len__(s): return len(s.ch)
    def __getitem__(s,i):
        c=s.ch[i]; raw=load_img(c); x=(raw-MU)/SD; y=load_lbl(c).astype(np.int64)
        amp=None
        if s.tr:
            if random.random()<.5: x=x[:,:,::-1]; y=y[:,::-1]; raw=raw[:,:,::-1]
            k=random.randint(0,3); x=np.rot90(x,k,(1,2)); y=np.rot90(y,k); raw=np.rot90(raw,k,(1,2))
        return (torch.from_numpy(x.copy()).float(), torch.from_numpy(raw.copy()).float(),
                torch.from_numpy(y.copy()).long())

# %%
# --- decoder (UPerNet-lite) + losses (shared with v2) ------------------------
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
        s.emb=ch; s.scales=[4,2,1,0.5][:s.L]
    def forward(s,feats,out_hw=224,return_feat=False):
        base=feats[0].shape[-2]; lats=[]
        for i,f in enumerate(feats):
            th=max(1,int(round(base*s.scales[i]))); lats.append(F.interpolate(s.lat[i](f),(th,th),mode="bilinear",align_corners=False))
        lats[-1]=s.ppm(lats[-1])
        for i in range(s.L-2,-1,-1): lats[i]=lats[i]+F.interpolate(lats[i+1],lats[i].shape[-2:],mode="bilinear",align_corners=False)
        lats=[s.fpn[i](lats[i]) for i in range(s.L)]; tgt=lats[0].shape[-2:]
        fused=s.fuse(torch.cat([F.interpolate(l,tgt,mode="bilinear",align_corners=False) for l in lats],1))
        lg=F.interpolate(s.cls(fused),(out_hw,out_hw),mode="bilinear",align_corners=False)
        return (lg,fused) if return_feat else lg
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

# %%
# --- MPATE encoder: shared Prithvi per phase + monsoon-phase embedding + fuse -
def _to_map(t):
    if t.dim()==4: return t
    B,N,D=t.shape; hw=int(round(N**.5))
    if hw*hw!=N: t=t[:,1:,:]; N=t.shape[1]; hw=int(round(N**.5))
    return t.transpose(1,2).reshape(B,D,hw,hw)
class SARStem(nn.Module):
    def __init__(s,inc,outc=128):
        super().__init__(); s.net=nn.Sequential(
            nn.Conv2d(inc,64,3,stride=2,padding=1),nn.GroupNorm(8,64),nn.GELU(),
            nn.Conv2d(64,outc,3,stride=2,padding=1),nn.GroupNorm(8,outc),nn.GELU(),
            nn.Conv2d(outc,outc,3,stride=2,padding=1),nn.GroupNorm(8,outc),nn.GELU(),
            nn.Conv2d(outc,outc,3,stride=2,padding=1),nn.GroupNorm(8,outc),nn.GELU())  # 224->14
        s.out=outc
    def forward(s,x): return s.net(x)
class MPATE(nn.Module):
    """shared Prithvi over phases + learned monsoon-phase embedding + phase attention + SAR."""
    def __init__(s,enc,run,D,finetune):
        super().__init__(); s.enc,s.run,s.D,s.ft=enc,run,D,finetune
        s.phase_emb=nn.Parameter(torch.zeros(P_PHASES,D)); nn.init.trunc_normal_(s.phase_emb,std=.02)
        s.attn=nn.Sequential(nn.Conv2d(D,D,1),nn.GELU(),nn.Conv2d(D,1,1))    # per-phase spatial attn logit
        s.sar=SARStem(P_PHASES*SAR); s.outdim=D+s.sar.out
        if not finetune:
            for p in s.enc.parameters(): p.requires_grad=False
    def _encode(s,opt_p):                                   # opt_p:[B,6,H,W] -> [B,D,14,14]
        if s.ft: f=s.run(opt_p)
        else:
            with torch.no_grad(): f=s.run(opt_p)
        f=f[-1] if isinstance(f,(list,tuple)) else f; return _to_map(f)
    def forward(s,x_norm):
        opt,sar=phase_slices(x_norm)                        # [B,P,6,H,W], [B,P*2,H,W]
        maps=[]
        for p in range(P_PHASES):
            m=s._encode(opt[:,p])                            # [B,D,14,14]
            m=m+s.phase_emb[p].view(1,-1,1,1)               # <-- MPATE: inject monsoon-phase identity
            maps.append(m)
        stk=torch.stack(maps,1)                             # [B,P,D,14,14]
        w=torch.softmax(torch.stack([s.attn(maps[p]) for p in range(P_PHASES)],1),1)  # [B,P,1,14,14]
        opt_fused=(stk*w).sum(1)                            # phase-aware temporal fusion -> [B,D,14,14]
        sar_feat=s.sar(sar)                                 # [B,128,14,14]
        return torch.cat([opt_fused,sar_feat],1)           # [B,D+128,14,14]

def make_prithvi(name="prithvi_eo_v2_300"):
    from terratorch import BACKBONE_REGISTRY
    enc=BACKBONE_REGISTRY.build(name,pretrained=True,num_frames=1,
        bands=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]).to(dev)
    def run(x):
        try: return enc(x)
        except Exception: return enc(x.unsqueeze(2))
    enc.eval()
    with torch.no_grad():
        f=run(torch.zeros(1,OPT,224,224,device=dev)); f=f[-1] if isinstance(f,(list,tuple)) else f; D=_to_map(f).shape[1]
    return enc,run,D

# %%
# --- train the MPATE + physics-head model ------------------------------------
def run_mpate(name="prithvi_eo_v2_300", seed=0):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    enc,run,D=make_prithvi(name); ft=(MODE=="finetune")
    mp=MPATE(enc,run,D,ft).to(dev)
    head=UPerHead([mp.outdim]*4,K).to(dev)                  # replicate fused map to a 4-scale pyramid
    phys=nn.Conv2d(head.emb,1,1).to(dev)                    # physics head: NDVI-amplitude regressor
    params=list(mp.parameters())+list(head.parameters())+list(phys.parameters())
    opt=torch.optim.AdamW([p for p in params if p.requires_grad],lr=(1.5e-4 if ft else 1e-3),weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
    sc=torch.amp.GradScaler("cuda",enabled=(dev=="cuda")); ce=nn.CrossEntropyLoss(weight=CW.to(dev),ignore_index=255)
    bs=4 if ft else 8
    trdl=torch.utils.data.DataLoader(DS(tr,True),batch_size=bs,shuffle=True,num_workers=2,drop_last=True)
    vadl=torch.utils.data.DataLoader(DS(va),batch_size=8,num_workers=2)
    for ep in range(EPOCHS):
        mp.train(); head.train(); phys.train()
        for xn,raw,y in trdl:
            xn,raw,y=xn.to(dev),raw.to(dev),y.to(dev); opt.zero_grad()
            with torch.amp.autocast("cuda",enabled=(dev=="cuda")):
                feat=mp(xn); lg,fused=head([feat]*4,return_feat=True)
                amp_pred=F.interpolate(phys(fused),(224,224),mode="bilinear",align_corners=False)[:,0]
                amp_gt=ndvi_amp(raw)
                seg=ce(lg,y)+dice(lg,y)+lovasz(lg,y)
                physl=F.smooth_l1_loss(amp_pred,amp_gt)      # physics-informed auxiliary
                loss=seg+LAM_PHYS*physl
            sc.scale(loss).backward(); sc.step(opt); sc.update()
        sch.step()
    mp.eval(); head.eval(); P,T=[],[]
    with torch.no_grad():
        for xn,raw,y in vadl:
            lg=head([mp(xn.to(dev))]*4)
            p=lg.argmax(1).cpu().numpy().ravel(); t=y.numpy().ravel(); k=t!=255; P.append(p[k]); T.append(t[k])
    P,T=np.concatenate(P),np.concatenate(T)
    return dict(oa=accuracy_score(T,P),mf1=f1_score(T,P,average="macro",zero_division=0),
                miou=jaccard_score(T,P,average="macro",zero_division=0))

t0=time.time(); r=run_mpate()
print(f"\n=== Bharat-EarthFM MPATE + physics head ({MODE}) ===  [{(time.time()-t0)/60:.1f}m]")
print(f"  OA {r['oa']:.3f}  macroF1 {r['mf1']:.3f}  mIoU {r['miou']:.3f}")
json.dump(r,open("mpate_physics_results.json","w"),indent=2)

# %% [markdown]
# ## Reading it
# - Compare macro-F1 to **Prithvi-Seg v2 single-date** (same decoder, same classes):
#   the lift is what **monsoon-phase temporal + SAR fusion** buys — the Bharat-EarthFM thesis.
# - **Ablations to run next** (flags at top): set `phase_emb=0` (no MPATE), drop the SAR
#   stem, or set `LAM_PHYS=0` (no physics head) to isolate each component's contribution.
# - This is a working prototype of patents **P2 (MPATE)** + a physics-informed head — the
#   first sovereign-design piece trained end-to-end, not just *using* a borrowed FM.
