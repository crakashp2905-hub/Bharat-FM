# %% [markdown]
# # Bharat-EO-Bench v1 · T1 — Prithvi-Seg **v2** (highly-improvised model)
#
# Everything the earlier runs left on the table, in one model:
# - **UPerNet-on-ViT decoder** — pulls features from 4 transformer depths (forward
#   hooks), builds a scale pyramid, PPM + FPN fusion (vs the old single-map upsampler).
# - **Loss = CE(capped-weights) + Dice + Lovász-softmax** (direct IoU optimisation).
# - **Layer-wise LR decay** fine-tuning (encoder early blocks learn slower).
# - **Test-time augmentation** (flips/rot averaged) at eval.
# - **Multi-backbone sweep**: Prithvi-300 (fine-tune), Prithvi-600 (frozen — fits),
#   ImageNet ViT, best-effort Clay / TerraMind, then a **softmax ensemble**.
#
# Same T1 chips + splits as the benchmark, so v2 is directly comparable to v1
# (Prithvi-300 v1 tie ≈ 0.59 macro-F1). Kaggle: + Add Input `bharat-eo-chips-t1`,
# GPU T4, Internet ON, Run All. Budget ~2–3 h for the full sweep (trim RUNS to cut).

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
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev)
IN_BANDS, K, EPOCHS, WARMUP, SEEDS = 6, 11, 35, 3, [0]
# (name, mode): 'finetune' updates the encoder (LLRD); 'frozen' trains only the decoder.
RUNS = [
    ("prithvi_eo_v2_300", "finetune"),   # the star: v2 decoder + LLRD should beat v1's 0.59
    ("prithvi_eo_v2_600", "frozen"),     # 600M fits frozen (no encoder optimiser states)
    ("imagenet_vit",      "finetune"),   # fair generic baseline, SAME v2 decoder
    ("clay_v1",           "frozen"),     # best-effort (skipped cleanly if registry/bands differ)
    ("terramind_v1_base", "frozen"),     # best-effort
]

# %%
# --- robust chip locate (extracted dir / auto-extract zip / manifest) ---------
import zipfile
def _find_npy():
    for pat in ("/kaggle/input/**/img/*.npy","/kaggle/working/**/img/*.npy","data/chips_v1/img/*.npy"):
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
    for m in glob.glob("/kaggle/input/**/manifest.csv",recursive=True):
        r=os.path.dirname(m); h=glob.glob(f"{r}/img/*.npy")
        if h: imgs=h; break
if not imgs:
    print("!! No chips. /kaggle/input:"); [print("  ",p) for p in glob.glob("/kaggle/input/*")]
    raise SystemExit("Attach dataset: '+ Add Input' -> 'bharat-eo-chips-t1', then Run All.")
ROOT=os.path.dirname(os.path.dirname(imgs[0])); print("root",ROOT,"| n=",len(imgs))
man=pd.read_csv(os.path.join(ROOT,"manifest.csv"))
if "split" not in man.columns: man["split"]="train"
print("  splits:",man.split.value_counts().to_dict())
def load_img(c): return np.transpose(np.load(f"{ROOT}/img/{c}.npy").astype(np.float32),(2,0,1))[:IN_BANDS]
def load_lbl(c): return np.load(f"{ROOT}/lbl/{c}.npy").astype(np.uint8)
tr_chips=man.loc[man.split=="train","chip"].tolist()
va_chips=man.loc[man.split.isin(["val","test"]),"chip"].tolist()
st=np.stack([load_img(c) for c in tr_chips[:200]]); MU=st.mean((0,2,3),keepdims=True)[0]; SD=st.std((0,2,3),keepdims=True)[0]+1e-6
cnt=np.zeros(K)
for c in tr_chips:
    v=load_lbl(c).ravel(); v=v[v!=255]; cnt+=np.bincount(v,minlength=K)
CW=torch.tensor(np.clip(cnt.sum()/(K*np.clip(cnt,1,None)),0.5,8.0).astype(np.float32))
class DS(torch.utils.data.Dataset):
    def __init__(s,ch,tr=False): s.ch,s.tr=ch,tr
    def __len__(s): return len(s.ch)
    def __getitem__(s,i):
        c=s.ch[i]; x=(load_img(c)-MU)/SD; y=load_lbl(c).astype(np.int64)
        if s.tr:
            if random.random()<.5: x=x[:,:,::-1]; y=y[:,::-1]
            if random.random()<.5: x=x[:,::-1,:]; y=y[::-1,:]
            k=random.randint(0,3); x=np.rot90(x,k,(1,2)); y=np.rot90(y,k)
        return torch.from_numpy(x.copy()).float(), torch.from_numpy(y.copy()).long()

# %%
# --- UPerNet-on-ViT decoder + losses -----------------------------------------
class PPM(nn.Module):
    def __init__(s,inc,outc,scales=(1,2,3,6)):
        super().__init__()
        s.st=nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(sc),nn.Conv2d(inc,outc,1),
             nn.GroupNorm(8,outc),nn.GELU()) for sc in scales])
        s.out=nn.Sequential(nn.Conv2d(inc+len(scales)*outc,outc,3,padding=1),nn.GroupNorm(8,outc),nn.GELU())
    def forward(s,x):
        h,w=x.shape[-2:]; outs=[x]
        for st in s.st: outs.append(F.interpolate(st(x),(h,w),mode="bilinear",align_corners=False))
        return s.out(torch.cat(outs,1))

class UPerHead(nn.Module):
    """4 ViT feature levels -> resample to a pyramid -> PPM(coarsest) + top-down FPN -> logits."""
    def __init__(s,dims,k,ch=256):
        super().__init__(); s.L=len(dims)
        s.lat=nn.ModuleList([nn.Conv2d(d,ch,1) for d in dims])
        s.ppm=PPM(ch,ch)
        s.fpn=nn.ModuleList([nn.Sequential(nn.Conv2d(ch,ch,3,padding=1),nn.GroupNorm(8,ch),nn.GELU()) for _ in range(s.L)])
        s.fuse=nn.Sequential(nn.Conv2d(s.L*ch,ch,3,padding=1),nn.GroupNorm(8,ch),nn.GELU())
        s.cls=nn.Conv2d(ch,k,1)
        s.scales=[4,2,1,0.5][:s.L]                        # finest..coarsest resample factors
    def forward(s,feats,out_hw=224):
        base=feats[0].shape[-2]; lats=[]
        for i,f in enumerate(feats):
            th=max(1,int(round(base*s.scales[i])))
            lats.append(F.interpolate(s.lat[i](f),(th,th),mode="bilinear",align_corners=False))
        lats[-1]=s.ppm(lats[-1])                           # PPM on the coarsest level
        for i in range(s.L-2,-1,-1):                       # top-down fuse
            lats[i]=lats[i]+F.interpolate(lats[i+1],lats[i].shape[-2:],mode="bilinear",align_corners=False)
        lats=[s.fpn[i](lats[i]) for i in range(s.L)]
        tgt=lats[0].shape[-2:]
        x=s.fuse(torch.cat([F.interpolate(l,tgt,mode="bilinear",align_corners=False) for l in lats],1))
        return F.interpolate(s.cls(x),(out_hw,out_hw),mode="bilinear",align_corners=False)

def _lovasz_grad(g):
    p=len(g); gts=g.sum(); inter=gts-g.float().cumsum(0); union=gts+(1-g).float().cumsum(0)
    jac=1.-inter/union
    if p>1: jac[1:p]=jac[1:p]-jac[0:-1]
    return jac
def lovasz(logits,labels,ignore=255):
    pr=logits.softmax(1); B,C,H,W=pr.shape
    pr=pr.permute(0,2,3,1).reshape(-1,C); lb=labels.reshape(-1); m=lb!=ignore
    pr,lb=pr[m],lb[m]
    if lb.numel()==0: return logits.sum()*0.
    L=[]
    for c in range(C):
        fg=(lb==c).float()
        if fg.sum()==0: continue
        err=(fg-pr[:,c]).abs(); es,pm=torch.sort(err,0,descending=True)
        L.append(torch.dot(es,_lovasz_grad(fg[pm])))
    return torch.stack(L).mean() if L else logits.sum()*0.
def dice(lg,t,ig=255,eps=1.):
    m=(t!=ig); tt=t.clone(); tt[~m]=0
    oh=F.one_hot(tt,lg.shape[1]).permute(0,3,1,2).float()*m.unsqueeze(1)
    p=lg.softmax(1)*m.unsqueeze(1); i=(p*oh).sum((0,2,3)); u=p.sum((0,2,3))+oh.sum((0,2,3))
    return 1-((2*i+eps)/(u+eps)).mean()

# %%
# --- encoders: multi-level features via forward hooks (works for ViT + Prithvi) -
def _find_blocks(enc):
    best=None
    for _,mod in enc.named_modules():
        if isinstance(mod,nn.ModuleList) and len(mod)>=8:
            if best is None or len(mod)>len(best): best=mod
    return best
def _to_map(t):
    if t.dim()==4: return t
    B,N,D=t.shape; hw=int(round(N**.5))
    if hw*hw!=N: t=t[:,1:,:]; N=t.shape[1]; hw=int(round(N**.5))   # drop cls token
    return t.transpose(1,2).reshape(B,D,hw,hw)

def make_encoder(name):
    """returns (enc, feats_fn, dims); feats_fn(x)->list of 4 [B,D,h,w] from 4 depths."""
    if name=="imagenet_vit":
        import timm
        enc=timm.create_model("vit_base_patch16_224",pretrained=True,in_chans=IN_BANDS,num_classes=0).to(dev)
        run=lambda x: enc.forward_features(x)
    else:
        from terratorch import BACKBONE_REGISTRY
        enc=BACKBONE_REGISTRY.build(name,pretrained=True,num_frames=1,
            bands=["BLUE","GREEN","RED","NIR_NARROW","SWIR_1","SWIR_2"]).to(dev)
        def run(x):
            try: return enc(x)
            except Exception: return enc(x.unsqueeze(2))
    blocks=_find_blocks(enc)
    cap={}
    if blocks is not None:
        idx=[int(round(f*(len(blocks)-1))) for f in (0.25,0.5,0.75,1.0)]
        def mk(j):
            def hook(m,i,o): cap[j]=o
            return hook
        for j,bi in enumerate(idx): blocks[bi].register_forward_hook(mk(j))
        def feats_fn(x):
            cap.clear(); _=run(x); return [_to_map(cap[j]) for j in range(4)]
    else:                                                  # fallback: single map replicated
        def feats_fn(x):
            f=run(x); f=f[-1] if isinstance(f,(list,tuple)) else f; f=_to_map(f); return [f,f,f,f]
    enc.eval()
    with torch.no_grad(): dims=[m.shape[1] for m in feats_fn(torch.zeros(1,IN_BANDS,224,224,device=dev))]
    return enc, feats_fn, dims

def build_optim(enc, dec, mode, base=1.5e-4, dcy=0.85):
    if mode=="frozen":
        for p in enc.parameters(): p.requires_grad=False
        return torch.optim.AdamW(dec.parameters(),lr=1e-3,weight_decay=1e-4)
    blocks=_find_blocks(enc); nb=len(blocks) if blocks else 1
    groups=[{"params":list(dec.parameters()),"lr":1e-3}]; seen=set()
    if blocks is not None:
        for i,blk in enumerate(blocks):
            ps=[p for p in blk.parameters() if p.requires_grad]
            for p in ps: seen.add(id(p))
            groups.append({"params":ps,"lr":base*(dcy**(nb-1-i))})
    rest=[p for p in enc.parameters() if p.requires_grad and id(p) not in seen]
    if rest: groups.append({"params":rest,"lr":base*(dcy**nb)})
    return torch.optim.AdamW(groups,weight_decay=1e-4)

@torch.no_grad()
def tta(fwd,x):                                            # average softmax over 4 flips
    ts=[lambda a:a, lambda a:torch.flip(a,[3]), lambda a:torch.flip(a,[2]), lambda a:torch.flip(a,[2,3])]
    out=0
    for t in ts: out=out+t(fwd(t(x))).softmax(1)
    return out/len(ts)

# %%
# --- train one (backbone, mode) ----------------------------------------------
def train_one(name, mode, seed=0, return_val_prob=False):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    enc, feats_fn, dims = make_encoder(name)
    dec = UPerHead(dims, K).to(dev)
    ft = (mode=="finetune")
    opt = build_optim(enc, dec, mode)
    bs = 4 if ft else 8; accum = 8//bs
    warm=torch.optim.lr_scheduler.LinearLR(opt,0.1,1.0,total_iters=WARMUP)
    cos=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS-WARMUP)
    sch=torch.optim.lr_scheduler.SequentialLR(opt,[warm,cos],milestones=[WARMUP])
    sc=torch.amp.GradScaler("cuda",enabled=(dev=="cuda")); ce=nn.CrossEntropyLoss(weight=CW.to(dev),ignore_index=255)
    trdl=torch.utils.data.DataLoader(DS(tr_chips,True),batch_size=bs,shuffle=True,num_workers=2,drop_last=True)
    vadl=torch.utils.data.DataLoader(DS(va_chips),batch_size=8,num_workers=2)
    def fwd(x):
        f=feats_fn(x) if ft else [t.detach() for t in feats_fn(x)]
        return dec(f)
    for ep in range(EPOCHS):
        enc.train() if ft else enc.eval(); dec.train(); opt.zero_grad()
        for j,(x,y) in enumerate(trdl):
            x,y=x.to(dev),y.to(dev)
            with torch.amp.autocast("cuda",enabled=(dev=="cuda")):
                if ft: lg=fwd(x)
                else:
                    with torch.no_grad(): f=[t.detach() for t in feats_fn(x)]
                    lg=dec(f)
                loss=(ce(lg,y)+dice(lg,y)+lovasz(lg,y))/accum
            sc.scale(loss).backward()
            if (j+1)%accum==0: sc.step(opt); sc.update(); opt.zero_grad()
        sch.step()
    enc.eval(); dec.eval(); PROB,LAB=[],[]
    with torch.no_grad():
        for x,y in vadl:
            prob=tta(lambda a: dec([t.detach() for t in feats_fn(a)]), x.to(dev))   # [B,K,224,224]
            y=y.numpy()
            for b in range(prob.shape[0]):                       # keep ONLY valid pixels (fp16)
                m=y[b].ravel()!=255
                PROB.append(prob[b].permute(1,2,0).reshape(-1,K).cpu().numpy()[m].astype(np.float16))
                LAB.append(y[b].ravel()[m].astype(np.uint8))
    pf=np.concatenate(PROB); lab=np.concatenate(LAB); P=pf.argmax(1)
    del enc,dec; gc.collect(); torch.cuda.empty_cache()
    r=dict(oa=accuracy_score(lab,P), mf1=f1_score(lab,P,average="macro",zero_division=0),
           miou=jaccard_score(lab,P,average="macro",zero_division=0))
    return (r, (pf if return_val_prob else None), (lab if return_val_prob else None))

# %%
# --- run the sweep + softmax ensemble ----------------------------------------
def _fmt(s): return f"{s/60:.1f}m" if s>=60 else f"{s:.0f}s"
t0=time.time(); res={}; ens_sum=None; y_true=None; n_ens=0
for name,mode in RUNS:
    ts=time.time()
    try:
        print(f"\n>>> {name} [{mode}] ...")
        r,pf,yt=train_one(name,mode,return_val_prob=True); res[f"{name}:{mode}"]=r
        if pf is not None:                                   # accumulate ensemble in place (fp32)
            ens_sum = pf.astype(np.float32) if ens_sum is None else ens_sum+pf.astype(np.float32)
            y_true=yt; n_ens+=1; del pf
        print(f"    OA {r['oa']:.3f}  macroF1 {r['mf1']:.3f}  mIoU {r['miou']:.3f}   "
              f"[{_fmt(time.time()-ts)} | elapsed {_fmt(time.time()-t0)}]")
        json.dump(res,open("seg_v2_results.json","w"),indent=2); gc.collect()
    except Exception as e:
        print(f"    [skip] {name}:{mode}: {str(e).splitlines()[-1][:90]}")
if n_ens>=2 and y_true is not None:                          # soft-vote ensemble
    ens=(ens_sum/n_ens).argmax(1)
    res["ENSEMBLE"]=dict(oa=accuracy_score(y_true,ens),
        mf1=f1_score(y_true,ens,average="macro",zero_division=0),
        miou=jaccard_score(y_true,ens,average="macro",zero_division=0))
    json.dump(res,open("seg_v2_results.json","w"),indent=2)

print(f"\n=== Prithvi-Seg v2 · T1 land cover ===  [total {_fmt(time.time()-t0)}]")
print(f"  {'model':<28} {'OA':>6} {'macroF1':>8} {'mIoU':>6}")
for n,r in sorted(res.items(),key=lambda x:-x[1]['mf1']):
    print(f"  {n:<28} {r['oa']:.3f}  {r['mf1']:.3f}  {r['miou']:.3f}")

# %% [markdown]
# ## Reading it
# - **vs v1**: Prithvi-300:finetune here vs v1's ~0.59 macro-F1 tells you what the v2
#   decoder + Lovász + LLRD + TTA bought (expect a lift; that's the "improvised model").
# - **frozen 600 vs 300**: encoder-quality ranking with the *strong* decoder.
# - **ENSEMBLE** row: the best deployable single number (mean softmax of all models).
# - `seg_v2_results.json` is checkpointed after every model, so partial runs persist.
