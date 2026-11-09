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

# %%
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

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"   # cut allocator fragmentation
import glob, random, json, gc, sys, time, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, jaccard_score, accuracy_score
dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev)

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
USE_AUXSTEM, USE_CONF, USE_SOFT, USE_COPYPASTE, USE_BALANCED = True, True, True, True, True
LAM_SOFT, CONF_MIN, P_CP = 0.3, 2, 0.4

# %%
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
print("root",ROOT,"| n=",len(imgs),"| splits",man.split.value_counts().to_dict())
def load_img(c): return np.transpose(np.load(f"{ROOT}/img/{c}.npy").astype(np.float32),(2,0,1))       # [17,H,W]
# `aux` is a reserved name on Windows (os.walk skips it silently), so local exports
# use `auxb`; datasets already on Kaggle still carry `aux`. Resolve either.
AUXD="auxb" if os.path.isdir(f"{ROOT}/auxb") else "aux"
def load_aux(c): return np.transpose(np.load(f"{ROOT}/{AUXD}/{c}.npy").astype(np.float32),(2,0,1))   # [14,H,W]
def load_lbl(c): return np.load(f"{ROOT}/lbl/{c}.npy").astype(np.uint8)
CIN=len(load_img(man.chip.iloc[0])); print("input channels:",CIN,"| Prithvi",len(PRITHVI_IDX),"| aux",len(AUX_IDX))
tr=man.loc[man.split=="train","chip"].tolist(); va=man.loc[man.split.isin(["val","test"]),"chip"].tolist()
st=np.stack([load_img(c) for c in tr[:200]]); MU=st.mean((0,2,3),keepdims=True)[0]; SD=st.std((0,2,3),keepdims=True)[0]+1e-6
cnt=np.zeros(K)
for c in tr:
    v=load_lbl(c).ravel(); v=v[v!=255]; cnt+=np.bincount(v,minlength=K)
CW=torch.tensor(np.clip(cnt.sum()/(K*np.clip(cnt,1,None)),0.5,8.0).astype(np.float32))
CP_POOL=man.loc[man.target.isin(RARE_CP),"chip"].tolist() if "target" in man.columns else []
print("copy-paste rare pool:",len(CP_POOL),"chips | class pixel counts:",dict(enumerate(cnt.astype(int))))

def soft_from_aux(aux):                                       # aux[2:11]=dwp0..8 -> 11-class soft target
    dwp=aux[2:11]; soft=np.zeros((K,)+dwp.shape[1:],np.float32)
    for i,c in enumerate(DW_TO_IDX): soft[c]+=dwp[i]
    return soft/(soft.sum(0,keepdims=True)+1e-6)

class DS(torch.utils.data.Dataset):
    def __init__(s,ch,tr=False): s.ch,s.tr=ch,tr
    def __len__(s): return len(s.ch)
    def __getitem__(s,i):
        c=s.ch[i]; raw=load_img(c); x=(raw-MU)/SD; aux=load_aux(c); y=load_lbl(c).astype(np.int64)
        soft=soft_from_aux(aux) if USE_SOFT else np.zeros((K,)+y.shape,np.float32)
        if USE_CONF: y=np.where(aux[12]>=CONF_MIN, y, 255).astype(np.int64)   # low-agreement -> ignore
        if s.tr and USE_COPYPASTE and CP_POOL and random.random()<P_CP:
            sc=random.choice(CP_POOL); sx=(load_img(sc)-MU)/SD; sl=load_lbl(sc)
            m=np.isin(sl,RARE_CP)
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

# %%
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

# %%
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

# %%
# --- train v3 ----------------------------------------------------------------
def run_v3(name="prithvi_eo_v2_300",seed=SEED):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    enc,feats_fn,dims=make_encoder(name); model=V3(enc,feats_fn,dims).to(dev)
    ft=(MODE=="finetune"); opt=build_optim(enc,model,MODE); bs=BS_FT if ft else 8; accum=max(1,8//bs)   # effective batch stays ~8 via accumulation
    warm=torch.optim.lr_scheduler.LinearLR(opt,0.1,1.0,total_iters=WARMUP)
    cos=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS-WARMUP)
    sch=torch.optim.lr_scheduler.SequentialLR(opt,[warm,cos],milestones=[WARMUP])
    sc=torch.amp.GradScaler("cuda",enabled=(dev=="cuda")); ce=nn.CrossEntropyLoss(weight=CW.to(dev),ignore_index=255)
    smp=make_sampler()
    trdl=torch.utils.data.DataLoader(DS(tr,True),batch_size=bs,sampler=smp,shuffle=(smp is None),num_workers=2,drop_last=True)
    vadl=torch.utils.data.DataLoader(DS(va),batch_size=8,num_workers=2)
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
    model.eval(); P,T=[],[]
    with torch.no_grad():
        for x,soft,y in vadl:
            prob=tta(lambda a: model(a,False), x.to(dev))
            p=prob.argmax(1).cpu().numpy().ravel(); t=y.numpy().ravel(); k=t!=255; P.append(p[k]); T.append(t[k])
    P,T=np.concatenate(P),np.concatenate(T)
    iou=jaccard_score(T,P,average=None,zero_division=0,labels=list(range(K)))
    out=dict(oa=accuracy_score(T,P),mf1=f1_score(T,P,average="macro",zero_division=0),
             miou=jaccard_score(T,P,average="macro",zero_division=0),per_class_iou=[round(float(v),3) for v in iou])
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
def _fmt(s): return f"{s/60:.1f}m" if s>=60 else f"{s:.0f}s"
CLS=["tree","shrub","grass","crop","built","bare","snow","water","wetland","mangrove","moss"]

# The single-seed ablation found -balanced BEATING full (mIoU 0.619 -> 0.628), which
# inverts the design assumption. +0.009 sits inside plausible seed noise, so re-run
# both arms on further seeds and read the PAIRED per-seed delta, not the absolutes.
ARMS = {
    "full":      dict(USE_AUXSTEM=1, USE_CONF=1, USE_SOFT=1, USE_COPYPASTE=1, USE_BALANCED=1),
    "-balanced": dict(USE_AUXSTEM=1, USE_CONF=1, USE_SOFT=1, USE_COPYPASTE=1, USE_BALANCED=0),
}
SEEDS = [1, 2]

# seed 0 comes from the completed ablation run (notebookeae3a6dfc5)
SEED0 = {"full": dict(oa=0.872, mf1=0.738, miou=0.619),
         "-balanced": dict(oa=0.8783444865920942, mf1=0.7442068624718025, miou=0.6276521333685366)}
res["s0"] = SEED0

CKPT = "seg_v3_seedcheck.json"
for p in (CKPT, f"/kaggle/working/{CKPT}"):
    if os.path.exists(p):
        try: res.update(json.load(open(p)))          # resume a partial run
        except Exception: pass

todo = [(s, a) for s in SEEDS for a in ARMS if f"s{s}" not in res or a not in res.get(f"s{s}", {})]
print(f"to run: {[(f's{s}', a) for s, a in todo]}", flush=True)

for sd, arm in todo:
    for k, v in ARMS[arm].items(): globals()[k] = bool(v)   # flags are module globals read by DS/V3
    ts = time.time()
    try:
        r = run_v3(seed=sd); oom = False
    except torch.cuda.OutOfMemoryError:              # never retry inside the handler: the live
        oom = True                                   # traceback pins the failed model on the GPU
    if oom:
        purge()
        print(f"  [oom] s{sd}/{arm}: purged, retrying ...", flush=True)
        r = run_v3(seed=sd)
    res.setdefault(f"s{sd}", {})[arm] = r
    print(f"  s{sd} {arm:<10} OA {r['oa']:.3f}  macroF1 {r['mf1']:.3f}  mIoU {r['miou']:.3f}   [{_fmt(time.time()-ts)}]", flush=True)
    json.dump(res, open(CKPT, "w"), indent=2)        # checkpoint after every run
    purge()

print()
print(f"=== USE_BALANCED seed check ===  [total {_fmt(time.time()-t0)}]")
print(f"  {'seed':<6} {'full mIoU':>10} {'-bal mIoU':>10} {'delta':>8}   {'full MF1':>9} {'-bal MF1':>9} {'delta':>8}")
dm, df1 = [], []
for k in sorted(res):
    g = res[k]
    if "full" not in g or "-balanced" not in g: continue
    a, b = g["full"], g["-balanced"]
    d, e = b["miou"] - a["miou"], b["mf1"] - a["mf1"]
    dm.append(d); df1.append(e)
    print(f"  {k:<6} {a['miou']:>10.4f} {b['miou']:>10.4f} {d:>+8.4f}   {a['mf1']:>9.4f} {b['mf1']:>9.4f} {e:>+8.4f}")

if dm:
    n = len(dm)
    mu = sum(dm)/n
    sd_ = (sum((x-mu)**2 for x in dm)/(n-1))**0.5 if n > 1 else float("nan")
    mu1 = sum(df1)/n
    print()
    print(f"  mean delta mIoU {mu:+.4f}" + (f"  sd {sd_:.4f}  (n={n})" if n > 1 else f"  (n={n})"))
    print(f"  mean delta MF1  {mu1:+.4f}")
    print(f"  sign agreement: {sum(1 for x in dm if x > 0)}/{n} seeds favour turning the sampler OFF")
    print()
    if n > 1 and all(x > 0 for x in dm) and mu > sd_:
        print("  READ: consistent across seeds and mean exceeds the spread ->")
        print("        USE_BALANCED is genuinely HURTING; drop the class-balanced sampler.")
    elif n > 1 and not all(x > 0 for x in dm):
        print("  READ: sign flips across seeds -> the seed-0 result was noise.")
        print("        Keep USE_BALANCED; the single-seed finding does not replicate.")
    else:
        print("  READ: direction consistent but spread is comparable to the effect ->")
        print("        suggestive, not conclusive. Add seeds before claiming it in a paper.")

# %% [markdown]
# ## Reading it
# - This is **not** an ablation sweep — it re-runs only `full` and `-balanced`, on extra
#   seeds, to test whether one surprising single-seed result replicates.
# - The number that matters is the **paired per-seed delta** (`-balanced` minus `full` on the
#   *same* seed), not either absolute score. Absolutes move with seed; the paired delta
#   cancels most of that.
# - `sign agreement` is the honest headline: if the delta flips sign across seeds, the
#   seed-0 finding was noise regardless of how large the mean looks.
# - Two extra seeds is the minimum to distinguish "real" from "noise", and it still cannot
#   give a real confidence interval. Treat a positive result as grounds to drop the sampler,
#   not as a significance claim.
