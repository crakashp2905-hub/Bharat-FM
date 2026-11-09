"""Build the PROTOCOL-2 Kaggle kernels (supersedes build_spatial/eval/abl/bl).

What protocol 2 changes relative to the kernels that produced outputs/spatial_v3_results/:
  1. Dataset integrity + spatial-distance guard runs at kernel start (bench_integrity inlined).
  2. Copy-paste pool is derived from TRAIN-split label files only (the old pool read
     manifest['target'], which is -1 for every chip -> the pool was empty -> copy-paste was a no-op),
     and is asserted train-only.
  3. Train / val / test are scored SEPARATELY (plus a train subset for over-fit diagnostics and the
     pooled val+test number for comparability with protocol 1). No per-epoch model selection: the
     final epoch is evaluated; val is only *logged* every 5 epochs.
  4. Both pixel views (confident / all) for every split.
  5. Deterministic mode + seeded DataLoader workers; the run file records the environment.

Usage:  python build_protocol2.py          # stages push_p2_* directories
"""
import io, json, os, re, hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "prithvi_seg_v3_spatial.py")
BI = os.path.join(HERE, "..", "..", "src", "bench_integrity.py")
base = io.open(SRC, encoding="utf-8").read()
bi_src = io.open(BI, encoding="utf-8").read().replace("from __future__ import annotations\n", "")
compile(bi_src, "bench_integrity", "exec")

FLAGS = "USE_AUXSTEM, USE_CONF, USE_SOFT, USE_COPYPASTE, USE_BALANCED = {a}, {c}, {s}, {p}, False"
ARMS = {
    "full": FLAGS.format(a="True", c="True", s="True", p="True"),
    "base": FLAGS.format(a="False", c="False", s="False", p="False"),
}


def build(arm, seeds, tag):
    s = base

    def rep(old, new):
        nonlocal s
        assert s.count(old) == 1, f"patch target not unique/missing: {old[:80]!r} (n={s.count(old)})"
        s = s.replace(old, new, 1)

    # --- 0. determinism (before the first CUDA op) -------------------------------------------------
    rep('os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"   # cut allocator fragmentation',
        'os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"   # cut allocator fragmentation\n'
        'os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"')
    rep('dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev)',
        'dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev)\n'
        'torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False\n'
        'torch.use_deterministic_algorithms(True, warn_only=True)\n'
        f'PROTOCOL="p2: train-only label-derived rare pool; val/test scored separately; final-epoch; deterministic"\n'
        f'ARM_TAG={tag!r}; SEEDS={tuple(seeds)!r}')
    # --- 1. flags -----------------------------------------------------------------------------------
    old_flags = re.search(r"USE_AUXSTEM, USE_CONF, USE_SOFT, USE_COPYPASTE, USE_BALANCED = .*", s).group(0)
    rep(old_flags, ARMS[arm])
    # --- 2. integrity guard (inlined module) right after the dataset is located ---------------------
    marker = 'print("root",ROOT,"| n=",len(imgs),"| splits",man.split.value_counts().to_dict())'
    rep(marker, marker + "\n# ===== inlined: src/bench_integrity.py =====\n" + bi_src +
        "\n# ===== end inlined =====\n"
        "_integ=require_clean(ROOT,sample_files=200)\n"
        'print("integrity OK:",{k:len(v) for k,v in _integ.items()},"| min train->val %.1f km, train->test %.1f km" % '
        "(min_distance_km(man,'train','val'),min_distance_km(man,'train','test')),flush=True)")
    # --- 3. explicit train / val / test chip lists ------------------------------------------------------
    rep('tr=man.loc[man.split=="train","chip"].tolist(); va=man.loc[man.split.isin(["val","test"]),"chip"].tolist()',
        'tr=man.loc[man.split=="train","chip"].tolist()\n'
        'va_chips=man.loc[man.split=="val","chip"].tolist(); te_chips=man.loc[man.split=="test","chip"].tolist()\n'
        'assert not (set(tr)&set(va_chips)) and not (set(tr)&set(te_chips)) and not (set(va_chips)&set(te_chips))\n'
        'tr_eval=sorted(np.random.RandomState(0).choice(tr,min(150,len(tr)),replace=False).tolist())   # fixed train subset (over-fit diagnostic)\n'
        'va=va_chips+te_chips')
    # --- 4. rare pool from TRAIN labels, asserted train-only ----------------------------------------
    rep('CP_POOL=man.loc[man.target.isin(RARE_CP),"chip"].tolist() if "target" in man.columns else []',
        'POOL=build_rare_pool(ROOT,man,RARE_CP,min_px=200,split="train"); assert_pool_train_only(list(POOL),man)\n'
        'POOL_BY_CLS={k:[c for c,v in POOL.items() if k in v] for k in RARE_CP}; POOL_BY_CLS={k:v for k,v in POOL_BY_CLS.items() if v}\n'
        'POOL_CLS=sorted(POOL_BY_CLS); CP_POOL=list(POOL)\n'
        'print("rare pool (train only):",{CLS_NAMES_[k]:len(v) for k,v in POOL_BY_CLS.items()},"| unserved:",check_pool_covers_rare(POOL,None),flush=True)\n'
        'if USE_COPYPASTE: assert CP_POOL, "copy-paste is ON but the train-only pool is empty (silent no-op)"')
    s = s.replace("CLS_NAMES_", "CLASS_NAMES")
    # --- 5. dataset: eval mode (all pixels kept, low-agreement tagged) + label-derived paste -----------
    rep("def __init__(s,ch,tr=False): s.ch,s.tr=ch,tr", "def __init__(s,ch,tr=False,ev=False): s.ch,s.tr,s.ev=ch,tr,ev")
    rep("        if USE_CONF: y=np.where(aux[12]>=CONF_MIN, y, 255).astype(np.int64)   # low-agreement -> ignore",
        "        lowc=aux[12]<CONF_MIN                       # <2 label sources agree\n"
        "        if USE_CONF and not s.ev: y=np.where(lowc,255,y).astype(np.int64)   # mask TRAIN only\n"
        "        if s.ev: y=np.where(lowc&(y!=255),y+20,y).astype(np.int64)         # eval: keep every pixel, tag low-agreement (+20)")
    rep("            sc=random.choice(CP_POOL); sx=(load_img(sc)-MU)/SD; sl=load_lbl(sc)\n            m=np.isin(sl,RARE_CP)",
        "            k_=random.choice(POOL_CLS); sc=random.choice(POOL_BY_CLS[k_]); sx=(load_img(sc)-MU)/SD; sl=load_lbl(sc)\n"
        "            m=np.isin(sl,POOL_CLS)")
    rep("if s.tr and USE_COPYPASTE and CP_POOL and random.random()<P_CP:", "if s.tr and USE_COPYPASTE and random.random()<P_CP:")
    # --- 6. seeded workers + separate evaluation --------------------------------------------------------
    rep("trdl=torch.utils.data.DataLoader(DS(tr,True),batch_size=bs,sampler=smp,shuffle=(smp is None),num_workers=2,drop_last=True)",
        "def _winit(w): random.seed(seed*1000+w); np.random.seed(seed*1000+w)\n"
        "    _g=torch.Generator(); _g.manual_seed(seed)\n"
        "    trdl=torch.utils.data.DataLoader(DS(tr,True),batch_size=bs,sampler=smp,shuffle=(smp is None),num_workers=2,drop_last=True,worker_init_fn=_winit,generator=_g)")
    rep("    vadl=torch.utils.data.DataLoader(DS(va),batch_size=8,num_workers=2)\n",
        "    def _dl(ch): return torch.utils.data.DataLoader(DS(ch,ev=True),batch_size=8,num_workers=2)\n"
        "    vadl=_dl(va_chips)                                         # monitoring only; never used to pick a checkpoint\n")
    # periodic val monitor (no TTA, log only) -> goes after sch.step() in the epoch loop
    rep("        sch.step()\n    model.eval(); P,T=[],[]",
        "        sch.step()\n"
        "        if (ep+1)%5==0 or ep==EPOCHS-1:\n"
        "            model.eval(); _m=_collect(model,vadl,tta_on=False); model.train() if ft else None\n"
        "            print(f'    ep {ep+1:>2}/{EPOCHS} val(conf-px mIoU, no TTA) {_score(*_m)[\"conf_px\"][\"miou\"]:.4f}',flush=True)\n"
        "    model.eval(); P,T=[],[]")
    i = s.index("    model.eval(); P,T=[],[]")
    j = s.index("per_class_iou=[round(float(v),3) for v in iou])") + len("per_class_iou=[round(float(v),3) for v in iou])")
    NEW = '''    model.eval(); out={}
    cols={"train":_collect(model,_dl(tr_eval)),"val":_collect(model,_dl(va_chips)),"test":_collect(model,_dl(te_chips))}
    for nm,(P_,T_,C_) in cols.items(): out[nm]=_score(P_,T_,C_)
    P_,T_,C_=[np.concatenate([cols["val"][i_],cols["test"][i_]]) for i_ in range(3)]
    out["val+test"]=_score(P_,T_,C_)                               # protocol-1 comparable pooled number'''
    s = s[:i] + NEW + s[j:]
    # helpers placed before run_v3
    HELP = '''
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

'''
    rep("# --- train v3 ----------------------------------------------------------------\ndef run_v3(",
        "# --- train v3 ----------------------------------------------------------------" + HELP + "def run_v3(")
    # --- 7. driver --------------------------------------------------------------------------------------
    head, sep, _ = s.partition("t0=time.time(); res={}")
    assert sep
    DRV = f'''t0=time.time(); res={{}}
import hashlib, platform
def _md5(p):
    try: return hashlib.md5(open(p,"rb").read()).hexdigest()
    except Exception: return None
META=dict(protocol=PROTOCOL,arm=ARM_TAG,seeds=list(SEEDS),flags=dict(auxstem=USE_AUXSTEM,conf=USE_CONF,soft=USE_SOFT,copypaste=USE_COPYPASTE,balanced=USE_BALANCED),
          epochs=EPOCHS,manifest_md5=_md5(os.path.join(ROOT,"manifest.csv")),split_type=_st,
          n_train=len(tr),n_val=len(va_chips),n_test=len(te_chips),pool={{CLASS_NAMES[k]:len(v) for k,v in POOL_BY_CLS.items()}},
          torch=torch.__version__,python=platform.python_version(),gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
try:
    import terratorch; META["terratorch"]=getattr(terratorch,"__version__",None)
except Exception: pass
CKPT="seg_p2_{tag}.json"
for _i,sd in enumerate(SEEDS):
    try: r=run_v3(seed=sd); oom=False
    except torch.cuda.OutOfMemoryError: oom=True
    if oom: purge(); r=run_v3(seed=sd)
    res[f"s{{sd}}#{{_i}}"]=r; json.dump(dict(meta=META,runs=res),open(CKPT,"w"),indent=2); purge()
    print(f"seed {{sd}} (#{{_i}}) done | TRAIN conf mIoU {{r['train']['conf_px']['miou']:.4f}} | VAL {{r['val']['conf_px']['miou']:.4f}}/{{r['val']['all_px']['miou']:.4f}} | TEST {{r['test']['conf_px']['miou']:.4f}}/{{r['test']['all_px']['miou']:.4f}} (conf/all)",flush=True)
print("=== {tag}: all runs done ===",flush=True)
'''
    out = head + DRV
    compile(out, "kernel", "exec")
    return out


KERNELS = {
    "p2-full": ("full", (0, 1, 2)),
    "p2-base": ("base", (0, 1, 2)),
    "p2-repro": ("base", (0, 0)),       # same config, same seed, twice, in a fresh kernel: measures the noise floor
}

if __name__ == "__main__":
    meta = json.load(io.open(os.path.join(HERE, "push", "kernel-metadata.json"), encoding="utf-8"))
    for tag, (arm, seeds) in KERNELS.items():
        code = build(arm, seeds, tag)
        d = os.path.join(HERE, "push_" + tag.replace("-", "_")); os.makedirs(d, exist_ok=True)
        fn = f"{tag.replace('-', '_')}.py"
        io.open(os.path.join(d, fn), "w", encoding="utf-8", newline="\n").write(code)
        m = dict(meta); m.update(id=f"crakashpreetham/bharat-eo-{tag}", title=f"bharat-eo-{tag}", code_file=fn)
        json.dump(m, io.open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8"), indent=2)
        print("staged", d, "md5", hashlib.md5(code.encode()).hexdigest()[:10])
