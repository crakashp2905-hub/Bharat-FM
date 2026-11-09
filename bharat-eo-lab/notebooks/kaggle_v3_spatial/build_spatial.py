import io, re, json
SRC="notebookd8a73acf15.py"; DST="prithvi_seg_v3_spatial.py"
s=io.open(SRC,encoding="utf-8").read()

def rep(old,new):
    global s
    assert old in s, old[:60]
    s=s.replace(old,new,1)

rep("USE_AUXSTEM, USE_CONF, USE_SOFT, USE_COPYPASTE, USE_BALANCED = True, True, True, True, True",
    "# best config: 3/3 seeds showed the class-balanced sampler HURTS (mean dmIoU +0.0111)\n"
    "USE_AUXSTEM, USE_CONF, USE_SOFT, USE_COPYPASTE, USE_BALANCED = True, True, True, True, False")
rep('man=pd.read_csv(os.path.join(ROOT,"manifest.csv"))\nif "split" not in man.columns: man["split"]="train"',
    'man=pd.read_csv(os.path.join(ROOT,"manifest.csv"))\nif "split" not in man.columns: man["split"]="train"\n'
    '_st=str(man["split_type"].iloc[0]) if "split_type" in man.columns else "MISSING"\n'
    'assert _st.startswith("spatial_block"), f"refusing to run: split_type={_st!r}; attach bharat-eo-chips-dsv2-sp"\n'
    'print("split_type OK:",_st,flush=True)')

head,sep,_=s.partition("t0=time.time(); res={}")
assert sep
DRIVER='''t0=time.time(); res={}
def _fmt(s): return f"{s/60:.1f}m" if s>=60 else f"{s:.0f}s"
CLS=["tree","shrub","grass","crop","built","bare","snow","water","wetland","mangrove","moss"]
try:
    r=run_v3(seed=SEED); oom=False
except torch.cuda.OutOfMemoryError:
    oom=True
if oom:
    purge(); print("[oom] purged, retrying",flush=True); r=run_v3(seed=SEED)
json.dump({"best":r}, open("seg_v3_spatial.json","w"), indent=2)
sup=np.zeros(K,dtype=np.int64)
for c in va:
    l=load_lbl(c); v=l[l!=255]
    if v.size: sup+=np.bincount(v,minlength=K)
share=sup/max(int(sup.sum()),1); ok=[i for i in range(K) if share[i]>=0.001]
print(); print(f"=== Prithvi-Seg v3 on LEAKAGE-FREE spatial split === [{_fmt(time.time()-t0)}]")
print(f"  split_type {_st} | train {int((man.split=='train').sum())} val {int((man.split=='val').sum())} test {int((man.split=='test').sum())}")
print(f"  OA {r['oa']:.4f}  macroF1(11) {r['mf1']:.4f}  mIoU(11) {r['miou']:.4f}")
pc=r.get("per_class_iou")
if pc:
    for i in range(K): print(f"  {CLS[i]:<9} IoU {pc[i]:.4f}  px {int(sup[i]):>10,} {share[i]:6.2%}  {'ok' if i in ok else 'TOO FEW'}")
    print(f"  mIoU over {len(ok)} supported classes: {np.mean([pc[i] for i in ok]):.4f}  (excluded {[CLS[i] for i in range(K) if i not in ok]})")
print("  NOT comparable to the dsv2 ablation (chip-level split). A lower score here is expected and honest.")
'''
out=head+DRIVER
io.open(DST,"w",encoding="utf-8",newline="\n").write(out)
compile(out,"spatial","exec"); print("compile OK; magic lines:",sum(1 for l in out.split("\n") if l.lstrip()[:1] in "!%" and l.strip()))
meta=json.load(open("kernel-metadata.json"))
meta.update(id="crakashpreetham/bharat-eo-v3-spatial",title="bharat-eo-v3-spatial",code_file=DST,
            dataset_sources=["crakashpreetham/bharat-eo-chips-dsv2-sp"],enable_gpu=True,enable_internet=True)
meta.pop("id_no",None)
import os; os.makedirs("push",exist_ok=True)
io.open("push/"+DST,"w",encoding="utf-8",newline="\n").write(out)
json.dump(meta,open("push/kernel-metadata.json","w"),indent=2); print("staged push/")
