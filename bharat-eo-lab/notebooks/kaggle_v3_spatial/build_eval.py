import io, json, os
base=io.open("prithvi_seg_v3_spatial.py",encoding="utf-8").read()

def patch(s, flags_line, tag):
    def rep(old,new):
        nonlocal s
        assert old in s, old[:70]
        s=s.replace(old,new,1)
    rep("def __init__(s,ch,tr=False): s.ch,s.tr=ch,tr","def __init__(s,ch,tr=False,ev=False): s.ch,s.tr,s.ev=ch,tr,ev")
    rep("        if USE_CONF: y=np.where(aux[12]>=CONF_MIN, y, 255).astype(np.int64)   # low-agreement -> ignore",
        "        lowc=aux[12]<CONF_MIN                       # <2 label sources agree\n"
        "        if USE_CONF and not s.ev: y=np.where(lowc,255,y).astype(np.int64)   # mask TRAIN only\n"
        "        if s.ev: y=np.where(lowc&(y!=255),y+20,y).astype(np.int64)         # eval: keep every pixel, tag low-agreement (+20)")
    rep("vadl=torch.utils.data.DataLoader(DS(va),batch_size=8,num_workers=2)",
        "vadl=torch.utils.data.DataLoader(DS(va,ev=True),batch_size=8,num_workers=2)")
    i=s.index("    model.eval(); P,T=[],[]"); j=s.index("per_class_iou=[round(float(v),3) for v in iou])")+len("per_class_iou=[round(float(v),3) for v in iou])")
    NEW='''    model.eval(); P,T,C=[],[],[]
    with torch.no_grad():
        for x,soft,y in vadl:
            prob=tta(lambda a: model(a,False), x.to(dev))
            p=prob.argmax(1).cpu().numpy().ravel(); t=y.numpy().ravel(); k=t!=255
            lowc=(t>=20)&k; t=np.where(lowc,t-20,t)
            P.append(p[k]); T.append(t[k]); C.append(~lowc[k])
    P,T,C=np.concatenate(P),np.concatenate(T),np.concatenate(C)
    def _m(P_,T_):
        iou=jaccard_score(T_,P_,average=None,zero_division=0,labels=list(range(K)))
        return dict(oa=accuracy_score(T_,P_),mf1=f1_score(T_,P_,average="macro",zero_division=0),
                    miou=jaccard_score(T_,P_,average="macro",zero_division=0),
                    per_class_iou=[round(float(v),3) for v in iou],
                    px=np.bincount(T_,minlength=K).tolist())
    out=dict(all_px=_m(P,T), conf_px=_m(P[C],T[C]), frac_confident=float(C.mean()))'''
    s=s[:i]+NEW+s[j:]
    rep(s[s.index("USE_AUXSTEM, USE_CONF"):s.index("\n",s.index("USE_AUXSTEM, USE_CONF"))], flags_line)
    head,sep,_=s.partition("t0=time.time(); res={}"); assert sep
    DRV=f'''t0=time.time(); res={{}}
CKPT="seg_v3_eval_{tag}.json"
for sd in (0,1):
    try: r=run_v3(seed=sd); oom=False
    except torch.cuda.OutOfMemoryError: oom=True
    if oom: purge(); r=run_v3(seed=sd)
    res[f"s{{sd}}"]=r; json.dump(res,open(CKPT,"w"),indent=2); purge()
    print(f"seed {{sd}} done: conf-px mIoU {{r['conf_px']['miou']:.4f}} | all-px mIoU {{r['all_px']['miou']:.4f}} | confident frac {{r['frac_confident']:.3f}}",flush=True)
print("=== {tag}: both seeds done ===",flush=True)
'''
    out=head+DRV; compile(out,"x","exec"); return out

V3="USE_AUXSTEM, USE_CONF, USE_SOFT, USE_COPYPASTE, USE_BALANCED = True, True, True, True, False"
BASE="USE_AUXSTEM, USE_CONF, USE_SOFT, USE_COPYPASTE, USE_BALANCED = False, False, False, False, False"
meta=json.load(open("push/kernel-metadata.json"))
for tag,fl in (("full",V3),("base",BASE)):
    code=patch(base,fl,tag); d=f"push_eval_{tag}"; os.makedirs(d,exist_ok=True)
    fn=f"eval_{tag}.py"; io.open(f"{d}/{fn}","w",encoding="utf-8",newline="\n").write(code)
    m=dict(meta); m.update(id=f"crakashpreetham/bharat-eo-v3-eval-{tag}",title=f"bharat-eo-v3-eval-{tag}",code_file=fn)
    json.dump(m,open(f"{d}/kernel-metadata.json","w"),indent=2); print("staged",d)
