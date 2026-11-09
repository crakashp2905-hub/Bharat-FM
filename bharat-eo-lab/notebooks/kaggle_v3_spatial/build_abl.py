import io, json, os
src=io.open("build_eval.py",encoding="utf-8").read().split('V3="USE_AUXSTEM')[0]
exec(src)                      # defines patch() and base
V3="USE_AUXSTEM, USE_CONF, USE_SOFT, USE_COPYPASTE, USE_BALANCED = True, True, True, True, False"
ALL=dict(USE_AUXSTEM=1,USE_CONF=1,USE_SOFT=1,USE_COPYPASTE=1,USE_BALANCED=0)
ARMS={"-auxstem":{**ALL,"USE_AUXSTEM":0},"-conf":{**ALL,"USE_CONF":0},
      "-soft":{**ALL,"USE_SOFT":0},"-copypaste":{**ALL,"USE_COPYPASTE":0}}
meta=json.load(open("push/kernel-metadata.json"))
for tag,mine in (("a",["-auxstem","-conf"]),("b",["-soft","-copypaste"])):
    full=patch(base,V3,"x")
    head=full.partition("t0=time.time(); res={}")[0]
    drv=f'''t0=time.time(); res={{}}
ARMS={ARMS!r}
CKPT="seg_v3_abl_{tag}.json"
for arm in {mine!r}:
    for k,v in ARMS[arm].items(): globals()[k]=bool(v)
    for sd in (0,1):
        try: r=run_v3(seed=sd); oom=False
        except torch.cuda.OutOfMemoryError: oom=True
        if oom: purge(); r=run_v3(seed=sd)
        res[f"{{arm}}|s{{sd}}"]=r; json.dump(res,open(CKPT,"w"),indent=2); purge()
        print(f"{{arm}} s{{sd}}: conf-px mIoU {{r['conf_px']['miou']:.4f}} all-px mIoU {{r['all_px']['miou']:.4f}}",flush=True)
print("=== abl {tag} done ===",flush=True)
'''
    code=head+drv; compile(code,"x","exec")
    d=f"push_abl_{tag}"; os.makedirs(d,exist_ok=True); fn=f"abl_{tag}.py"
    io.open(f"{d}/{fn}","w",encoding="utf-8",newline="\n").write(code)
    m=dict(meta); m.update(id=f"crakashpreetham/bharat-eo-v3-abl-{tag}",title=f"bharat-eo-v3-abl-{tag}",code_file=fn)
    json.dump(m,open(f"{d}/kernel-metadata.json","w"),indent=2); print("staged",d)
