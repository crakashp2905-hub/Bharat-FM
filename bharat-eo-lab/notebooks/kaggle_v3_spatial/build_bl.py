import io, json, os
src=io.open("build_eval.py",encoding="utf-8").read().split('V3="USE_AUXSTEM')[0]
exec(src)
OFF="USE_AUXSTEM, USE_CONF, USE_SOFT, USE_COPYPASTE, USE_BALANCED = False, False, False, False, False"
meta=json.load(open("push/kernel-metadata.json"))
for tag,pre,mode in (("randinit",False,"finetune"),("frozen",True,"frozen")):
    code=patch(base,OFF,"x")
    assert "pretrained=True" in code
    code=code.replace("pretrained=True","pretrained=PRETRAINED",1)
    head=code.partition("t0=time.time(); res={}")[0]
    head=head.replace("K, EPOCHS, WARMUP, SEED, MODE = 11, 35, 3, 0,",f"PRETRAINED={pre}\nK, EPOCHS, WARMUP, SEED, MODE = 11, 35, 3, 0,",1)
    assert f"PRETRAINED={pre}" in head and f'"finetune"' in head
    head=head.replace('MODE = 11, 35, 3, 0, "finetune"','MODE = 11, 35, 3, 0, "finetune"') 
    drv=f'''t0=time.time(); res={{}}
MODE="{mode}"; print("arm {tag}: pretrained=",PRETRAINED,"mode=",MODE,flush=True)
CKPT="seg_v3_bl_{tag}.json"
for sd in (0,1):
    try: r=run_v3(seed=sd); oom=False
    except torch.cuda.OutOfMemoryError: oom=True
    if oom: purge(); r=run_v3(seed=sd)
    res[f"s{{sd}}"]=r; json.dump(res,open(CKPT,"w"),indent=2); purge()
    print(f"seed {{sd}}: conf-px mIoU {{r['conf_px']['miou']:.4f}} all-px mIoU {{r['all_px']['miou']:.4f}}",flush=True)
print("=== {tag} done ===",flush=True)
'''
    out=head+drv; compile(out,"x","exec")
    d=f"push_bl_{tag}"; os.makedirs(d,exist_ok=True); fn=f"bl_{tag}.py"
    io.open(f"{d}/{fn}","w",encoding="utf-8",newline="\n").write(out)
    m=dict(meta); m.update(id=f"crakashpreetham/bharat-eo-v3-bl-{tag}",title=f"bharat-eo-v3-bl-{tag}",code_file=fn)
    json.dump(m,open(f"{d}/kernel-metadata.json","w"),indent=2); print("staged",d)
