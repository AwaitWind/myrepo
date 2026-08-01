import glob, re, os, torch
KD = "/tmp/kd_imp2"
KV_LORA = 512  # nope dims; rest = rope pe
def loadimp(tag, L):
    f = KD + "/" + tag + "_IMP_L" + str(L) + ".pt"
    return set(int(x) for x in torch.load(f, map_location="cpu").tolist()) if os.path.exists(f) else None
def loadK(tag, L):
    cs = glob.glob(KD + "/" + tag + "_L" + str(L) + "_c*.pt")
    if not cs:
        f = KD + "/" + tag + "_L" + str(L) + ".pt"
        return torch.load(f, map_location="cpu").flatten(1).float() if os.path.exists(f) else None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()
def cvec(a, b):
    n = min(a.shape[0], b.shape[0]); return torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=-1)
ip = loadimp("pic_a3", 1); io = loadimp("pic_a3_oracle", 1)
SEGS = {"SYS(d0)": (0,64), "C1(d0)": (64,1216), "C2(d1152)": (1216,2304), "C3(d2240)": (2304,3392)}
print("oracle-vs-pic_a3  1-cos, split NOPE[:512] vs PE[512:]  (non-imp-both)   dim=%d" % (loadK("pic_a3",2).shape[1]))
for L in [2, 12, 19]:
    pa = loadK("pic_a3", L); oa = loadK("pic_a3_oracle", L)
    print("-- L%d --" % L)
    for s,(a,b) in SEGS.items():
        idx = [p for p in range(a, min(b, pa.shape[0])) if p not in ip and p not in io]
        if not idx: continue
        t = torch.tensor(idx)
        nope = float((1 - cvec(oa[:, :KV_LORA], pa[:, :KV_LORA])[t]).clamp(0,2).mean())
        pe   = float((1 - cvec(oa[:, KV_LORA:], pa[:, KV_LORA:])[t]).clamp(0,2).mean())
        print("   %-10s nope=%.6f  pe=%.6f" % (s, nope, pe))
