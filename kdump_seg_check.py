import glob, re, os, torch
KD = "/tmp/kd_imp2"
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
def cosvec(a, b):
    n = min(a.shape[0], b.shape[0])
    return torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=-1)
SEGS = {"SYS": (0,64), "C1": (64,1216), "C2": (1216,2304), "C3": (2304,3392), "Q": (3392,3456)}
ip = loadimp("pic_a3", 1); io = loadimp("pic_a3_oracle", 1)
print("per-SEGMENT mean(1-cos) oracle-vs-pic_a3, restricted to NON-imp-in-both positions")
print("  (SYS = published by SYS-prewarm via NORMAL writeback in BOTH modes → should ~0)")
print("  (C1/C2/C3 = published during pic_wN where SYS is a hit → KEEPALIVE publish for oracle)")
hdr = "  L  | " + " | ".join(s.rjust(8) for s in SEGS)
print(hdr)
for L in [2, 8, 12, 19, 40]:
    pa = loadK("pic_a3", L); oa = loadK("pic_a3_oracle", L)
    if pa is None or oa is None: continue
    c = cosvec(oa, pa)
    row = []
    for s,(a,b) in SEGS.items():
        idx = [p for p in range(a, min(b, c.shape[0])) if p not in ip and p not in io]
        if not idx:
            row.append(float("nan")); continue
        row.append(float((1 - c[torch.tensor(idx)]).clamp(0,2).mean()))
    print("%4d | " % L + " | ".join(("%.5f" % v).rjust(8) for v in row))
