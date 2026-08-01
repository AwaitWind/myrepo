import glob, os, re, torch
import torch.nn.functional as F
KD = "/tmp/kd_hh"; MODE = "oracle_hh"; REF = "fullref"; KV = 512
def loadK(tag, L):
    cs = glob.glob(f"{KD}/{tag}_L{L}_c*.pt")
    if not cs: return None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()
def cerr(a, b):  # 1-cos mean
    return float((1 - F.cosine_similarity(a, b, dim=-1)).clamp(0, 2).mean())
# seg -> (start, end, delta at measure)  [warmup caches each doc at ~pos 64]
SEGS = {"SYS(d0)": (0,64,0), "C1(d0)": (64,1216,0), "C2(d1152)": (1216,2304,1152),
        "C3(d2240)": (2304,3392,2240), "Q(d0)": (3392,3456,0)}
print(f"{MODE} vs {REF}: per-seg  NOPE[:512] 1-cos | ROPE[512:] 1-cos | ||nope|| ||rope|| (mean norm)")
for L in [2, 20, 40]:
    o, f = loadK(MODE, L), loadK(REF, L)
    if o is None or f is None: continue
    n = min(o.shape[0], f.shape[0]); o, f = o[:n], f[:n]
    print(f"-- L{L} --")
    for name, (s, e, d) in SEGS.items():
        e = min(e, n)
        nope = cerr(o[s:e, :KV], f[s:e, :KV])
        rope = cerr(o[s:e, KV:], f[s:e, KV:])
        nnorm = float(o[s:e, :KV].norm(dim=-1).mean())
        rnorm = float(o[s:e, KV:].norm(dim=-1).mean())
        print(f"   {name:>9}: nope={nope:.5f}  rope={rope:.5f}  |  ||nope||={nnorm:6.2f} ||rope||={rnorm:6.2f}")
