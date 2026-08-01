import glob, os, re, torch
import torch.nn.functional as F
KD = "/tmp/kd_hh"; MODE = "oracle_hh"; REF = "fullref"
def loadK(tag, L):
    cs = glob.glob(f"{KD}/{tag}_L{L}_c*.pt")
    if not cs: return None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()
def loadQ(tag, L):
    f = f"{KD}/{tag}_Q_L{L}.pt"
    if not os.path.exists(f): return None
    d = torch.load(f, map_location="cpu"); return d["q"].float(), float(d["scale"]), d["qpos"].long()
def attn_mass(Q, K):
    q, scale, qpos = Q; n = K.shape[0]
    score = (q @ K.T) * scale
    kidx = torch.arange(n)
    score = score.masked_fill(kidx[None, :] > qpos[:, None], float("-inf"))
    return F.softmax(score, dim=1).sum(0)
SEGS = {"SYS": (0,64), "C1": (64,1216), "C2": (1216,2304), "C3": (2304,3392), "Q": (3392,3456)}
print(f"{MODE} vs {REF}: per-segment RAW K error (1-cos) | attn-mass%% | weighted(a*rawerr)")
for L in [2, 20, 40, 60, 77]:
    Km, Kf, Q = loadK(MODE, L), loadK(REF, L), loadQ(MODE, L)
    if Km is None or Kf is None or Q is None:
        print(f"L{L}: missing"); continue
    n = min(Km.shape[0], Kf.shape[0]); Km, Kf = Km[:n], Kf[:n]
    a = attn_mass(Q, Km)[:n]; an = a / a.sum()
    rerr = (1 - F.cosine_similarity(Km, Kf, dim=-1)).clamp(0, 2)
    print(f"-- L{L} --")
    for name, (s, e) in SEGS.items():
        e = min(e, n)
        raw = float(rerr[s:e].mean())
        mass = float(an[s:e].sum()) * 100
        w = float((an[s:e] * rerr[s:e]).sum())
        print(f"   {name:>4}: raw={raw:.4f}  attn-mass={mass:5.1f}%  weighted={w:.5f}")
