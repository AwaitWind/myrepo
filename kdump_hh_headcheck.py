import glob, os, re, torch
import torch.nn.functional as F
KD = "/tmp/kd_hh"; REF = "fullref"
def loadK(tag, L):
    cs = glob.glob(f"{KD}/{tag}_L{L}_c*.pt")
    if not cs: return None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()
def cvec(a, b):
    n = min(a.shape[0], b.shape[0]); return F.cosine_similarity(a[:n], b[:n], dim=-1)
# hit doc segments (measure): SYS 0-64, C1 64-1216, C2 1216-2304, C3 2304-3392.
# FORCE_HIT_HEAD=128 forces first 128 of each: 0-64, 64-192, 1216-1344, 2304-2432.
HEADS = [(0,64),(64,192),(1216,1344),(2304,2432)]
def head_idx(n):
    xs=[]
    for s,e in HEADS: xs += [p for p in range(s,min(e,n))]
    return torch.tensor(xs)
print("raw K error (1-cos vs fullref) — does forcing segment-heads fresh drop it?")
print("  L  | region      | baseline oracle | oracle+HEAD128")
for L in [2, 20, 40, 60, 77]:
    rf = loadK(REF, L); bs = loadK("pic_a3_oracle", L); hh = loadK("oracle_hh", L)
    if rf is None or bs is None or hh is None:
        print(f"L{L}: missing (rf={rf is not None} bs={bs is not None} hh={hh is not None})"); continue
    n = min(rf.shape[0], bs.shape[0], hh.shape[0])
    hi = head_idx(n)
    def me(K, idx=None):
        c = cvec(K, rf)
        c = c[idx] if idx is not None else c
        return float((1 - c).clamp(0, 2).mean())
    print(f"{L:>4} | seg-heads   | {me(bs, hi):>15.4f} | {me(hh, hi):>13.4f}")
    print(f"     | ALL pos     | {me(bs):>15.4f} | {me(hh):>13.4f}")
