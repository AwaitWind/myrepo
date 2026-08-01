import glob, os, re, torch
import torch.nn.functional as F
KD = "/tmp/kd_obs2"; MODE = "pic_a3_oracle"; REF = "full_recompute"
def loadK(tag, L):
    cs = glob.glob(f"{KD}/{tag}_L{L}_c*.pt")
    if not cs: return None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()
def loadQ(tag, L):
    f = f"{KD}/{tag}_Q_L{L}.pt"
    if not os.path.exists(f): return None
    d = torch.load(f, map_location="cpu"); return d["q"].float(), float(d["scale"]), d["qpos"].long()
def loadimp(tag, L):
    f = f"{KD}/{tag}_IMP_L{L}.pt"
    return set(int(x) for x in torch.load(f, map_location="cpu").tolist()) if os.path.exists(f) else None
def attn_mass(Q, K):
    q, scale, qpos = Q; n = K.shape[0]
    score = (q @ K.T) * scale
    kidx = torch.arange(n)
    score = score.masked_fill(kidx[None, :] > qpos[:, None], float("-inf"))
    return F.softmax(score, dim=1).sum(0)
SEG_HEADS = [(0,64),(64,128),(1216,1280),(2304,2368),(3392,3456)]  # segment-start sinks
print("Does attention read already-fresh(imp) or reused(non-imp) K?  MODE=%s vs %s" % (MODE, REF))
print("  massImp/massNon = %% attn mass on imp vs non-imp keys")
print("  errImp/errNon   = attn-weighted raw-K-error contributed by each")
print("  recompGain      = errNon/(errImp+errNon) = %% of attn-weighted error removable by recomputing the reused keys attention actually reads")
# oracle reselects imp at 20/40/60; use that layer's own imp set
for L in [20, 40, 60]:
    K, Kf, Q = loadK(MODE, L), loadK(REF, L), loadQ(MODE, L)
    imp = loadimp(MODE, L) or loadimp(MODE, 1)
    if K is None or Kf is None or Q is None or imp is None:
        print(f"L{L}: missing"); continue
    n = min(K.shape[0], Kf.shape[0])
    K, Kf = K[:n], Kf[:n]
    a = attn_mass(Q, K)[:n]; a = a / a.sum()
    rerr = (1 - F.cosine_similarity(K, Kf, dim=-1)).clamp(0, 2)
    m = torch.zeros(n, dtype=torch.bool)
    for p in imp:
        if p < n: m[p] = True
    massImp = float(a[m].sum()); massNon = float(a[~m].sum())
    errImp = float((a[m] * rerr[m]).sum()); errNon = float((a[~m] * rerr[~m]).sum())
    gain = errNon / max(errImp + errNon, 1e-12) * 100
    # top-50 attention keys: how many already imp?
    topk = a.topk(min(50, n)).indices
    top_imp = float(m[topk].float().mean()) * 100
    # attention mass landing on the segment-head sinks, and are they imp?
    head_mask = torch.zeros(n, dtype=torch.bool)
    for (s, e) in SEG_HEADS:
        head_mask[s:min(e, n)] = True
    head_mass = float(a[head_mask].sum()) * 100
    head_imp = float(m[head_mask].float().mean()) * 100
    print(f"L{L}: massImp={massImp*100:.1f}%% massNon={massNon*100:.1f}%%  "
          f"errImp={errImp:.4f} errNon={errNon:.4f}  recompGain={gain:.0f}%%  "
          f"| top50-attn imp={top_imp:.0f}%%  segHeads: mass={head_mass:.1f}%% imp={head_imp:.0f}%%")
