"""Explain + reveal the (kv*attn - kv_fc*attn_fc)^2 "dark" plot.

Same weighted-K deviation as kdump_wkv_euclid.py, PLUS:
  1. PER-LAYER normalization (each layer row scaled to its own max) so the
     within-layer structure is visible even though absolute values are tiny.
  2. A printed SUPPRESSION diagnostic per layer: raw K error (1-cos) vs the
     attention-WEIGHTED K error, and how much attention mass lands on the
     highest-error keys — quantifying WHY the raw plot is dark (attention
     barely looks at the corrupted/reused keys).

reads {KD}: {tag}_L*.pt (K), {tag}_Q_L*.pt ({q,scale,qpos})
USAGE: KD=/tmp/kd_obs2 REF=full_recompute MODES=pic_a3,pic_a3_oracle \
       python kdump_wkv_clear.py
ENV: KD REF MODES BIN OUT SEGS
"""
import glob, os, re
import matplotlib; matplotlib.use("Agg")
import numpy as np, torch
import matplotlib.pyplot as plt  # noqa: E402
import torch.nn.functional as F  # noqa: E402

KD = os.environ.get("KD", "/tmp/kd_obs2")
REF = os.environ.get("REF", "full_recompute")
BIN = int(os.environ.get("BIN", "64"))
MODES = os.environ.get("MODES", "pic_a3,pic_a3_oracle").split(",")
OUT = os.environ.get("OUT", f"{KD}/kv_wkv_clear.png")
_SEG = "SYS:0-64,C1:64-1216,C2:1216-2304,C3:2304-3392,Q:3392-3456"
SEGS = {}
for _p in os.environ.get("SEGS", _SEG).split(","):
    if ":" in _p and "-" in _p:
        n, r = _p.split(":"); s, e = r.split("-"); SEGS[n] = (int(s), int(e))


def loadK(tag, L):
    cs = glob.glob(f"{KD}/{tag}_L{L}_c*.pt")
    if not cs:
        return None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()


def loadQ(tag, L):
    f = f"{KD}/{tag}_Q_L{L}.pt"
    if not os.path.exists(f):
        return None
    d = torch.load(f, map_location="cpu")
    return d["q"].float(), float(d["scale"]), d["qpos"].long()


def layersK(tag):
    s = set()
    for p in glob.glob(f"{KD}/{tag}_L*.pt"):
        m = re.search(rf"{re.escape(tag)}_L(\d+)(?:_c\d+)?\.pt$", p)
        if m:
            s.add(int(m.group(1)))
    return s


def attn_mass(Q, K):
    q, scale, qpos = Q
    n = K.shape[0]
    score = (q @ K.T) * scale
    kidx = torch.arange(n)
    score = score.masked_fill(kidx[None, :] > qpos[:, None], float("-inf"))
    w = F.softmax(score, dim=1)
    return w.sum(0)                                    # (n,) total mass per key


ref_Ls = layersK(REF)
mats, layer_axes = {}, {}
print("SUPPRESSION diagnostic (why the weighted plot is dark):")
print("  rawErr = mean(1-cos K vs ref) over keys  |  wErr = attention-mass-weighted 1-cos")
print("  massTop = %% of total attn mass on the worst-10%%-error keys (uniform would be 10%%)")
for mode in MODES:
    Ls = sorted(layersK(mode) & ref_Ls)
    rows, nmin = [], None
    print(f"-- {mode} --")
    for L in Ls:
        Km, Kf = loadK(mode, L), loadK(REF, L)
        Qm = loadQ(mode, L)
        if Km is None or Kf is None or Qm is None:
            rows.append(None); continue
        n = min(Km.shape[0], Kf.shape[0])
        Km, Kf = Km[:n], Kf[:n]
        am = attn_mass(Qm, Km)[:n]                     # this mode's attention mass per key
        # weighted euclidean deviation (same as kdump_wkv_euclid, SQUARED)
        af = attn_mass(loadQ(REF, L), Kf)[:n] if loadQ(REF, L) is not None else am
        d = (am[:, None] * Km - af[:, None] * Kf).norm(dim=-1)
        d = (d * d).numpy()
        rows.append(d)
        nmin = n if nmin is None else min(nmin, n)
        # diagnostic
        rawerr = (1 - F.cosine_similarity(Km, Kf, dim=-1)).clamp(0, 2)   # (n,)
        massn = (am / am.sum()).numpy()
        rerr = rawerr.numpy()
        wErr = float((massn * rerr).sum())             # attn-weighted mean raw err
        rawE = float(rerr.mean())
        thr = np.quantile(rerr, 0.90)
        top = rerr >= thr
        massTop = float(massn[top].sum() * 100)
        if L in (2, 20, 40, 60, 77):
            print(f"  L{L:>2}: rawErr={rawE:.3f}  wErr={wErr:.4f}  "
                  f"suppress={rawE/max(wErr,1e-9):.0f}x  massTop10%err={massTop:.1f}%")
    if nmin is None:
        print(f"[skip] {mode}"); continue
    nb = (nmin + BIN - 1) // BIN
    M = np.full((len(Ls), nb), np.nan, np.float32)
    for ri, d in enumerate(rows):
        if d is None:
            continue
        for k in range(nb):
            seg = d[k * BIN:min((k + 1) * BIN, nmin)]
            if len(seg):
                M[ri, k] = float(np.mean(seg))
    mats[mode], layer_axes[mode] = M, Ls

# PER-LAYER normalized plot (reveal structure hidden by the tiny absolute values)
n = len(mats)
fig, axes = plt.subplots(n, 1, figsize=(15, 3.2 * n), squeeze=False)
for ax, mode in zip(axes[:, 0], mats):
    M, Ls = mats[mode].copy(), layer_axes[mode]
    absmax = np.nanmax(M)
    Mn = np.full_like(M, np.nan)
    for r in range(M.shape[0]):
        rmax = np.nanmax(M[r])
        if rmax and rmax > 0:
            Mn[r] = M[r] / rmax                        # each layer scaled to its own max
    im = ax.imshow(Mn, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_title(f"{mode} vs {REF}  —  ||K*attn - K_fc*attn_fc||^2  "
                 f"(PER-LAYER normalized; abs max over all ={absmax:.4f})", fontsize=9)
    ax.set_ylabel("layer")
    yt = list(range(0, len(Ls), max(1, len(Ls) // 12)))
    ax.set_yticks(yt); ax.set_yticklabels([Ls[i] for i in yt])
    nb = M.shape[1]
    xt = list(range(0, nb, max(1, nb // 12)))
    ax.set_xticks(xt); ax.set_xticklabels([str(i * BIN) for i in xt])
    for name, (s, _e) in SEGS.items():
        xb = s / BIN - 0.5
        if -0.5 <= xb <= nb:
            ax.axvline(xb, color="cyan", lw=0.7, ls="--", alpha=0.7)
            ax.text(max(xb, 0), len(Ls) - 0.5, name, color="cyan", fontsize=8, va="top")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="per-layer normalized")
axes[-1, 0].set_xlabel("key position")
fig.suptitle("weighted-K deviation, PER-LAYER normalized (reveals structure hidden by "
             "attention suppression)", fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.savefig(OUT, dpi=110)
print("saved", OUT)
