"""(K_mode * attn_mode - K_fc * attn_fc) euclidean deviation, per key / layer.

For each mode vs full_recompute, per (layer L, key position k):
    value = || a_mode[k] * K_mode[k]  -  a_fc[k] * K_fc[k] ||   (euclidean)
where
    a[k]  = total attention mass the QUERY segment puts on key k
            (reconstructed from summed-Q · K, causal-masked softmax, summed over
             query tokens — each mode uses its OWN Q and K),
    K[k]  = the 576-dim cached MLA latent at (layer, key).
Combines K corruption AND attention-weight difference into one "weighted
key-contribution" euclidean deviation. Uses the CLEAN (post probe-fix) dumps.

Files (kd_clean): {tag}_L{L}_c*.pt (K),  {tag}_Q_L{L}.pt ({q, scale, qpos}).
USAGE: KD=/tmp/kd_clean REF=full_recompute python kdump_wkv_euclid.py
ENV: KD REF MODES BIN SQUARED(=1 → value^2) OUT SEGS
"""
import glob
import os
import re

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch.nn.functional as F  # noqa: E402

KD = os.environ.get("KD", "/tmp/kd_clean")
REF = os.environ.get("REF", "full_recompute")
BIN = int(os.environ.get("BIN", "64"))
MODES = os.environ.get("MODES", "pic,pic_cacheblend,pic_a3,pic_a3_oracle").split(",")
SQUARED = os.environ.get("SQUARED", "0") == "1"
OUT = os.environ.get("OUT", f"{KD}/kdump_wkv_euclid.png")
_SEG = "SYS:0-64,C1:64-1216,C2:1216-2304,C3:2304-3392,Q:3392-3456"
SEGS = {}
for _p in os.environ.get("SEGS", _SEG).split(","):
    if ":" in _p and "-" in _p:
        n, r = _p.split(":")
        s, e = r.split("-")
        SEGS[n] = (int(s), int(e))


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
    ls = set()
    for p in glob.glob(f"{KD}/{tag}_L*.pt"):
        m = re.search(rf"{re.escape(tag)}_L(\d+)(?:_c\d+)?\.pt$", p)
        if m:
            ls.add(int(m.group(1)))
    return ls


def attn_mass(Q, K):
    q, scale, qpos = Q
    n = K.shape[0]
    score = (q @ K.T) * scale                      # (Nq, n)
    kidx = torch.arange(n)
    score = score.masked_fill(kidx[None, :] > qpos[:, None], float("-inf"))
    w = F.softmax(score, dim=1)
    return w.sum(0)                                 # (n,) total mass per key


def weighted_dev(mode, L):
    Km, Kf = loadK(mode, L), loadK(REF, L)
    Qm, Qf = loadQ(mode, L), loadQ(REF, L)
    if any(x is None for x in (Km, Kf, Qm, Qf)):
        return None
    n = min(Km.shape[0], Kf.shape[0])
    Km, Kf = Km[:n], Kf[:n]
    am = attn_mass(Qm, Km)[:n]
    af = attn_mass(Qf, Kf)[:n]
    d = (am[:, None] * Km - af[:, None] * Kf).norm(dim=-1)   # (n,) euclidean per key
    if SQUARED:
        d = d * d
    return d.numpy(), n


ref_Ls = layersK(REF)
mats, layer_axes = {}, {}
for mode in MODES:
    Ls = sorted(layersK(mode) & ref_Ls)
    rows, nmin = [], None
    for L in Ls:
        r = weighted_dev(mode, L)
        if r is None:
            rows.append(None)
            continue
        d, n = r
        rows.append(d)
        nmin = n if nmin is None else min(nmin, n)
    if nmin is None:
        print(f"[skip] {mode}: missing K/Q")
        continue
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
    print(f"{mode:>16s}: layers={len(Ls)} keys={nmin} "
          f"mean={np.nanmean(M):.4f} max={np.nanmax(M):.4f}")

if not mats:
    raise SystemExit("nothing to plot (need K + Q for modes and REF)")
_lab = "||K*attn_mode - K*attn_fc||" + ("^2" if SQUARED else "")
vmax = max(float(np.nanmax(M)) for M in mats.values()) or 1.0
n = len(mats)
fig, axes = plt.subplots(n, 1, figsize=(15, 3.1 * n), squeeze=False)
for ax, mode in zip(axes[:, 0], mats):
    M, Ls = mats[mode], layer_axes[mode]
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="magma",
                   vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{mode}  vs  {REF}   —   {_lab}   "
                 f"mean={np.nanmean(M):.4f} max={np.nanmax(M):.4f}", fontsize=10)
    ax.set_ylabel("layer")
    yt = list(range(0, len(Ls), max(1, len(Ls) // 12)))
    ax.set_yticks(yt)
    ax.set_yticklabels([Ls[i] for i in yt])
    nb = M.shape[1]
    xt = list(range(0, nb, max(1, nb // 12)))
    ax.set_xticks(xt)
    ax.set_xticklabels([str(i * BIN) for i in xt])
    for name, (s, _e) in SEGS.items():
        xb = s / BIN - 0.5
        if -0.5 <= xb <= nb:
            ax.axvline(xb, color="cyan", lw=0.7, ls="--", alpha=0.7)
            ax.text(max(xb, 0), len(Ls) - 0.5, name, color="cyan",
                    fontsize=8, va="top", ha="left")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label=_lab)
axes[-1, 0].set_xlabel("key position")
fig.suptitle(f"weighted-K euclidean deviation  {_lab}  (dir={KD}, bin={BIN})",
             fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.savefig(OUT, dpi=110)
print(f"saved {OUT}")
