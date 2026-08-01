"""Per-layer hidden-state INCREMENT (delta = h[L] - h[L-1]) deviation: oracle vs full.

Each layer L writes attn+mlp output into the residual stream; delta_h[L] =
h[L]-h[L-1] is that layer's CONTRIBUTION. This compares oracle's per-layer
contribution (rs_full, isolated) to full_recompute's, per (layer, token) —
showing whether oracle's layers actively compute different updates (not just
accumulate drift). METRIC=cos (1-cos) | euclidean (||.||).

reads kd_rs78: {ORACLE}_RSFULL_L{L}.pt, {REF}_RS_L{L}.pt  ({h,pos})
USAGE: KD=/tmp/kd_rs78 METRIC=cos python kdump_hidden_delta.py
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

KD = os.environ.get("KD", "/tmp/kd_rs78")
ORACLE = os.environ.get("ORACLE", "pic_a3_oracle")
REF = os.environ.get("REF", "full_recompute")
METRIC = os.environ.get("METRIC", "cos")
MLABEL = {"cos": "1-cos(delta)", "euclidean": "L2(delta) euclidean"}.get(METRIC, METRIC)
BIN = int(os.environ.get("BIN", "64"))
QSTART = int(os.environ.get("QSTART", "3392"))
OUT = os.environ.get("OUT", f"{KD}/hidden_delta_{METRIC}.png")
SEGS = {}
for _p in os.environ.get("SEGS", "SYS:0-64,C1:64-1216,C2:1216-2304,C3:2304-3392").split(","):
    if ":" in _p and "-" in _p:
        n, r = _p.split(":")
        s, e = r.split("-")
        SEGS[n] = (int(s), int(e))


def _load(tag, kind, L):
    f = f"{KD}/{tag}_{kind}_L{L}.pt"
    if not os.path.exists(f):
        return None
    d = torch.load(f, map_location="cpu")
    return d["h"].float(), d["pos"].long()


def _avail(tag, kind):
    s = set()
    for p in glob.glob(f"{KD}/{tag}_{kind}_L*.pt"):
        m = re.search(rf"{tag}_{kind}_L(\d+)\.pt$", p)
        if m:
            s.add(int(m.group(1)))
    return s


def _delta(tag, kind, L):
    a, b = _load(tag, kind, L), _load(tag, kind, L - 1)
    if a is None or b is None:
        return None
    ah, _ = a
    bh, _ = b
    n = min(ah.shape[0], bh.shape[0])
    return ah[:n] - bh[:n]                      # (n, dim) layer-L contribution


both = _avail(ORACLE, "RSFULL") & _avail(REF, "RS")
Ls = sorted(L for L in both if (L - 1) in both)
rows = []
for L in Ls:
    do, df = _delta(ORACLE, "RSFULL", L), _delta(REF, "RS", L)
    if do is None or df is None:
        rows.append(None)
        continue
    n = min(do.shape[0], df.shape[0])
    if METRIC == "euclidean":
        vals = (do[:n] - df[:n]).norm(dim=-1)
    else:
        vals = 1 - F.cosine_similarity(do[:n], df[:n], dim=-1)
    dev = np.full(QSTART, np.nan, np.float32)
    v = vals.numpy()
    for p in range(min(n, QSTART)):
        dev[p] = v[p]
    rows.append(dev)
    sm = {name: float(np.nanmean(dev[s:min(e, QSTART)])) for name, (s, e) in SEGS.items()}
    print(f"L{L}: " + "  ".join(f"{k}={sm[k]:.3f}" for k in SEGS))

valid = [(L, d) for L, d in zip(Ls, rows) if d is not None]
if not valid:
    raise SystemExit("nothing to plot")
nb = (QSTART + BIN - 1) // BIN
M = np.full((len(valid), nb), np.nan, np.float32)
for r, (L, d) in enumerate(valid):
    for k in range(nb):
        seg = d[k * BIN:min((k + 1) * BIN, QSTART)]
        if np.any(~np.isnan(seg)):
            M[r, k] = np.nanmean(seg)

_h = float(min(9.0, max(4.0, 0.11 * len(valid) + 3.0)))
fig, ax = plt.subplots(figsize=(14, _h))
im = ax.imshow(M, aspect="auto", origin="lower", cmap="magma", vmin=0,
               interpolation="nearest")
_ys = max(1, len(valid) // 16)
_yi = list(range(0, len(valid), _ys))
ax.set_yticks(_yi)
ax.set_yticklabels([f"L{valid[i][0]}" for i in _yi])
xt = list(range(0, nb, max(1, nb // 12)))
ax.set_xticks(xt)
ax.set_xticklabels([str(i * BIN) for i in xt])
for name, (s, e) in SEGS.items():
    ax.axvline(s / BIN - 0.5, color="cyan", lw=0.8, ls="--", alpha=0.8)
    ax.text(s / BIN, len(valid) - 0.5, name, color="cyan", fontsize=10, va="top")
ax.set_xlabel("doc token position")
ax.set_ylabel("layer")
ax.set_title(f"per-layer hidden INCREMENT (h[L]-h[L-1]) dev: {ORACLE} vs {REF}  ({MLABEL})")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, label=MLABEL)
plt.tight_layout()
plt.savefig(OUT, dpi=120)
print("saved", OUT)
