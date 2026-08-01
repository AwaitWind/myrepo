"""Per-layer hidden-state evolution: oracle (rs_full) vs full_recompute.

Line plots over layers (mean over doc positions):
  Panel 1  cross-mode similarity   cos(h_oracle[L], h_full[L])       — how close
           oracle's hidden is to full's, layer by layer.
  Panel 2  per-layer INCREMENT     ||h[L] - h[L-1]||                 — how much each
           layer changes the residual stream (oracle vs full).
  Panel 3  layer-to-layer sim      cos(h[L], h[L-1])                 — direction
           stability of each layer's step (oracle vs full).

reads kd_rs78: {ORACLE}_RSFULL_L{L}.pt, {REF}_RS_L{L}.pt  ({h,pos})
USAGE: KD=/tmp/kd_rs78 python kdump_hidden_evolution.py
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
QSTART = int(os.environ.get("QSTART", "3392"))
OUT = os.environ.get("OUT", f"{KD}/hidden_evolution.png")


def _load(tag, kind, L):
    f = f"{KD}/{tag}_{kind}_L{L}.pt"
    if not os.path.exists(f):
        return None
    d = torch.load(f, map_location="cpu")
    h, pos = d["h"].float(), d["pos"].long()
    m = pos < QSTART                       # doc positions only
    return h[m]


def _avail(tag, kind):
    s = set()
    for p in glob.glob(f"{KD}/{tag}_{kind}_L*.pt"):
        mm = re.search(rf"{tag}_{kind}_L(\d+)\.pt$", p)
        if mm:
            s.add(int(mm.group(1)))
    return s


Ls = sorted(_avail(ORACLE, "RSFULL") & _avail(REF, "RS"))
xcross, cross = [], []          # cos(h_o[L], h_f[L])
xinc, inc_o, inc_f = [], [], []     # ||h[L]-h[L-1]||
coscon_o, coscon_f = [], []         # cos(h[L], h[L-1])
prev_o = prev_f = None
prev_L = None
for L in Ls:
    ho, hf = _load(ORACLE, "RSFULL", L), _load(REF, "RS", L)
    if ho is None or hf is None:
        prev_o = prev_f = None
        continue
    n = min(ho.shape[0], hf.shape[0])
    ho, hf = ho[:n], hf[:n]
    xcross.append(L)
    cross.append(float(F.cosine_similarity(ho, hf, dim=-1).mean()))
    if prev_o is not None and prev_L == L - 1:
        m = min(ho.shape[0], prev_o.shape[0])
        inc_o.append(float((ho[:m] - prev_o[:m]).norm(dim=-1).mean()))
        inc_f.append(float((hf[:m] - prev_f[:m]).norm(dim=-1).mean()))
        coscon_o.append(float(F.cosine_similarity(ho[:m], prev_o[:m], dim=-1).mean()))
        coscon_f.append(float(F.cosine_similarity(hf[:m], prev_f[:m], dim=-1).mean()))
        xinc.append(L)
    prev_o, prev_f, prev_L = ho, hf, L

print(f"layers={len(xcross)}  cross-cos range [{min(cross):.3f},{max(cross):.3f}]")

fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
axes[0].plot(xcross, cross, "-o", ms=3, color="crimson")
axes[0].set_ylabel("cos(h_oracle, h_full)")
axes[0].set_title("① cross-mode hidden similarity per layer (oracle rs_full vs full_recompute)")
axes[0].grid(alpha=0.3)
axes[0].axhline(1.0, color="gray", lw=0.6, ls=":")

axes[1].plot(xinc, inc_o, "-o", ms=3, label="oracle", color="darkorange")
axes[1].plot(xinc, inc_f, "-o", ms=3, label="full_recompute", color="steelblue")
axes[1].set_ylabel("||h[L] - h[L-1]||")
axes[1].set_title("② per-layer hidden INCREMENT magnitude (how much each layer changes h)")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(xinc, coscon_o, "-o", ms=3, label="oracle", color="darkorange")
axes[2].plot(xinc, coscon_f, "-o", ms=3, label="full_recompute", color="steelblue")
axes[2].set_ylabel("cos(h[L], h[L-1])")
axes[2].set_title("③ layer-to-layer similarity within each mode (step direction stability)")
axes[2].set_xlabel("layer")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=120)
print("saved", OUT)
