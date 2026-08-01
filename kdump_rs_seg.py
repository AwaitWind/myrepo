"""Reused doc-segment hidden-state deviation at oracle check layers.

Compares oracle's reconstructed full-length residual stream `rs_full` (assembled
from the isolated store — the hidden oracle's deep re-selection actually USES)
against full_recompute's TRUE in-context hidden, for the REUSED document segments
(SYS/C1/C2/C3), at the check layers (1/20/40/60). rs_full IS full-length, so
every doc position is present ("全有的").

Files:
  {oracle}_RSFULL_L{L}.pt = {h:(full_len,6144) bf16, pos}   (isolated-store rebuild)
  {ref}_RS_L{L}.pt        = {h:(seqlen,6144) bf16, pos}      (FORCE_ALL_IMP dense = true)

USAGE: KD=/tmp/kd_rs ORACLE=pic_a3_oracle REF=full_recompute python kdump_rs_seg.py
"""
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch.nn.functional as F  # noqa: E402

KD = os.environ.get("KD", "/tmp/kd_rs")
ORACLE = os.environ.get("ORACLE", "pic_a3_oracle")
REF = os.environ.get("REF", "full_recompute")
import glob as _glob
import re as _re

_lenv = os.environ.get("LAYERS", "")
if _lenv:
    LAYERS = [int(x) for x in _lenv.split(",")]
else:
    # auto-detect: all layers where BOTH oracle RSFULL and ref RS exist
    def _avail(tag, kind):
        s = set()
        for _p in _glob.glob(f"{KD}/{tag}_{kind}_L*.pt"):
            _m = _re.search(rf"{tag}_{kind}_L(\d+)\.pt$", _p)
            if _m:
                s.add(int(_m.group(1)))
        return s

    LAYERS = sorted(
        _avail(os.environ.get("ORACLE", "pic_a3_oracle"), "RSFULL")
        & _avail(os.environ.get("REF", "full_recompute"), "RS")
    )
BIN = int(os.environ.get("BIN", "64"))
QSTART = int(os.environ.get("QSTART", "3392"))
OUT = os.environ.get("OUT", f"{KD}/kdump_rs_seg.png")
_SEG_DEFAULT = "SYS:0-64,C1:64-1216,C2:1216-2304,C3:2304-3392"
SEGS = {}
for _p in os.environ.get("SEGS", _SEG_DEFAULT).split(","):
    if ":" in _p and "-" in _p:
        n, r = _p.split(":")
        s, e = r.split("-")
        SEGS[n] = (int(s), int(e))

METRIC = os.environ.get("METRIC", "cos")
MLABEL = {"cos": "1-cos", "euclidean": "L2 euclidean dist"}.get(METRIC, METRIC)


def _load(tag, kind, L):
    f = f"{KD}/{tag}_{kind}_L{L}.pt"
    if not os.path.exists(f):
        return None
    d = torch.load(f, map_location="cpu")
    return d["h"].float(), d["pos"].long()


rows = []
for L in LAYERS:
    o, r = _load(ORACLE, "RSFULL", L), _load(REF, "RS", L)
    if o is None or r is None:
        print(f"L{L}: missing (oracle_RSFULL={o is not None} ref_RS={r is not None})")
        rows.append(None)
        continue
    oh, opos = o
    rh, rpos = r
    rmap = {int(p): i for i, p in enumerate(rpos.tolist())}
    oi, rj = [], []
    for i, p in enumerate(opos.tolist()):
        if p < QSTART and p in rmap:
            oi.append(i)
            rj.append(rmap[p])
    dev = np.full(QSTART, np.nan, np.float32)
    if oi:
        if METRIC == "euclidean":
            vals = (oh[oi] - rh[rj]).norm(dim=-1)            # ||a-b|| per position
        else:
            vals = 1 - F.cosine_similarity(oh[oi], rh[rj], dim=-1)
        pos_arr = [int(opos[i]) for i in oi]
        for p, c in zip(pos_arr, vals.tolist()):
            dev[p] = c
    rows.append(dev)
    sm = {name: float(np.nanmean(dev[s:min(e, QSTART)])) for name, (s, e) in SEGS.items()}
    cov = float(np.mean(~np.isnan(dev)))
    print(f"L{L}: coverage={cov:.0%}  " + "  ".join(f"{n}={sm[n]:.3f}" for n in SEGS))

valid = [(L, d) for L, d in zip(LAYERS, rows) if d is not None]
if not valid:
    raise SystemExit("nothing to plot")
nb = (QSTART + BIN - 1) // BIN
M = np.full((len(valid), nb), np.nan, np.float32)
for r, (L, d) in enumerate(valid):
    for k in range(nb):
        seg = d[k * BIN:min((k + 1) * BIN, QSTART)]
        if np.any(~np.isnan(seg)):
            M[r, k] = np.nanmean(seg)

_h = float(min(9.0, max(4.0, 0.11 * len(valid) + 3.0)))   # sane height, capped
fig, ax = plt.subplots(figsize=(14, _h))
cmap = plt.get_cmap("magma").copy()
cmap.set_bad("#333333")
im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=0,
               interpolation="nearest")
_ystep = max(1, len(valid) // 16)
_yidx = list(range(0, len(valid), _ystep))
ax.set_yticks(_yidx)
ax.set_yticklabels([f"L{valid[i][0]}" for i in _yidx])
xt = list(range(0, nb, max(1, nb // 12)))
ax.set_xticks(xt)
ax.set_xticklabels([str(i * BIN) for i in xt])
for name, (s, e) in SEGS.items():
    ax.axvline(s / BIN - 0.5, color="cyan", lw=0.8, ls="--", alpha=0.8)
    ax.text(s / BIN, len(valid) - 0.5, name, color="cyan", fontsize=10, va="top")
ax.set_xlabel("doc token position (reused segments)")
ax.set_ylabel("check layer")
ax.set_title(
    f"reused doc-segment hidden deviation:  {ORACLE} rs_full  vs  {REF}  ({MLABEL})"
)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, label=MLABEL)
plt.tight_layout()
plt.savefig(OUT, dpi=120)
print("saved", OUT)
