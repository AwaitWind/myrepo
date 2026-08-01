"""Per-layer per-token error 观测台 for PIC reuse modes.

Renders a heatmap per mode: x-axis = token position, y-axis = layer, color =
reuse error (1 - cos) between that mode's K and a REFERENCE mode's K at each
(layer, token). This makes the layer-by-layer drift of KV reuse VISIBLE — you
can see where (which segment) and when (which layer) a mode diverges from the
ground truth, and how it compounds with depth.

INPUT — produced by the deepseek_v2 K-dump probe run with:
    SGLANG_PIC_KDUMP_DIR=/tmp/kd_obs SGLANG_PIC_KDUMP_ALL=1 \
      python quick_test_online.py --modes full_recompute pic pic_cacheblend \
        pic_a3 pic_a3_oracle --oracle-layers 1 20 40 60 --tp 8
  Each file: {mode}_L{layer}_c{fwd}.pt = (seq_len, K_dim) float, POSITION order
  (the K each position's attention reads at that layer). quick_test_online.py
  sets SGLANG_PIC_KDUMP_TAG=<mode> per mode so all 5 dump under distinct tags.

USAGE:
    KD=/tmp/kd_obs REF=full_recompute python kdump_heatmap.py
ENV:
    KD      dump dir (default /tmp/kd_obs)
    REF     reference tag; error is measured against this (default full_recompute)
    MODES   comma list to plot (default: every tag found except REF)
    BIN     x-axis token bin width, mean-pooled (default 64 = page size)
    METRIC  cos → 1-cos(K) (default) | l2 → ||a-b|| / ||b||
    C       per-forward counter to select (default: the max c present per file)
    OUT     output PNG (default {KD}/kdump_heatmap.png)
    SEGS    "name:start-end,..." segment boundaries to annotate
            (default = the synthetic 3-doc layout SYS/C1/C2/C3/Q)
"""
import glob
import os
import re

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KD = os.environ.get("KD", "/tmp/kd_obs")
REF = os.environ.get("REF", "full_recompute")
BIN = int(os.environ.get("BIN", "64"))
METRIC = os.environ.get("METRIC", "cos")
MLABEL = {"cos": "1-cos(K)", "l2": "relL2(K)",
          "euclidean": "L2(K) euclidean dist"}.get(METRIC, METRIC)
IMP_OVERLAY = os.environ.get("IMP_OVERLAY", "0") == "1"
OUT = os.environ.get("OUT", f"{KD}/kdump_heatmap.png")
C_ENV = os.environ.get("C", "")
# Default segment annotation = the synthetic 3-doc test (matches kdump_cos.py).
_SEG_DEFAULT = "SYS:0-64,C1:64-1216,C2:1216-2304,C3:2304-3392,Q:3392-3456"
SEGS = {}
for _part in os.environ.get("SEGS", _SEG_DEFAULT).split(","):
    if ":" in _part and "-" in _part:
        _n, _r = _part.split(":")
        _s, _e = _r.split("-")
        SEGS[_n] = (int(_s), int(_e))

_FILE_RE = re.compile(r".*/(?P<tag>.+)_L(?P<L>\d+)(?:_c(?P<c>\d+))?\.pt$")


def _all_tags():
    tags = set()
    for p in glob.glob(f"{KD}/*_L*.pt"):
        m = _FILE_RE.match(p)
        if m:
            tags.add(m.group("tag"))
    return tags


def _layers(tag):
    ls = set()
    for p in glob.glob(f"{KD}/{tag}_L*.pt"):
        m = _FILE_RE.match(p)
        if m and m.group("tag") == tag:
            ls.add(int(m.group("L")))
    return sorted(ls)


def _load(tag, L):
    """Load (seq_len, K_dim). Pick file by C env, else the max-counter file."""
    if C_ENV:
        for f in (f"{KD}/{tag}_L{L}_c{C_ENV}.pt", f"{KD}/{tag}_L{L}.pt"):
            if os.path.exists(f):
                return torch.load(f, map_location="cpu").flatten(1).float()
        return None
    cands = glob.glob(f"{KD}/{tag}_L{L}_c*.pt")
    if cands:
        def _cnum(p):
            m = re.search(r"_c(\d+)\.pt$", p)
            return int(m.group(1)) if m else -1

        return torch.load(max(cands, key=_cnum), map_location="cpu").flatten(1).float()
    f0 = f"{KD}/{tag}_L{L}.pt"
    return torch.load(f0, map_location="cpu").flatten(1).float() if os.path.exists(f0) else None


def _err(a, b):
    """Per-token error between mode K (a) and ref K (b); min-length aligned.
    METRIC: cos → 1-cos ; l2 → relative L2 ||a-b||/||b|| ; euclidean → raw ||a-b||."""
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    if METRIC == "l2":
        return (((a - b).norm(dim=-1)) / (b.norm(dim=-1) + 1e-6)).numpy(), n
    if METRIC == "euclidean":
        return ((a - b).norm(dim=-1)).numpy(), n
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1)
    return (1.0 - cos).clamp(0.0, 2.0).numpy(), n


def main():
    tags = _all_tags()
    if REF not in tags:
        raise SystemExit(f"REF tag '{REF}' not found in {KD}. Tags present: {sorted(tags)}")
    modes = os.environ.get("MODES")
    modes = modes.split(",") if modes else [t for t in sorted(tags) if t != REF]
    ref_layers = set(_layers(REF))
    if not ref_layers:
        raise SystemExit(f"No layer dumps for REF '{REF}' in {KD}")

    mats, layer_axes = {}, {}
    for mode in modes:
        Ls = sorted(set(_layers(mode)) & ref_layers)
        if not Ls:
            print(f"[skip] {mode}: no layers shared with REF {REF}")
            continue
        rows, n_min = [], None
        for L in Ls:
            a, b = _load(mode, L), _load(REF, L)
            if a is None or b is None:
                rows.append(None)
                continue
            e, n = _err(a, b)
            rows.append(e)
            n_min = n if n_min is None else min(n_min, n)
        if n_min is None:
            print(f"[skip] {mode}: no loadable layer pairs")
            continue
        nb = (n_min + BIN - 1) // BIN
        M = np.full((len(Ls), nb), np.nan, dtype=np.float32)
        for r, e in enumerate(rows):
            if e is None:
                continue
            for k in range(nb):
                seg = e[k * BIN : min((k + 1) * BIN, n_min)]
                if len(seg):
                    M[r, k] = float(np.mean(seg))
        mats[mode], layer_axes[mode] = M, Ls
        print(
            f"{mode:>16s} vs {REF}: layers={len(Ls)} seqlen={n_min} "
            f"bins={nb} meanErr={np.nanmean(M):.3f} maxErr={np.nanmax(M):.3f}"
        )

    if not mats:
        raise SystemExit("nothing to plot")
    vmax = max(float(np.nanmax(M)) for M in mats.values()) or 1.0
    n = len(mats)
    fig, axes = plt.subplots(n, 1, figsize=(15, 3.1 * n), squeeze=False)
    for ax, mode in zip(axes[:, 0], mats):
        M, Ls = mats[mode], layer_axes[mode]
        im = ax.imshow(
            M, aspect="auto", origin="lower", cmap="magma",
            vmin=0.0, vmax=vmax, interpolation="nearest",
        )
        ax.set_title(
            f"{mode}  vs  {REF}   —   {MLABEL}"
            f"   meanErr={np.nanmean(M):.3f}  maxErr={np.nanmax(M):.3f}",
            fontsize=10,
        )
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
        # imp overlay: mark the imp (recomputed-fresh) bins at each check-layer row
        # (cyan squares, size ∝ imp fraction in the bin). Reads {mode}_IMP_L{L}.pt.
        # pic_a3/cacheblend only select at L1 (persists downward); oracle re-selects
        # at 1/20/40/60 (markers show the per-layer shift).
        if IMP_OVERLAY:
            _Lrow = {L: r for r, L in enumerate(Ls)}
            for _cl in (1, 20, 40, 60):
                _f = f"{KD}/{mode}_IMP_L{_cl}.pt"
                if not os.path.exists(_f) or _cl not in _Lrow:
                    continue
                _imp = set(int(x) for x in torch.load(_f, map_location="cpu").tolist())
                _xs, _sz = [], []
                for k in range(nb):
                    frac = sum(1 for p in range(k * BIN, (k + 1) * BIN) if p in _imp) / BIN
                    if frac > 0.1:
                        _xs.append(k)
                        _sz.append(6 + 55 * frac)
                if _xs:
                    ax.scatter(_xs, [_Lrow[_cl]] * len(_xs), s=_sz, marker="s",
                               facecolors="none", edgecolors="lime",
                               linewidths=0.9, alpha=0.95)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label=MLABEL)
    axes[-1, 0].set_xlabel("token position")
    fig.suptitle(
        f"PIC per-layer per-token K-error dashboard  (dir={KD}, bin={BIN})",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    plt.savefig(OUT, dpi=110)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
