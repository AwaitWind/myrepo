"""attention-deviation + attention-weighted-KV-deviation 观测台 for PIC modes.

Complements kdump_heatmap.py (raw per-key KV deviation) with two views that
account for what attention actually READS:

  PLOT 1 — attention-weighted KV deviation  (x=key position, y=layer):
    per (layer, key k):  attn_mass_on_k  ×  (1 - cos(K_mode[k], K_ref[k]))
    where attn_mass_on_k = how much the QUERY segment attends to key k (mode's
    own attention). Bright = key is both corrupted AND read by the answer.
    Attention is reconstructed from the dumped summed-Q · full-K using the A3
    merged-head formula (dense proxy — NOT DSA-sparse; slightly over-counts).

  PLOT 2 — query-region attention deviation  (x=query token, y=layer):
    per (layer, query token q):  1 - cos(attn_out_mode[q], attn_out_ref[q])
    i.e. how wrong the answer-generating positions' attention OUTPUT is, per
    layer. (Query is computed by every mode, so this is dense & comparable.)

INPUT — produced by the probes (deepseek_v2 + forward_mla) under:
    SGLANG_PIC_KDUMP_DIR=/tmp/kd_obs SGLANG_PIC_KDUMP_ALL=1 python quick_test_online.py ...
  Files per mode/layer:
    {tag}_L{L}_c{c}.pt        full K, (seq_len,1,576)         [KV pool]
    {tag}_Q_L{L}.pt           {q:(Nq,576), scale, qpos:(Nq,)} [query summed-Q]
    {tag}_A_L{L}.pt           query attn output, (Nq,hidden)  [attn output]

USAGE:
    KD=/tmp/kd_obs REF=pic_a3 python kdump_attn.py
ENV: KD, REF (default pic_a3), MODES (default pic,pic_cacheblend,pic_a3_oracle),
     BIN (key-axis bin, default 64), OUT_W / OUT_Q (png paths), SEGS.
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

KD = os.environ.get("KD", "/tmp/kd_obs")
REF = os.environ.get("REF", "pic_a3")
BIN = int(os.environ.get("BIN", "64"))
OUT_W = os.environ.get("OUT_W", f"{KD}/kdump_attn_weighted_kv.png")
OUT_Q = os.environ.get("OUT_Q", f"{KD}/kdump_attn_query_dev.png")
OUT_H = os.environ.get("OUT_H", f"{KD}/kdump_hidden_dev.png")
MODES = os.environ.get("MODES", "pic,pic_cacheblend,pic_a3_oracle").split(",")
_SEG_DEFAULT = "SYS:0-64,C1:64-1216,C2:1216-2304,C3:2304-3392,Q:3392-3456"
SEGS = {}
for _part in os.environ.get("SEGS", _SEG_DEFAULT).split(","):
    if ":" in _part and "-" in _part:
        _n, _r = _part.split(":")
        _s, _e = _r.split("-")
        SEGS[_n] = (int(_s), int(_e))


def _load_K(tag, L):
    cs = glob.glob(f"{KD}/{tag}_L{L}_c*.pt")
    if not cs:
        f = f"{KD}/{tag}_L{L}.pt"
        return torch.load(f, map_location="cpu").flatten(1).float() if os.path.exists(f) else None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()


def _load_Q(tag, L):
    f = f"{KD}/{tag}_Q_L{L}.pt"
    if not os.path.exists(f):
        return None
    d = torch.load(f, map_location="cpu")
    return d["q"].float(), float(d["scale"]), d["qpos"].long()


def _load_A(tag, L):
    f = f"{KD}/{tag}_A_L{L}.pt"
    return torch.load(f, map_location="cpu").flatten(1).float() if os.path.exists(f) else None


def _load_qtok(tag, L, kind):
    """kind='A' attn output, 'H' hidden-state residual stream. (Nq, dim)."""
    f = f"{KD}/{tag}_{kind}_L{L}.pt"
    return torch.load(f, map_location="cpu").flatten(1).float() if os.path.exists(f) else None


def _layers(tag, kind):
    pat = {
        "K": f"{tag}_L*.pt", "Q": f"{tag}_Q_L*.pt",
        "A": f"{tag}_A_L*.pt", "H": f"{tag}_H_L*.pt",
    }[kind]
    rgx = {
        "K": rf"{re.escape(tag)}_L(\d+)(?:_c\d+)?\.pt$",
        "Q": rf"{re.escape(tag)}_Q_L(\d+)\.pt$",
        "A": rf"{re.escape(tag)}_A_L(\d+)\.pt$",
        "H": rf"{re.escape(tag)}_H_L(\d+)\.pt$",
    }[kind]
    ls = set()
    for p in glob.glob(f"{KD}/{pat}"):
        m = re.search(rgx, p)
        if m:
            ls.add(int(m.group(1)))
    return sorted(ls)


def _dev(a, b):
    n = min(a.shape[0], b.shape[0])
    return (1 - F.cosine_similarity(a[:n], b[:n], dim=-1)).clamp(0, 2), n


def _grid(mats, layer_axes, title, out, xlabel, xbin, annotate_segs):
    if not mats:
        print(f"[skip] {title}: nothing to plot")
        return
    vmax = max(float(np.nanmax(M)) for M in mats.values()) or 1.0
    n = len(mats)
    fig, axes = plt.subplots(n, 1, figsize=(15, 3.1 * n), squeeze=False)
    for ax, mode in zip(axes[:, 0], mats):
        M, Ls = mats[mode], layer_axes[mode]
        im = ax.imshow(M, aspect="auto", origin="lower", cmap="magma",
                       vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{mode}  vs  {REF}   ({title})   mean={np.nanmean(M):.4f} max={np.nanmax(M):.4f}",
                     fontsize=10)
        ax.set_ylabel("layer")
        yt = list(range(0, len(Ls), max(1, len(Ls) // 12)))
        ax.set_yticks(yt)
        ax.set_yticklabels([Ls[i] for i in yt])
        nb = M.shape[1]
        xt = list(range(0, nb, max(1, nb // 12)))
        ax.set_xticks(xt)
        ax.set_xticklabels([str(i * xbin) for i in xt])
        if annotate_segs:
            for name, (s, _e) in SEGS.items():
                xb = s / xbin - 0.5
                if -0.5 <= xb <= nb:
                    ax.axvline(xb, color="cyan", lw=0.7, ls="--", alpha=0.7)
                    ax.text(max(xb, 0), len(Ls) - 0.5, name, color="cyan",
                            fontsize=8, va="top", ha="left")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[-1, 0].set_xlabel(xlabel)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.99))
    plt.savefig(out, dpi=110)
    print(f"saved {out}")


def plot_weighted_kv():
    ref_Ls = set(_layers(REF, "K"))
    mats, layer_axes = {}, {}
    for mode in MODES:
        Ls = sorted(set(_layers(mode, "K")) & set(_layers(mode, "Q")) & ref_Ls)
        if not Ls:
            print(f"[skip weighted] {mode}: missing K/Q/ref layers")
            continue
        rows, nmin, nb = [], None, None
        for L in Ls:
            Km, Kr, Q = _load_K(mode, L), _load_K(REF, L), _load_Q(mode, L)
            if Km is None or Kr is None or Q is None:
                rows.append(None)
                continue
            dev, n = _dev(Km, Kr)                       # (n,) per-key KV deviation
            q, scale, qpos = Q
            n = min(n, Km.shape[0])
            score = (q @ Km[:n].T) * scale             # (Nq, n)
            kidx = torch.arange(n)
            score = score.masked_fill(kidx[None, :] > qpos[:, None], float("-inf"))
            w = F.softmax(score, dim=1)                 # (Nq, n)
            mass = w.sum(0) / q.shape[0]                # (n,) avg attn prob per key
            weighted = (mass * dev[:n]).numpy()
            rows.append(weighted)
            nmin = n if nmin is None else min(nmin, n)
        if nmin is None:
            continue
        nb = (nmin + BIN - 1) // BIN
        M = np.full((len(Ls), nb), np.nan, np.float32)
        for r, e in enumerate(rows):
            if e is None:
                continue
            for k in range(nb):
                seg = e[k * BIN:min((k + 1) * BIN, nmin)]
                if len(seg):
                    M[r, k] = float(np.mean(seg))
        mats[mode], layer_axes[mode] = M, Ls
        print(f"{mode:>16s} weighted-KV: layers={len(Ls)} keys={nmin} "
              f"mean={np.nanmean(M):.4f} max={np.nanmax(M):.4f}")
    _grid(mats, layer_axes, "attention-weighted KV deviation  (attn_mass x (1-cosK))",
          OUT_W, "key position", BIN, True)


def plot_qtok_dev(kind, out, title, label):
    ref_Ls = set(_layers(REF, kind))
    mats, layer_axes = {}, {}
    for mode in MODES:
        Ls = sorted(set(_layers(mode, kind)) & ref_Ls)
        if not Ls:
            print(f"[skip {label}] {mode}: missing {kind} layers")
            continue
        rows, nq = [], None
        for L in Ls:
            Am, Ar = _load_qtok(mode, L, kind), _load_qtok(REF, L, kind)
            if Am is None or Ar is None:
                rows.append(None)
                continue
            d, n = _dev(Am, Ar)
            rows.append(d.numpy())
            nq = n if nq is None else min(nq, n)
        if nq is None:
            continue
        M = np.full((len(Ls), nq), np.nan, np.float32)
        for r, e in enumerate(rows):
            if e is not None:
                M[r, :nq] = e[:nq]
        mats[mode], layer_axes[mode] = M, Ls
        _pl = [f"L{Ls[i]}={np.nanmean(M[i]):.3f}"
               for i in range(len(Ls)) if Ls[i] in (1, 20, 40, 60, 77)]
        print(f"{mode:>16s} {label}: layers={len(Ls)} qtok={nq} "
              f"mean={np.nanmean(M):.4f}  checkL: {' '.join(_pl)}")
    _grid(mats, layer_axes, title, out, "query token", 1, False)


if __name__ == "__main__":
    plot_weighted_kv()
    plot_qtok_dev("A", OUT_Q,
                  "query-region attention-output deviation  (1-cos(attn_out))",
                  "query-attn-dev")
    plot_qtok_dev("H", OUT_H,
                  "query-region hidden-state deviation  (1-cos(residual stream))",
                  "hidden-dev")
