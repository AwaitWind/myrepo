"""imp-selection map: which token positions each mode recomputes fresh (imp).

pic_a3 (real Q·K), pic_cacheblend (V-diff), pic_a3_oracle (isolated-store Q·K,
re-picked at EVERY check layer). Shows WHERE each mode spends its ~15% recompute
budget, and how oracle's per-layer re-selection shifts it.

Files: {tag}_IMP_L{L}.pt = LongTensor of absolute imp positions.
Rows: pic_a3@L1, cacheblend@L1, oracle@L1/L20/L40/L60.
Color = fraction of positions in each 64-bin selected as imp.

USAGE: KD=/tmp/kd_imp python kdump_imp.py
"""
import os

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KD = os.environ.get("KD", "/tmp/kd_imp")
BIN = int(os.environ.get("BIN", "64"))
NPOS = int(os.environ.get("NPOS", "3456"))
OUT = os.environ.get("OUT", f"{KD}/kdump_imp.png")
_SEG_DEFAULT = "SYS:0-64,C1:64-1216,C2:1216-2304,C3:2304-3392,Q:3392-3456"
SEGS = {}
for _p in os.environ.get("SEGS", _SEG_DEFAULT).split(","):
    if ":" in _p and "-" in _p:
        n, r = _p.split(":")
        s, e = r.split("-")
        SEGS[n] = (int(s), int(e))

ROWS = [
    ("pic_a3", 1, "pic_a3 @L1"),
    ("pic_cacheblend", 1, "cacheblend @L1"),
    ("pic_a3_oracle", 1, "oracle @L1"),
    ("pic_a3_oracle", 20, "oracle @L20"),
    ("pic_a3_oracle", 40, "oracle @L40"),
    ("pic_a3_oracle", 60, "oracle @L60"),
]


def _load(tag, L):
    f = f"{KD}/{tag}_IMP_L{L}.pt"
    if not os.path.exists(f):
        return None
    return torch.load(f, map_location="cpu").long().tolist()


nb = (NPOS + BIN - 1) // BIN
labels, M = [], []
for tag, L, lab in ROWS:
    idx = _load(tag, L)
    if idx is None:
        print(f"[skip] {lab}: no {tag}_IMP_L{L}.pt")
        continue
    imp_set = set(i for i in idx if 0 <= i < NPOS)
    row = np.zeros(nb, np.float32)
    for k in range(nb):
        s, e = k * BIN, min((k + 1) * BIN, NPOS)
        row[k] = sum(1 for p in range(s, e) if p in imp_set) / max(1, e - s)
    M.append(row)
    labels.append(lab)
    sm = {
        name: sum(1 for p in range(s, min(e, NPOS)) if p in imp_set) / max(1, min(e, NPOS) - s)
        for name, (s, e) in SEGS.items()
    }
    print(f"{lab:>16s}: n_imp={len(imp_set):5d}  "
          + "  ".join(f"{n}={sm[n]:.0%}" for n in SEGS))

if not M:
    raise SystemExit("no IMP dumps found")
M = np.array(M)
fig, ax = plt.subplots(figsize=(15, 1.6 + 0.6 * len(labels)))
im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1,
               interpolation="nearest")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
xt = list(range(0, nb, max(1, nb // 12)))
ax.set_xticks(xt)
ax.set_xticklabels([str(i * BIN) for i in xt])
for name, (s, e) in SEGS.items():
    ax.axvline(s / BIN - 0.5, color="white", lw=0.8, ls="--", alpha=0.7)
    ax.text(s / BIN, len(labels) - 0.5, name, color="white", fontsize=10, va="top")
ax.set_xlabel("token position")
ax.set_ylabel("mode @ layer")
ax.set_title("imp-selection map: fraction of each 64-token bin recomputed (imp)")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, label="imp fraction")
plt.tight_layout()
plt.savefig(OUT, dpi=120)
print("saved", OUT)
