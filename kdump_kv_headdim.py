"""KV error across the feature/head dimension: oracle K vs full_recompute K.

The MLA cached key is 576-dim = [kv_lora_rank(512) | qk_rope_head_dim(64)]. This
shows WHICH feature dimensions the reuse error lives in, per layer:
  METRIC=euclidean → per (layer, dim d): RMS over positions of (oK-fK)[:,d]
  METRIC=cos       → per (layer, dim d): 1 - cos over positions of oK[:,d] vs fK[:,d]
x = K latent dim (0-575; dashed line at 512 = latent|rope boundary), y = layer.

reads kd_clean: {MODE}_L{L}_c*.pt, {REF}_L{L}_c*.pt  (K, shape (n,1,576))
USAGE: KD=/tmp/kd_clean MODE=pic_a3_oracle METRIC=euclidean python kdump_kv_headdim.py
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
MODE = os.environ.get("MODE", "pic_a3_oracle")
REF = os.environ.get("REF", "full_recompute")
METRIC = os.environ.get("METRIC", "cos")
MLABEL = {"cos": "1-cos (per-dim, over pos)",
          "euclidean": "RMS L2 (per-dim, over pos)"}.get(METRIC, METRIC)
DIMBIN = int(os.environ.get("DIMBIN", "8"))
NPOS = int(os.environ.get("NPOS", "3392"))     # use doc positions (reuse region)
KV_LORA = int(os.environ.get("KV_LORA", "512"))
OUT = os.environ.get("OUT", f"{KD}/kv_headdim_{METRIC}.png")


def _loadK(tag, L):
    cs = glob.glob(f"{KD}/{tag}_L{L}_c*.pt")
    if not cs:
        return None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()   # (n, 576)


def _layers(tag):
    s = set()
    for p in glob.glob(f"{KD}/{tag}_L*.pt"):
        m = re.search(rf"{re.escape(tag)}_L(\d+)(?:_c\d+)?\.pt$", p)
        if m:
            s.add(int(m.group(1)))
    return s


Ls = sorted(_layers(MODE) & _layers(REF))
rows, D = [], None
for L in Ls:
    oK, fK = _loadK(MODE, L), _loadK(REF, L)
    if oK is None or fK is None:
        rows.append(None)
        continue
    n = min(oK.shape[0], fK.shape[0], NPOS)
    oK, fK = oK[:n], fK[:n]
    D = oK.shape[1]
    if METRIC == "euclidean":
        perdim = ((oK - fK) ** 2).mean(dim=0).sqrt()          # (D,) RMS over positions
    else:
        perdim = 1 - F.cosine_similarity(oK.t(), fK.t(), dim=-1)  # (D,) per-dim cos over pos
    rows.append(perdim.numpy())
    print(f"L{L}: lat[0:512] mean={np.nanmean(rows[-1][:KV_LORA]):.3f}  "
          f"rope[512:576] mean={np.nanmean(rows[-1][KV_LORA:]):.3f}")

valid = [(L, d) for L, d in zip(Ls, rows) if d is not None]
if not valid or D is None:
    raise SystemExit("nothing to plot (need K for MODE and REF in kd_clean)")
nb = (D + DIMBIN - 1) // DIMBIN
M = np.full((len(valid), nb), np.nan, np.float32)
for r, (L, d) in enumerate(valid):
    for k in range(nb):
        seg = d[k * DIMBIN:min((k + 1) * DIMBIN, D)]
        if len(seg):
            M[r, k] = float(np.mean(seg))

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
ax.set_xticklabels([str(i * DIMBIN) for i in xt])
# latent | rope boundary
ax.axvline(KV_LORA / DIMBIN - 0.5, color="cyan", lw=1.0, ls="--", alpha=0.9)
ax.text(KV_LORA / DIMBIN, len(valid) - 0.5, "rope→", color="cyan", fontsize=10, va="top")
ax.text(0, len(valid) - 0.5, "kv_lora latent", color="cyan", fontsize=10, va="top")
ax.set_xlabel("K latent feature dim (0-511 kv_lora | 512-575 rope)")
ax.set_ylabel("layer")
ax.set_title(f"KV error across feature dim: {MODE} K vs {REF} K  ({MLABEL})")
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.01, label=MLABEL)
plt.tight_layout()
plt.savefig(OUT, dpi=120)
print("saved", OUT)
