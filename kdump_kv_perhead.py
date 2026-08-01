"""Head-level KV error: per-head decompressed K, oracle vs full_recompute.

MLA caches a SHARED latent (no native per-head K). We decompress it to per-head
k_nope via w_kc[h] (192x512) for each of the 64 heads (w_kc dumped across all TP
ranks and concatenated), then compare oracle-K vs full-K PER HEAD. Shows how the
reuse error differs BETWEEN heads. (kv_a_layernorm skipped — head-RELATIVE
differences are set by w_kc[h] applied to a common latent, so the ranking holds.)

METRIC=cos (1-cos) | euclidean.  Plot: y=head (0-63), x=layer (0-77).

w_kc : {WKC}/wkc_L{L}_r{r}.pt  {"w_kc":(H_local,192,512)[, "w_scale"/"w_scale_k"]}
c_KV : {KD}/{tag}_L{L}_c*.pt   (K (n,1,576); shared latent = [:, :512])
USAGE: KD=/tmp/kd_clean WKC=/tmp/kd_wkc MODE=pic_a3_oracle METRIC=cos python kdump_kv_perhead.py
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
WKC = os.environ.get("WKC", "/tmp/kd_wkc")
MODE = os.environ.get("MODE", "pic_a3_oracle")
REF = os.environ.get("REF", "full_recompute")
METRIC = os.environ.get("METRIC", "cos")
MLABEL = {"cos": "1-cos (per head)", "euclidean": "L2 euclidean (per head)"}.get(METRIC, METRIC)
KVL = int(os.environ.get("KV_LORA", "512"))
NPOS = int(os.environ.get("NPOS", "3392"))
SAMPLE = int(os.environ.get("SAMPLE", "768"))       # subsample doc positions for speed
OUT = os.environ.get("OUT", f"{KD}/kv_perhead_{METRIC}.png")


def _loadK(tag, L):
    cs = glob.glob(f"{KD}/{tag}_L{L}_c*.pt")
    if not cs:
        return None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()


def _load_wkc(L):
    parts = []
    r = 0
    while True:
        f = f"{WKC}/wkc_L{L}_r{r}.pt"
        if not os.path.exists(f):
            break
        d = torch.load(f, map_location="cpu")
        w = d["w_kc"].float()                         # (H_local, 192, 512) expected
        for sk in ("w_scale", "w_scale_k"):           # dequant if fp8 scale present
            if sk in d:
                s = d[sk].float()
                try:
                    w = w * s if s.shape == w.shape else w * s.reshape(
                        [w.shape[0]] + [1] * (w.dim() - 1))
                except Exception:
                    pass
        parts.append(w)
        r += 1
    return torch.cat(parts, dim=0) if parts else None   # (H, 192, 512)


def _layersK(tag):
    s = set()
    for p in glob.glob(f"{KD}/{tag}_L*.pt"):
        m = re.search(rf"{re.escape(tag)}_L(\d+)(?:_c\d+)?\.pt$", p)
        if m:
            s.add(int(m.group(1)))
    return s


Ls = sorted(_layersK(MODE) & _layersK(REF))
rows, Hn = [], None
for L in Ls:
    wkc = _load_wkc(L)
    oK, fK = _loadK(MODE, L), _loadK(REF, L)
    if wkc is None or oK is None or fK is None:
        rows.append(None)
        continue
    n = min(oK.shape[0], fK.shape[0], NPOS)
    idx = torch.arange(0, n, max(1, n // SAMPLE))[:SAMPLE]
    cKV_o, cKV_f = oK[idx, :KVL], fK[idx, :KVL]         # (m, 512)
    ko = torch.einsum("nl,hjl->nhj", cKV_o, wkc)        # (m, H, 192)
    kf = torch.einsum("nl,hjl->nhj", cKV_f, wkc)
    Hn = wkc.shape[0]
    if METRIC == "euclidean":
        err = (ko - kf).norm(dim=-1).mean(dim=0)         # (H,)
    else:
        err = (1 - F.cosine_similarity(ko, kf, dim=-1)).mean(dim=0)
    rows.append(err.numpy())
    print(f"L{L}: heads={Hn} err[min={float(np.min(rows[-1])):.3f} "
          f"max={float(np.max(rows[-1])):.3f} spread={float(np.max(rows[-1])-np.min(rows[-1])):.3f}]")

valid = [(L, d) for L, d in zip(Ls, rows) if d is not None]
if not valid or Hn is None:
    raise SystemExit("nothing to plot (need w_kc + K for MODE and REF)")
M = np.full((Hn, len(valid)), np.nan, np.float32)       # (head, layer)
for c, (L, d) in enumerate(valid):
    M[:len(d), c] = d

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(M, aspect="auto", origin="lower", cmap="magma", vmin=0,
               interpolation="nearest")
ax.set_xticks(list(range(0, len(valid), max(1, len(valid) // 16))))
ax.set_xticklabels([str(valid[i][0]) for i in range(0, len(valid), max(1, len(valid) // 16))])
ax.set_yticks(list(range(0, Hn, max(1, Hn // 16))))
ax.set_yticklabels([str(h) for h in range(0, Hn, max(1, Hn // 16))])
ax.set_xlabel("layer")
ax.set_ylabel("head")
ax.set_title(f"head-level KV error: {MODE} K vs {REF} K  ({MLABEL})")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label=MLABEL)
plt.tight_layout()
plt.savefig(OUT, dpi=120)
print("saved", OUT)
