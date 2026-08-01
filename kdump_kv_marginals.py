"""Two marginals of the KV-reuse error, honest metrics:
  A) along the TOKEN dimension   (x=token, y=layer)  = 1-cos(K vs ref) per token
       -> scale-invariant; shows WHERE along the sequence the error is.
  B) along the HEAD/feature dim  (x=K latent dim 0-575, y=layer) = per-dim RELATIVE
       rms = rms_tok|ΔK[:,d]| / rms_tok|ref[:,d]|  (each dim normalized by its own
       magnitude, so the large-norm rope dims 512-575 don't dominate)
       -> shows WHICH feature dims carry the error (content vs rope).
USAGE: KD=/tmp/kd_hh MODE=oracle_hh REF=fullref python kdump_kv_marginals.py
"""
import glob, os, re, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt  # noqa: E402

KD = os.environ.get("KD", "/tmp/kd_hh")
MODE = os.environ.get("MODE", "oracle_hh")
REF = os.environ.get("REF", "fullref")
TOKBIN = int(os.environ.get("TOKBIN", "64"))
DIMBIN = int(os.environ.get("DIMBIN", "8"))
KV_LORA = int(os.environ.get("KV_LORA", "512"))
OUT = os.environ.get("OUT", f"{KD}/kv_marginals_{MODE}.png")
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


def layersK(tag):
    s = set()
    for p in glob.glob(f"{KD}/{tag}_L*.pt"):
        m = re.search(rf"{re.escape(tag)}_L(\d+)(?:_c\d+)?\.pt$", p)
        if m:
            s.add(int(m.group(1)))
    return s


Ls = sorted(layersK(MODE) & layersK(REF))
TOK, DIM = [], []
ntok = ndim = None
for L in Ls:
    o, f = loadK(MODE, L), loadK(REF, L)
    if o is None or f is None:
        TOK.append(None); DIM.append(None); continue
    n = min(o.shape[0], f.shape[0]); D = o.shape[1]
    o, f = o[:n], f[:n]
    # A) per-token 1-cos
    tok = (1 - F.cosine_similarity(o, f, dim=-1)).clamp(0, 2).numpy()
    # B) per-dim relative rms
    dnum = (o - f).pow(2).mean(0).sqrt()                 # (D,)
    dden = f.pow(2).mean(0).sqrt().clamp_min(1e-6)       # (D,)
    dim = (dnum / dden).numpy()
    _nt = (n + TOKBIN - 1) // TOKBIN
    _nd = (D + DIMBIN - 1) // DIMBIN
    ntok = _nt if ntok is None else min(ntok, _nt)
    ndim = _nd if ndim is None else min(ndim, _nd)
    TOK.append(np.array([tok[k*TOKBIN:min((k+1)*TOKBIN, n)].mean() for k in range(_nt)]))
    DIM.append(np.array([dim[k*DIMBIN:min((k+1)*DIMBIN, D)].mean() for k in range(_nd)]))

rows = [i for i, t in enumerate(TOK) if t is not None]
Lr = [Ls[i] for i in rows]
A = np.stack([TOK[i][:ntok] for i in rows])
B = np.stack([DIM[i][:ndim] for i in rows])

fig, axes = plt.subplots(2, 1, figsize=(15, 8))
# A: token marginal
imA = axes[0].imshow(A, aspect="auto", origin="lower", cmap="magma", vmin=0,
                     vmax=np.nanpercentile(A, 99.5), interpolation="nearest")
axes[0].set_title(f"A)  KV error along TOKEN   (1-cos per token; bright = that token's cached K is wrong)   "
                  f"{MODE} vs {REF}", fontsize=11, loc="left")
axes[0].set_ylabel("layer"); axes[0].set_xlabel("token position")
yt = list(range(0, len(Lr), max(1, len(Lr)//10)))
axes[0].set_yticks(yt); axes[0].set_yticklabels([Lr[i] for i in yt])
xt = list(range(0, ntok, max(1, ntok//12)))
axes[0].set_xticks(xt); axes[0].set_xticklabels([str(i*TOKBIN) for i in xt])
for name, (s, _e) in SEGS.items():
    xb = s/TOKBIN - 0.5
    axes[0].axvline(xb, color="cyan", lw=0.7, ls="--", alpha=0.7)
    axes[0].text(max(xb, 0), len(Lr)-0.5, name, color="cyan", fontsize=8, va="top")
fig.colorbar(imA, ax=axes[0], fraction=0.02, pad=0.01, label="1-cos")
# B: head/feature-dim marginal
imB = axes[1].imshow(B, aspect="auto", origin="lower", cmap="magma", vmin=0,
                     vmax=np.nanpercentile(B, 99.5), interpolation="nearest")
axes[1].set_title("B)  KV error along HEAD/FEATURE dim   (per-dim RELATIVE rms; normalized so rope's big norm "
                  "doesn't cheat)   0-511 = content c_KV | 512-575 = rope k_pe", fontsize=11, loc="left")
axes[1].set_ylabel("layer"); axes[1].set_xlabel("K latent feature dim")
axes[1].set_yticks(yt); axes[1].set_yticklabels([Lr[i] for i in yt])
axes[1].axvline(KV_LORA/DIMBIN - 0.5, color="cyan", lw=1.2, ls="--")
axes[1].text(KV_LORA/DIMBIN, len(Lr)-0.5, "rope->", color="cyan", fontsize=9, va="top")
axes[1].text(0, len(Lr)-0.5, "content (c_KV)", color="cyan", fontsize=9, va="top")
xtd = list(range(0, ndim, max(1, ndim//12)))
axes[1].set_xticks(xtd); axes[1].set_xticklabels([str(i*DIMBIN) for i in xtd])
fig.colorbar(imB, ax=axes[1], fraction=0.02, pad=0.01, label="relative rms")
fig.suptitle(f"KV-reuse error — TOKEN marginal (A) & HEAD/feature-dim marginal (B)   {MODE} vs {REF}",
             fontsize=12)
plt.tight_layout(rect=(0, 0, 1, 0.97))
plt.savefig(OUT, dpi=120)
print("saved", OUT)
print(f"token-marginal: mean={A.mean():.3f} max={A.max():.3f}")
print(f"dim-marginal(relative): content[:512] mean={B[:, :KV_LORA//DIMBIN].mean():.3f}  "
      f"rope[512:] mean={B[:, KV_LORA//DIMBIN:].mean():.3f}")
