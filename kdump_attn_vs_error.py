"""One figure: the KV-reuse error lives where attention does NOT look.
4 panels (x=token): 1 error, 2 attention, 3 TRUE impact (per-token a*err then binned),
4 zoom of the SYS/front region per-token (attention spike vs error) to bust the
"SYS bin looks bright" binning artifact — attention is on token-0 (clean), error is
on tokens 1-63 (unattended); a 64-token bin blurs them together.
USAGE: KD=/tmp/kd_hh MODE=oracle_hh REF=fullref ZOOM_LAYER=60 python kdump_attn_vs_error.py
"""
import glob, os, re, torch
import torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt  # noqa: E402

KD = os.environ.get("KD", "/tmp/kd_hh")
MODE = os.environ.get("MODE", "oracle_hh")
REF = os.environ.get("REF", "fullref")
BIN = int(os.environ.get("BIN", "64"))
ZOOM_LAYER = int(os.environ.get("ZOOM_LAYER", "60"))
ZOOM_END = int(os.environ.get("ZOOM_END", "256"))
OUT = os.environ.get("OUT", f"{KD}/attn_vs_error_{MODE}.png")
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


def loadQ(tag, L):
    f = f"{KD}/{tag}_Q_L{L}.pt"
    if not os.path.exists(f):
        return None
    d = torch.load(f, map_location="cpu")
    return d["q"].float(), float(d["scale"]), d["qpos"].long()


def attn_mass(Q, K):
    q, scale, qpos = Q
    n = K.shape[0]
    score = (q @ K.T) * scale
    kidx = torch.arange(n)
    score = score.masked_fill(kidx[None, :] > qpos[:, None], float("-inf"))
    return F.softmax(score, dim=1).sum(0)


def layersK(tag):
    s = set()
    for p in glob.glob(f"{KD}/{tag}_L*.pt"):
        m = re.search(rf"{re.escape(tag)}_L(\d+)(?:_c\d+)?\.pt$", p)
        if m:
            s.add(int(m.group(1)))
    return s


Ls = sorted(layersK(MODE) & layersK(REF))
nb = None
ERR, ATTN, IMP = [], [], []
zoom = None
for L in Ls:
    Km, Kf, Q = loadK(MODE, L), loadK(REF, L), loadQ(MODE, L)
    if Km is None or Kf is None or Q is None:
        ERR.append(None); ATTN.append(None); IMP.append(None); continue
    n = min(Km.shape[0], Kf.shape[0])
    err = (1 - F.cosine_similarity(Km[:n], Kf[:n], dim=-1)).clamp(0, 2).numpy()
    a = attn_mass(Q, Km)[:n]
    a = (a / a.sum()).numpy()
    imp = a * err                                       # TRUE per-token impact
    _nb = (n + BIN - 1) // BIN
    nb = _nb if nb is None else min(nb, _nb)
    ERR.append(np.array([err[k*BIN:min((k+1)*BIN, n)].mean() for k in range(_nb)]))
    ATTN.append(np.array([a[k*BIN:min((k+1)*BIN, n)].sum() for k in range(_nb)]))
    IMP.append(np.array([imp[k*BIN:min((k+1)*BIN, n)].sum() for k in range(_nb)]))  # sum, not mean*sum
    if int(L) == ZOOM_LAYER:
        zoom = (a[:ZOOM_END].copy(), err[:ZOOM_END].copy())

rows = [i for i, e in enumerate(ERR) if e is not None]
Lrows = [Ls[i] for i in rows]
E = np.stack([ERR[i][:nb] for i in rows])
A = np.stack([ATTN[i][:nb] for i in rows])
P = np.stack([IMP[i][:nb] for i in rows])
print(f"error   max={E.max():.3f} mean={E.mean():.3f}")
print(f"impact(TRUE per-token) max={P.max():.5f} mean={P.mean():.6f}")

fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.9], hspace=0.35)
axes = [fig.add_subplot(gs[i]) for i in range(3)]
panels = [
    (E, "magma", E.max(), "1)  WHERE the reuse ERROR is   (bright = K differs from full recompute)"),
    (A, "magma", np.percentile(A, 99.5), "2)  WHERE ATTENTION goes   (bright = answer token attends here; BLACK = ignored)"),
    (P, "magma", P.max(), "3)  TRUE IMPACT = per-token(attention x error) then summed   (bright = hurts output)"),
]
for ax, (M, cmap, vmax, title) in zip(axes, panels):
    im = ax.imshow(M, aspect="auto", origin="lower", cmap=cmap, vmin=0,
                   vmax=(vmax or 1.0), interpolation="nearest")
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_ylabel("layer")
    yt = list(range(0, len(Lrows), max(1, len(Lrows)//10)))
    ax.set_yticks(yt); ax.set_yticklabels([Lrows[i] for i in yt])
    for name, (s, _e) in SEGS.items():
        xb = s / BIN - 0.5
        ax.axvline(xb, color="white", lw=0.8, ls="--", alpha=0.6)
        ax.text(max(xb, 0), len(Lrows)-0.5, name, color="white", fontsize=9, va="top", ha="left")
    ax.set_xlim(-0.5, nb-0.5)
    xt = list(range(0, nb, max(1, nb//12)))
    ax.set_xticks(xt); ax.set_xticklabels([str(i*BIN) for i in xt])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
axes[2].set_xlabel("token position")

# Panel 4: per-token zoom of the front (SYS + C1 head) at ZOOM_LAYER
axz = fig.add_subplot(gs[3])
if zoom is not None:
    az, ez = zoom
    x = np.arange(len(az))
    axz.bar(x, az, width=1.0, color="#2c7fb8", label="attention mass (per token)")
    axz.set_ylabel("attention mass", color="#2c7fb8")
    axz.tick_params(axis="y", labelcolor="#2c7fb8")
    axz.set_xlim(-0.5, len(az)-0.5)
    ax2 = axz.twinx()
    ax2.plot(x, ez, color="#d7301f", lw=1.6, label="K error (1-cos)")
    ax2.set_ylabel("K error (1-cos)", color="#d7301f")
    ax2.tick_params(axis="y", labelcolor="#d7301f")
    for name, (s, _e) in SEGS.items():
        if s < len(az):
            axz.axvline(s-0.5, color="gray", ls="--", lw=0.7)
            axz.text(s, axz.get_ylim()[1]*0.9, name, fontsize=9, color="gray")
    axz.set_title(f"4)  ZOOM tokens 0-{ZOOM_END} at layer {ZOOM_LAYER}:  "
                  f"attention SPIKES on token 0 (error~0);  error is on tokens 1-63 (attention~0)",
                  fontsize=11, loc="left")
    axz.set_xlabel("token position")
fig.suptitle(f"The KV error lives where attention does NOT look   ({MODE} vs {REF})\n"
             f"TRUE impact max={P.max():.4f}  vs  raw-error max={E.max():.2f}   "
             f"(SYS looked bright only because a 64-tok bin blurs token-0-attention with token-1..63-error)",
             fontsize=12)
plt.savefig(OUT, dpi=120, bbox_inches="tight")
print("saved", OUT)
