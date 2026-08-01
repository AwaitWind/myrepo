"""KV error across BOTH feature dim AND token position, per layer.

Extends kdump_kv_headdim.py (which collapsed tokens) — one heatmap PER LAYER:
  x = token position (0..seqlen), y = K latent feature dim (0-575).
  color = per-(token,dim) reuse error |oK - fK| (RMS within the bin).
So you see WHERE (which tokens) AND WHICH feature dims carry the error, per layer.
Dashed horizontal line at dim 512 = kv_lora latent | rope boundary.
Vertical dashed lines = PIC segment boundaries.

reads {KD}: {MODE}_L{L}_c*.pt, {REF}_L{L}_c*.pt  (K, shape (n,1,576))
USAGE: KD=/tmp/kd_obs2 MODE=pic_a3_oracle REF=full_recompute \
       LAYERS=2,20,40,60,77 python kdump_kv_headdim_token.py
ENV: KD MODE REF LAYERS TOKBIN DIMBIN METRIC(euclidean|abs) KV_LORA OUT SEGS
"""
import glob, os, re
import matplotlib; matplotlib.use("Agg")
import numpy as np, torch
import matplotlib.pyplot as plt  # noqa: E402

KD = os.environ.get("KD", "/tmp/kd_obs2")
MODE = os.environ.get("MODE", "pic_a3_oracle")
REF = os.environ.get("REF", "full_recompute")
LAYERS = [int(x) for x in os.environ.get("LAYERS", "2,20,40,60,77").split(",")]
TOKBIN = int(os.environ.get("TOKBIN", "64"))
DIMBIN = int(os.environ.get("DIMBIN", "8"))
METRIC = os.environ.get("METRIC", "relative")   # relative=per-dim normalized (honest); euclidean=abs RMS; abs=mean
MLABEL = {"relative": "per-dim RELATIVE err (|ΔK|/dim-scale)",
          "euclidean": "abs |ΔK| RMS (norm-biased!)",
          "abs": "mean |ΔK|"}.get(METRIC, METRIC)
KV_LORA = int(os.environ.get("KV_LORA", "512"))
OUT = os.environ.get("OUT", f"{KD}/kv_headdim_token_{MODE}.png")
_SEG = "SYS:0-64,C1:64-1216,C2:1216-2304,C3:2304-3392,Q:3392-3456"
SEGS = {}
for _p in os.environ.get("SEGS", _SEG).split(","):
    if ":" in _p and "-" in _p:
        n, r = _p.split(":"); s, e = r.split("-"); SEGS[n] = (int(s), int(e))


def loadK(tag, L):
    cs = glob.glob(f"{KD}/{tag}_L{L}_c*.pt")
    if not cs:
        f = f"{KD}/{tag}_L{L}.pt"
        return torch.load(f, map_location="cpu").flatten(1).float() if os.path.exists(f) else None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()


mats = {}
for L in LAYERS:
    oK, fK = loadK(MODE, L), loadK(REF, L)
    if oK is None or fK is None:
        print(f"[skip] L{L}: missing K"); continue
    n = min(oK.shape[0], fK.shape[0]); D = oK.shape[1]
    diff = (oK[:n] - fK[:n]).abs()                       # (n, D)
    if METRIC == "relative":
        # normalize each feature dim by its own reference magnitude so the
        # large-norm rope dims (k_pe, ||~13||) don't dominate the small-norm
        # content dims (c_KV, ||~0.04|| early) — shows RELATIVE error per dim.
        _rmsd = fK[:n].pow(2).mean(0).sqrt().clamp_min(1e-6)   # (D,)
        diff = diff / _rmsd
    nb = (n + TOKBIN - 1) // TOKBIN
    db = (D + DIMBIN - 1) // DIMBIN
    M = np.full((db, nb), np.nan, np.float32)            # y=dim, x=token
    sq = diff * diff
    for tk in range(nb):
        tsl = slice(tk * TOKBIN, min((tk + 1) * TOKBIN, n))
        for dk in range(db):
            dsl = slice(dk * DIMBIN, min((dk + 1) * DIMBIN, D))
            blk = sq[tsl, dsl]
            if blk.numel():
                M[dk, tk] = float(blk.mean().sqrt() if METRIC in ("euclidean", "relative") else diff[tsl, dsl].mean())
    mats[L] = M
    print(f"L{L}: seqlen={n} lat[:512] mean={np.nanmean(M[:KV_LORA//DIMBIN]):.4f} "
          f"rope[512:] mean={np.nanmean(M[KV_LORA//DIMBIN:]):.4f} max={np.nanmax(M):.4f}")

if not mats:
    raise SystemExit("nothing to plot")
vmax = max(float(np.nanmax(M)) for M in mats.values()) or 1.0
nL = len(mats)
fig, axes = plt.subplots(nL, 1, figsize=(15, 2.7 * nL), squeeze=False)
for ax, (L, M) in zip(axes[:, 0], mats.items()):
    db, nb = M.shape
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.axhline(KV_LORA / DIMBIN - 0.5, color="cyan", lw=1.0, ls="--", alpha=0.9)
    ax.text(nb * 0.998, KV_LORA / DIMBIN, "rope↑", color="cyan", fontsize=8, ha="right", va="bottom")
    for name, (s, _e) in SEGS.items():
        xb = s / TOKBIN - 0.5
        if -0.5 <= xb <= nb:
            ax.axvline(xb, color="lime", lw=0.6, ls="--", alpha=0.6)
            ax.text(max(xb, 0), db - 0.5, name, color="lime", fontsize=7, va="top", ha="left")
    ax.set_title(f"L{L}   {MODE} K vs {REF} K   {MLABEL}   "
                 f"max={np.nanmax(M):.3f}", fontsize=9)
    ax.set_ylabel("K feat dim")
    yt = list(range(0, db, max(1, db // 6)))
    ax.set_yticks(yt); ax.set_yticklabels([str(i * DIMBIN) for i in yt])
    xt = list(range(0, nb, max(1, nb // 12)))
    ax.set_xticks(xt); ax.set_xticklabels([str(i * TOKBIN) for i in xt])
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label=MLABEL)
axes[-1, 0].set_xlabel("token position")
fig.suptitle(f"KV error: feature dim (y, 0-511 content | 512-575 rope) x token (x) per layer  "
             f"—  {MODE} vs {REF}   [{MLABEL}]",
             fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.99))
plt.savefig(OUT, dpi=115)
print("saved", OUT)
