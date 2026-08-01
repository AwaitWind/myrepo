"""Per-HEAD K-cache error: deep-recompute (oracle_hh) vs full_recompute.
MLA caches a SHARED 576-d latent; per-head K is reconstructed by up-projecting
the cached content latent c_KV (dims 0-511) with W_UK (kv_b_proj, FP8 block-dequant),
then appending the shared RoPE key k_pe (dims 512-575).  Per head h:
    K_h = [ c_KV @ W_UK[h]^T (192) | k_pe (64) ]   -> 256-d
Error = 1 - cos(K_h^oracle, K_h^full) per (token, layer).  One subplot per head
(x=token, y=layer), 8x8 grid of the 64 heads.
USAGE: KD=/tmp/kd_hh MODE=oracle_hh REF=fullref python kdump_perhead_k.py
"""
import json, glob, os, re, torch
import torch.nn.functional as F
from safetensors import safe_open
import matplotlib; matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt  # noqa: E402

KD = os.environ.get("KD", "/tmp/kd_hh")
MODE = os.environ.get("MODE", "oracle_hh")
REF = os.environ.get("REF", "fullref")
MODEL = os.environ.get("MODEL", "/workspace/models/GLM-5.2-FP8")
BIN = int(os.environ.get("BIN", "64"))
OUT = os.environ.get("OUT", f"{KD}/perhead_k_{MODE}.png")
KV_LORA, N_HEADS, QK_NOPE, V_HEAD, QK_ROPE = 512, 64, 192, 256, 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"
IDX = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]
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


def dequant_block(w_fp8, scale_inv, block=128):
    M, N = w_fp8.shape
    s = scale_inv.repeat_interleave(block, 0)[:M].repeat_interleave(block, 1)[:, :N]
    return w_fp8.float() * s


def load_wuk(L):
    key = f"model.layers.{L}.self_attn.kv_b_proj.weight"
    with safe_open(f"{MODEL}/{IDX[key]}", framework="pt") as f:
        w = f.get_tensor(key); s = f.get_tensor(key + "_scale_inv")
    wdeq = dequant_block(w, s)                              # (28672, 512)
    wdeq = wdeq.view(N_HEADS, QK_NOPE + V_HEAD, KV_LORA)[:, :QK_NOPE, :]  # (64,192,512)
    return wdeq


def layersK(tag):
    s = set()
    for p in glob.glob(f"{KD}/{tag}_L*.pt"):
        m = re.search(rf"{re.escape(tag)}_L(\d+)(?:_c\d+)?\.pt$", p)
        if m:
            s.add(int(m.group(1)))
    return s


Ls = sorted(layersK(MODE) & layersK(REF))
nb = None
ERR = None
for li, L in enumerate(Ls):
    ko, kf = loadK(MODE, L), loadK(REF, L)
    if ko is None or kf is None:
        continue
    n = min(ko.shape[0], kf.shape[0])
    ckv_o = ko[:n, :KV_LORA].to(DEV); pe_o = ko[:n, KV_LORA:].to(DEV)
    ckv_f = kf[:n, :KV_LORA].to(DEV); pe_f = kf[:n, KV_LORA:].to(DEV)
    wuk = load_wuk(L).to(DEV)                                # (64,192,512)
    kn_o = torch.einsum("nd,hpd->nhp", ckv_o, wuk)           # (n,64,192)
    kn_f = torch.einsum("nd,hpd->nhp", ckv_f, wuk)
    pe_oe = pe_o.unsqueeze(1).expand(-1, N_HEADS, -1)        # (n,64,64)
    pe_fe = pe_f.unsqueeze(1).expand(-1, N_HEADS, -1)
    Ko = torch.cat([kn_o, pe_oe], -1)                        # (n,64,256)
    Kf = torch.cat([kn_f, pe_fe], -1)
    cos = F.cosine_similarity(Ko, Kf, dim=-1)                # (n,64)
    err = (1 - cos).clamp(0, 2).cpu().numpy()               # (n,64)
    _nb = (n + BIN - 1) // BIN
    if ERR is None:
        nb = _nb; ERR = np.full((N_HEADS, len(Ls), nb), np.nan, np.float32)
    nb = min(nb, _nb)
    for b in range(_nb):
        seg = err[b * BIN:min((b + 1) * BIN, n)]            # (bin_tok, 64)
        if len(seg):
            ERR[:, li, b] = seg.mean(0)
    del wuk, kn_o, kn_f, Ko, Kf
    if DEV == "cuda":
        torch.cuda.empty_cache()
ERR = ERR[:, :, :nb]
hmean = np.nanmean(ERR, axis=(1, 2))
print(f"per-head mean 1-cos over all (layer,token): min={hmean.min():.3f} "
      f"max={hmean.max():.3f} (head{hmean.argmax()})  overall={np.nanmean(ERR):.3f}")

vmax = float(np.nanpercentile(ERR, 99))
fig, axes = plt.subplots(8, 8, figsize=(22, 20), sharex=True, sharey=True)
im = None
for h in range(N_HEADS):
    ax = axes[h // 8, h % 8]
    im = ax.imshow(ERR[h], aspect="auto", origin="lower", cmap="magma",
                   vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_title(f"head {h}  (mean {hmean[h]:.2f})", fontsize=7)
    for _n, (s, _e) in SEGS.items():
        ax.axvline(s / BIN - 0.5, color="cyan", lw=0.3, ls="--", alpha=0.5)
for i in range(8):
    axes[i, 0].set_ylabel("layer", fontsize=7)
    axes[7, i].set_xlabel("token", fontsize=7)
    axes[7, i].set_xticks([0, nb // 2, nb - 1])
    axes[7, i].set_xticklabels(["0", str(nb // 2 * BIN), str((nb - 1) * BIN)], fontsize=6)
fig.suptitle(f"Per-HEAD K-cache error  1-cos( {MODE} , {REF} )   x=token  y=layer  (64 heads)",
             fontsize=14)
fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="1-cos")
plt.savefig(OUT, dpi=100, bbox_inches="tight")
print("saved", OUT)
