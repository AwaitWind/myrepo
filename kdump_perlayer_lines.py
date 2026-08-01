import glob, re, os, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
KD = "/tmp/kd_realL1"; REF = "full_recompute"
def load(tag, L):
    cs = glob.glob(KD + "/" + tag + "_L" + str(L) + "_c*.pt")
    if not cs:
        f = KD + "/" + tag + "_L" + str(L) + ".pt"
        return torch.load(f, map_location="cpu").flatten(1).float() if os.path.exists(f) else None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()
def coserr(a, b):
    n = min(a.shape[0], b.shape[0]); a, b = a[:n], b[:n]
    return float((1 - torch.nn.functional.cosine_similarity(a, b, dim=-1)).clamp(0, 2).mean())
def l2err(a, b):
    n = min(a.shape[0], b.shape[0]); a, b = a[:n], b[:n]
    return float((a - b).norm(dim=-1).mean())
modes = ["pic_a3", "pic_a3_oracle", "oracle_iso"]
labels = {"pic_a3": "pic_a3 (single-layer baseline)",
          "pic_a3_oracle": "pic_a3_oracle  NEW  real-L1",
          "oracle_iso": "pic_a3_oracle  OLD  isolated-L1"}
colors = {"pic_a3": "#2c7fb8", "pic_a3_oracle": "#d7301f", "oracle_iso": "#238b45"}
Ls = list(range(78))
cos = {m: [] for m in modes}; euc = {m: [] for m in modes}; xs = {m: [] for m in modes}
for m in modes:
    for L in Ls:
        a = load(m, L); b = load(REF, L)
        if a is None or b is None: continue
        xs[m].append(L); cos[m].append(coserr(a, b)); euc[m].append(l2err(a, b))
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
for m in modes:
    ax[0].plot(xs[m], cos[m], label=labels[m], color=colors[m],
               lw=2.0 if m != "pic_a3" else 1.6,
               ls="--" if m == "oracle_iso" else "-", marker="." , ms=4)
    ax[1].plot(xs[m], euc[m], label=labels[m], color=colors[m],
               lw=2.0 if m != "pic_a3" else 1.6,
               ls="--" if m == "oracle_iso" else "-", marker=".", ms=4)
for a_, t, yl in ((ax[0], "K error: 1 - cos(K vs full_recompute)", "1 - cos"),
                  (ax[1], "K error: euclidean ||K - K_full||", "L2 dist")):
    a_.set_title(t, fontsize=11); a_.set_xlabel("layer"); a_.set_ylabel(yl)
    a_.axvspan(1, 20, color="orange", alpha=0.06)
    a_.axvline(20, color="gray", ls=":", lw=0.8); a_.axvline(40, color="gray", ls=":", lw=0.8)
    a_.axvline(60, color="gray", ls=":", lw=0.8)
    a_.grid(alpha=0.25); a_.legend(fontsize=8, loc="upper left")
fig.suptitle("Per-layer K-cache reuse error vs ground truth (synthetic Mount-Kilimanjaro, 3456 tok)\n"
             "orange band = layers 1-19 (front); dotted lines = oracle deep re-select layers 20/40/60",
             fontsize=11)
plt.tight_layout(rect=(0, 0, 1, 0.94))
plt.savefig(KD + "/kv_err_perlayer_lines.png", dpi=120)
print("saved", KD + "/kv_err_perlayer_lines.png")
