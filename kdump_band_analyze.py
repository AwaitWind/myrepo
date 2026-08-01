import glob, re, torch, os, statistics as st
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
layers = [1, 2, 4, 8, 12, 16, 19, 20, 30, 40, 50, 60, 70, 77]
print("per-layer mean(1-cos) vs full_recompute:")
print("  L  | " + " | ".join(m.rjust(13) for m in modes))
for L in layers:
    b = load(REF, L)
    row = []
    for m in modes:
        a = load(m, L)
        row.append(coserr(a, b) if (a is not None and b is not None) else float("nan"))
    print(str(L).rjust(4) + " | " + " | ".join(("%.4f" % e).rjust(13) for e in row))
print("")
print("band means over ALL 78 layers (front=L0-19, deep=L20-77):")
for m in modes:
    frc, dpc, frl, dpl = [], [], [], []
    for L in range(78):
        a = load(m, L); b = load(REF, L)
        if a is None or b is None: continue
        (frc if L < 20 else dpc).append(coserr(a, b))
        (frl if L < 20 else dpl).append(l2err(a, b))
    print("%14s: cos front=%.4f deep=%.4f | euclid front=%.3f deep=%.3f"
          % (m, st.mean(frc), st.mean(dpc), st.mean(frl), st.mean(dpl)))
