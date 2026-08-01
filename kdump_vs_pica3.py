import glob, re, os, torch
KD = "/tmp/kd_realL1"
def load(tag, L):
    cs = glob.glob(KD + "/" + tag + "_L" + str(L) + "_c*.pt")
    if not cs:
        f = KD + "/" + tag + "_L" + str(L) + ".pt"
        return torch.load(f, map_location="cpu").flatten(1).float() if os.path.exists(f) else None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()
def cosvec(a, b):
    n = min(a.shape[0], b.shape[0]); a, b = a[:n], b[:n]
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)
# Direct comparison: how close is each oracle variant to pic_a3 (not to full_recompute)?
print("per-layer  mean(1-cos) vs pic_a3 (DIRECT)  |  #pos with cos<0.99 (of 3456)")
print("  L  | oracle_real | oracle_iso | full_reco || nbad_real nbad_iso nbad_full")
for L in [1, 2, 4, 8, 12, 16, 19, 20, 40, 60]:
    pa = load("pic_a3", L)
    out = []
    nbad = []
    for m in ["pic_a3_oracle", "oracle_iso", "full_recompute"]:
        a = load(m, L)
        if a is None or pa is None:
            out.append(float("nan")); nbad.append(-1); continue
        c = cosvec(a, pa)
        out.append(float((1 - c).clamp(0, 2).mean()))
        nbad.append(int((c < 0.99).sum()))
    print("%4d | %11.5f | %10.5f | %9.5f || %8d %8d %8d"
          % (L, out[0], out[1], out[2], nbad[0], nbad[1], nbad[2]))
# Also: are pic_a3 and full_recompute even identical at L1? (sanity)
print("\nsanity: pic_a3 vs full_recompute mean(1-cos) at L1 =",
      float((1 - cosvec(load("pic_a3",1), load("full_recompute",1))).mean()))
