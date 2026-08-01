import glob, re, os, torch
KD = "/tmp/kd_imp2"
def loadimp(tag, L):
    f = KD + "/" + tag + "_IMP_L" + str(L) + ".pt"
    return set(int(x) for x in torch.load(f, map_location="cpu").tolist()) if os.path.exists(f) else None
def loadK(tag, L):
    cs = glob.glob(KD + "/" + tag + "_L" + str(L) + "_c*.pt")
    if not cs:
        f = KD + "/" + tag + "_L" + str(L) + ".pt"
        return torch.load(f, map_location="cpu").flatten(1).float() if os.path.exists(f) else None
    f = max(cs, key=lambda p: int(re.search(r"_c(\d+)", p).group(1)))
    return torch.load(f, map_location="cpu").flatten(1).float()
def cosvec(a, b):
    n = min(a.shape[0], b.shape[0])
    return torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=-1)

print("========== 1) layer-1 imp SET comparison ==========")
ip, io, ib = loadimp("pic_a3", 1), loadimp("pic_a3_oracle", 1), loadimp("pic_a3b", 1)
if ip and io:
    j = len(io & ip) / len(io | ip)
    print("sizes: pic_a3=%d oracle=%d" % (len(ip), len(io)))
    print("  oracle vs pic_a3 : only_oracle=%d only_pica3=%d jaccard=%.4f"
          % (len(io - ip), len(ip - io), j))
if ip and ib:
    j = len(ib & ip) / len(ib | ip)
    print("  pic_a3b vs pic_a3 (FP8 noise floor): only_b=%d only_a=%d jaccard=%.4f"
          % (len(ib - ip), len(ip - ib), j))

print("\n========== 2) K at NON-imp-in-both positions (pure prepop cache) ==========")
print("   oracle-vs-pica3 should ~= pica3b-vs-pica3 IF the diff is only FP8 noise")
if ip is not None and io is not None:
    for L in [2, 8, 12, 16, 19, 20]:
        pa = loadK("pic_a3", L); oa = loadK("pic_a3_oracle", L); ob = loadK("pic_a3b", L)
        if pa is None or oa is None:
            continue
        both_nonimp = [p for p in range(pa.shape[0]) if p not in ip and p not in io]
        idx = torch.tensor(both_nonimp, dtype=torch.long)
        e_oracle = float((1 - cosvec(oa, pa)[idx]).clamp(0, 2).mean())
        e_noise = float((1 - cosvec(ob, pa)[idx]).clamp(0, 2).mean()) if ob is not None else float("nan")
        print("  L%2d  non-imp-both=%d  |  oracle-vs-pica3=%.6f   noise(pica3b-vs-pica3)=%.6f"
              % (L, len(both_nonimp), e_oracle, e_noise))

print("\n========== 3) K over ALL positions (for reference) ==========")
for L in [2, 8, 12, 19]:
    pa = loadK("pic_a3", L); oa = loadK("pic_a3_oracle", L); ob = loadK("pic_a3b", L)
    if pa is None or oa is None:
        continue
    e_oracle = float((1 - cosvec(oa, pa)).clamp(0, 2).mean())
    e_noise = float((1 - cosvec(ob, pa)).clamp(0, 2).mean()) if ob is not None else float("nan")
    print("  L%2d  ALL pos  |  oracle-vs-pica3=%.6f   noise(pica3b-vs-pica3)=%.6f"
          % (L, e_oracle, e_noise))
