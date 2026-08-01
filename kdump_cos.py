import torch, glob, os
KD = os.environ.get("KD", "/tmp/kd")
segs = {"SYS": (0, 64), "C1": (64, 1216), "C2": (1216, 2304),
        "C3": (2304, 3392), "Q": (3392, 3456)}
print(f"K cos(clip vs full_recompute ref) per segment, dir={KD}")
print(f"{'layer':>6} | " + " ".join(f"{n:>7}" for n in segs))
for L in [2, 20, 40, 60, 77]:
    fa, fb = f"{KD}/ref_L{L}.pt", f"{KD}/clip_L{L}.pt"
    if not (os.path.exists(fa) and os.path.exists(fb)):
        print(f"L{L:>4}  | missing (ref={os.path.exists(fa)} clip={os.path.exists(fb)})")
        continue
    a = torch.load(fa).flatten(1).float()
    b = torch.load(fb).flatten(1).float()
    n = min(a.shape[0], b.shape[0])
    cos = torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=-1)
    row = f"L{L:>4}  | "
    for name, (s, e) in segs.items():
        e2 = min(e, n)
        row += f"{cos[s:e2].mean().item():>7.3f} " if e2 > s else f"{'--':>7} "
    print(row)
