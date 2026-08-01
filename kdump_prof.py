"""3-way positional K-cos profiler for pic_a3_oracle keepalive debugging.

Compares K buffers (gathered in POSITION order) across:
  ref       = full_recompute  (ground truth)
  clip      = pic_a3          (single-layer clip; WORKS on dataset)
  keepalive = pic_a3_oracle   (multi-window keepalive; COLLAPSES on dataset)

The key comparison is keepalive-vs-clip: both reuse the SAME isolated cached-K,
but pic_a3 works and keepalive collapses — so where keepalive-K diverges from
clip-K (in position/layer) localizes the keepalive-specific bug.

Prints cos binned over BIN tokens; flags bins with cos<0.9.
Usage: KD=/tmp/kd_cmp BIN=128 python kdump_prof.py
"""
import torch, os

KD = os.environ.get("KD", "/tmp/kd_cmp")
BIN = int(os.environ.get("BIN", "128"))
C = os.environ.get("C", "")  # per-forward counter suffix (e.g. C=4 → sample 2 full)
LAYERS = [int(x) for x in os.environ.get("LAYERS", "2,20,40,60,77").split(",")]

_suf = f"_c{C}" if C else ""


def _load(tag, L):
    p = f"{KD}/{tag}_L{L}{_suf}.pt"
    if not os.path.exists(p):
        return None
    return torch.load(p).flatten(1).float()


def _cmp(a, b, label, L):
    if a is None or b is None:
        print(f"  {label} L{L}: missing (a={a is not None} b={b is not None})")
        return
    n = min(a.shape[0], b.shape[0])
    cos = torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=-1)
    nb = (n + BIN - 1) // BIN
    means = [cos[k * BIN:min((k + 1) * BIN, n)].mean().item() for k in range(nb)]
    bad = [(k * BIN, min((k + 1) * BIN, n), m) for k, m in enumerate(means) if m < 0.9]
    print(f"  {label} L{L}: seqlen={n} mean={cos.mean():.3f} min={cos.min():.3f} "
          f"nbad={len(bad)}/{nb}")
    if bad:
        rng = ", ".join(f"[{s}:{e}]={m:.2f}" for s, e, m in bad[:10])
        print(f"      bad(<0.90): {rng}{' ...' if len(bad) > 10 else ''}")


print(f"K-cos 3-way, dir={KD}, bin={BIN}")
for L in LAYERS:
    ref = _load("ref", L)
    clip = _load("clip", L)
    ka = _load("keepalive", L)
    print(f"--- L{L} ---")
    _cmp(clip, ref, "pic_a3   vs full_recompute", L)   # should be GOOD
    _cmp(ka, ref, "keepalive vs full_recompute", L)    # BAD where it collapses
    _cmp(ka, clip, "keepalive vs pic_a3       ", L)    # KEY: bug localization
