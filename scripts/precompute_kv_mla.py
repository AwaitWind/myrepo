#!/usr/bin/env python3
"""precompute_kv_mla.py — generate A³ / CacheBlend precomputed-KV .pt files
for a single prompt on the GLM-5.2 (GlmMoeDsaForCausalLM) MLA/DSA architecture.

**Multi-TP capable.** Uses env-gated capture hooks in forward_mla.py and
dsa_indexer.py (SGLANG_A3_CAPTURE_KV_DIR) which inherit into Engine's TP
worker subprocesses; each rank dumps its own shard, then this script
stitches per-rank layer shards into per-rank consolidated .pt files.

Output layout::

    <output_dir>/
        rank0.pt          # {'latent': (L, T, D_lat), 'index_k': (L, T, D_idx),
                          #  'metadata': {...}}
        rank1.pt
        ...
        rank{N-1}.pt

At inference time, set the request's ``precomputed_kv_path`` to a path
containing the ``{rank}`` template (e.g. ``/tmp/kv/rank{rank}.pt``);
``Req.get_precomputed_kv`` substitutes ``{rank}`` with the local TP rank so
every worker loads its own shard.

Usage::

    python scripts/precompute_kv_mla.py \\
        --model /workspace/models/GLM-5.2-FP8 \\
        --prompt-file prompt.txt \\
        --output-dir /tmp/precomputed_kv/ \\
        --tp 8
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from typing import Dict, Optional

# ── CUDA 13 兼容库路径注入(与 quick_test_online.py:199-202 一致)──
# deep_gemm 的 _C.so 在 import 时 dlopen 需要 libnvrtc.so.13,不在默认路径。
# LD_LIBRARY_PATH 必须在 Python 进程启动前设置,所以若发现没设,就 exec 自己一遍。
def _ensure_cuda13_ld_path() -> None:
    cu13_lib = "/opt/dynamo/venv/lib/python3.12/site-packages/nvidia/cu13/lib"
    cuda_compat = "/usr/local/cuda-13.0/compat"
    if not (os.path.isdir(cu13_lib) or os.path.isdir(cuda_compat)):
        return  # 非本环境,不做任何事
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    needed = f"{cuda_compat}:{cu13_lib}"
    if all(p in existing.split(":") for p in [cuda_compat, cu13_lib]):
        return  # 已设,直接继续
    # 未设 → 注入后 exec 自身一遍(动态加载器只读一次 LD_LIBRARY_PATH)
    new_env = os.environ.copy()
    new_env["LD_LIBRARY_PATH"] = f"{needed}:{existing}" if existing else needed
    # 打个标记避免无限循环 exec
    if os.environ.get("_PRECOMPUTE_KV_LDPATH_INJECTED") == "1":
        return
    new_env["_PRECOMPUTE_KV_LDPATH_INJECTED"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], new_env)


_ensure_cuda13_ld_path()

import torch  # noqa: E402 — 必须在 LD_LIBRARY_PATH 注入之后


_SHARD_RE = re.compile(
    r"rank(?P<rank>\d+)_layer(?P<layer>\d+)_(?P<kind>latent|index_k)\.pt"
)


def _stitch_rank(
    shard_dir: str,
    rank: int,
    num_layers: int,
) -> Dict[str, Optional[torch.Tensor]]:
    """Collect all layer shards for a given rank into two stacked tensors."""
    latent_by_layer: Dict[int, torch.Tensor] = {}
    index_k_by_layer: Dict[int, torch.Tensor] = {}

    for fname in os.listdir(shard_dir):
        m = _SHARD_RE.match(fname)
        if not m or int(m["rank"]) != rank:
            continue
        layer_id = int(m["layer"])
        kind = m["kind"]
        try:
            t = torch.load(os.path.join(shard_dir, fname), map_location="cpu")
        except Exception as e:  # noqa: BLE001
            print(f"[stitch] rank{rank} {fname}: load failed — {e}", file=sys.stderr)
            continue
        if kind == "latent":
            latent_by_layer[layer_id] = t
        else:
            index_k_by_layer[layer_id] = t

    if not latent_by_layer:
        print(f"[stitch] rank{rank}: no latent shards found", file=sys.stderr)
        return {"latent": None, "index_k": None}

    first = latent_by_layer[min(latent_by_layer)]
    T, D_lat = first.shape
    dtype = first.dtype
    latent = torch.zeros((num_layers, T, D_lat), dtype=dtype)
    for lid, t in latent_by_layer.items():
        if t.shape != (T, D_lat):
            print(f"[stitch] rank{rank} layer{lid}: shape mismatch {t.shape}", file=sys.stderr)
            continue
        latent[lid] = t

    index_k: Optional[torch.Tensor] = None
    if index_k_by_layer:
        first_ik = index_k_by_layer[min(index_k_by_layer)]
        T_ik, D_idx = first_ik.shape
        if T_ik != T:
            print(
                f"[stitch] rank{rank}: latent T={T} vs index_k T={T_ik} mismatch; "
                f"using latent's T for shape",
                file=sys.stderr,
            )
        index_k = torch.zeros((num_layers, T, D_idx), dtype=first_ik.dtype)
        for lid, t in index_k_by_layer.items():
            if t.shape[0] != T:
                continue
            index_k[lid] = t

    return {"latent": latent, "index_k": index_k}


def main():
    parser = argparse.ArgumentParser(
        description="Generate A³/CacheBlend precomputed-KV .pt for GLM-5.2 MLA/DSA "
                    "(multi-TP via env-gated capture)."
    )
    parser.add_argument("--model", required=True, help="Path to GLM-5.2-FP8 model")
    parser.add_argument("--prompt-file", help="File containing the prompt.")
    parser.add_argument("--prompt", help="Prompt string.")
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory for per-rank .pt files (rank0.pt, rank1.pt, ...)."
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--mem-fraction", type=float, default=0.82)
    parser.add_argument(
        "--shard-dir", default=None,
        help="Temp dir for per-layer per-rank shards. Default: <output_dir>/.shards"
    )
    parser.add_argument(
        "--keep-shards", action="store_true",
        help="Do not delete the per-layer shard directory after stitching."
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true", default=True
    )
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompt = f.read().rstrip("\n")
    elif args.prompt:
        prompt = args.prompt
    else:
        print("ERROR: must supply --prompt or --prompt-file", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    shard_dir = args.shard_dir or os.path.join(output_dir, ".shards")
    os.makedirs(output_dir, exist_ok=True)
    # Clean stale shards from any prior run so the stitcher only sees fresh output.
    if os.path.isdir(shard_dir):
        for f in os.listdir(shard_dir):
            if _SHARD_RE.match(f):
                os.remove(os.path.join(shard_dir, f))
    else:
        os.makedirs(shard_dir, exist_ok=True)

    # Tell every TP worker to dump per-layer shards here.
    os.environ["SGLANG_A3_CAPTURE_KV_DIR"] = shard_dir
    # DeepGEMM fast warmup:每个 GEMM shape 的 warmup m_list 从 65536 降到 ~3073
    # (约 21×)。GLM-5.2 (DeepSeek DSA/V3) 有几十个 GEMM shape,--chunked-prefill-size=-1
    # 会触发 m_max=65536 的全量 warmup,启动时间容易过长。
    # 与 quick_test_online.py:53+208 同一配置。
    # 用 setdefault 允许用户在 shell 里显式覆盖(export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=0)
    os.environ.setdefault("SGLANG_JIT_DEEPGEMM_FAST_WARMUP", "1")
    # 额外的严格内存检查降级为警告(A³ / precompute 会临时占用 KV pool 私有 slots)
    os.environ.setdefault("SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE", "0")
    print(f"[precompute-kv] model={args.model} tp={args.tp} prompt_len={len(prompt)}")
    print(f"[precompute-kv] shard_dir={shard_dir}")
    print(f"[precompute-kv] output_dir={output_dir}")
    print(f"[precompute-kv] SGLANG_JIT_DEEPGEMM_FAST_WARMUP="
          f"{os.environ['SGLANG_JIT_DEEPGEMM_FAST_WARMUP']}"
          f" (=1 时日志 DeepGEMM warmup 进度条 total 应为 ~3073 而非 65536)")

    from sglang import Engine

    engine_args = dict(
        model_path=args.model,
        tp_size=args.tp,
        trust_remote_code=args.trust_remote_code,
        mem_fraction_static=args.mem_fraction,
        chunked_prefill_size=-1,
        disable_overlap_schedule=True,
        disable_cuda_graph=True,
        # 关键:关闭 piecewise cuda graph — 否则 warmup 阶段 torch.compile 会
        # 编译 forward_absorb_prepare,is_compiling() guard 让 capture 分支
        # 被跳过,编译版 graph 永远不含 torch.save,导致后续真实请求也不 dump。
        disable_piecewise_cuda_graph=True,
    )
    print(f"[precompute-kv] Engine kwargs = {engine_args}")
    engine = Engine(**engine_args)

    try:
        t0 = time.perf_counter()
        _ = engine.generate(
            prompt=prompt,
            sampling_params={"max_new_tokens": 1, "temperature": 0.0},
        )
        print(f"[precompute-kv] prefill done in {time.perf_counter()-t0:.2f}s")
    finally:
        try:
            engine.shutdown()
        except Exception:  # noqa: BLE001
            pass

    # Give TP workers a moment to flush torch.save (in case they're slower).
    time.sleep(1.0)

    # Load HF config for metadata + num_layers.
    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    num_layers = hf_cfg.num_hidden_layers
    kv_lora_rank = getattr(hf_cfg, "kv_lora_rank", 0)
    qk_rope_head_dim = getattr(hf_cfg, "qk_rope_head_dim", 0)
    index_head_dim = getattr(hf_cfg, "index_head_dim", 0)

    # Discover which ranks actually wrote shards. If some ranks are missing,
    # something went wrong per-rank — print a clear error but still stitch
    # whatever we have.
    ranks_seen = set()
    for fname in os.listdir(shard_dir):
        m = _SHARD_RE.match(fname)
        if m:
            ranks_seen.add(int(m["rank"]))
    ranks_seen = sorted(ranks_seen)
    print(f"[precompute-kv] shard ranks present: {ranks_seen}")
    if not ranks_seen:
        print("ERROR: no shards captured — env var may not have reached "
              "TP workers, or capture hooks were not exercised.", file=sys.stderr)
        sys.exit(3)
    if len(ranks_seen) != args.tp:
        print(f"WARNING: expected {args.tp} ranks, got {len(ranks_seen)}",
              file=sys.stderr)

    # Stitch and save per rank.
    for rank in ranks_seen:
        combined = _stitch_rank(shard_dir, rank, num_layers)
        payload = {
            "latent": combined["latent"],
            "index_k": combined["index_k"],
            "metadata": {
                "model": args.model,
                "prompt": prompt if len(prompt) < 8192 else prompt[:8192] + "...(trunc)",
                "total_len": (
                    combined["latent"].shape[1]
                    if combined["latent"] is not None else 0
                ),
                "num_layers": num_layers,
                "kv_lora_rank": kv_lora_rank,
                "qk_rope_head_dim": qk_rope_head_dim,
                "index_head_dim": index_head_dim,
                "tp_rank": rank,
                "tp_size": args.tp,
            },
        }
        out_path = os.path.join(output_dir, f"rank{rank}.pt")
        torch.save(payload, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        lat_shape = tuple(payload["latent"].shape) if payload["latent"] is not None else None
        idx_shape = tuple(payload["index_k"].shape) if payload["index_k"] is not None else None
        print(f"  rank{rank} → {out_path}  ({size_mb:.1f} MB)  "
              f"latent={lat_shape}  index_k={idx_shape}")

    # Cleanup.
    if not args.keep_shards:
        shutil.rmtree(shard_dir, ignore_errors=True)
        print(f"[precompute-kv] shard_dir cleaned")

    print(f"\n[precompute-kv] DONE. At inference time, set:")
    print(f"  precomputed_kv_path = \"{output_dir}/rank{{rank}}.pt\"")
    print(f"  (the {{rank}} template is substituted per TP rank by "
          f"Req.get_precomputed_kv)")


if __name__ == "__main__":
    main()
