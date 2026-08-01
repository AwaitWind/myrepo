#!/usr/bin/env python
"""
PIC attention-sink demo (annotated).

Same three scenarios as before, but the plot now clearly shows
  - WHICH document (SYS/C1/C2/C3/Q) each sink belongs to
  - WHICH position (token index + decoded token text)
  - HOW MUCH attention mass that position captures

Scenarios (all aggregate Q rows across the three middle knowledge docs C1+C2+C3):
  A. Full recompute (standard causal)
     -> natural sink at pos 0 (SYS/BOS region).
  B. PIC segment-isolated, NO sink prefix
     -> one sink per doc, at each doc's first content token.
  C. PIC segment-isolated + N-token sink prefix per doc
     -> sinks land on the dummy prefix; content tokens stay clean.

Usage:
  python scripts/pic_attention_sink_demo.py \\
      --model /root/Qwen3 \\
      --sink-len 16
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.transforms import blended_transform_factory
from transformers import AutoModelForCausalLM, AutoTokenizer


SEGMENT_TEXTS = [
    "You are a helpful assistant that answers questions using the provided documents.",
    "Python was created by Guido van Rossum and first released in 1991. It uses dynamic typing and automatic memory management via garbage collection.",
    "JavaScript was designed by Brendan Eich in 1995. It runs in web browsers and is dynamically typed with prototype-based inheritance.",
    "Rust was created by Graydon Hoare in 2010 and is developed by the Rust Foundation. It emphasizes memory safety through ownership rules without a garbage collector.",
    "Which language uses garbage collection?",
]
SEGMENT_LABELS = ["SYS", "C1", "C2", "C3", "Q"]
SEGMENT_BG_COLORS = ["#eaeaea", "#ffe0e0", "#e0f2ff", "#e5ffe0", "#fff0d0"]
SEGMENT_TEXT_COLORS = ["#555", "#a02020", "#204070", "#207030", "#805020"]

MIDDLE_SEG_IDS = (1, 2, 3)  # C1, C2, C3


def tokenize_segments(
    tokenizer,
    texts: List[str],
    sink_len: int = 0,
    sink_token_id: Optional[int] = None,
) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
    if sink_token_id is None:
        sink_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.bos_token_id or 0)
        )
    all_ids: List[int] = []
    segments: List[Tuple[int, int]] = []
    for text in texts:
        start = len(all_ids)
        if sink_len > 0:
            all_ids.extend([int(sink_token_id)] * sink_len)
        all_ids.extend(tokenizer.encode(text, add_special_tokens=False))
        segments.append((start, len(all_ids)))
    return torch.tensor([all_ids], dtype=torch.long), segments


def make_seg_isolated_allowed(
    N: int, segments: List[Tuple[int, int]], last_seg_full_causal: bool = True
) -> torch.Tensor:
    allowed = torch.tril(torch.ones(N, N, dtype=torch.bool))
    middle = segments[:-1] if last_seg_full_causal else segments
    for (s, e) in middle:
        for i in range(s, e):
            allowed[i, :s] = False
    return allowed


def restrict_and_renormalize(
    probs: torch.Tensor, allowed: torch.Tensor
) -> torch.Tensor:
    p = probs * allowed.to(probs.dtype).unsqueeze(0).unsqueeze(0)
    p = p / (p.sum(dim=-1, keepdim=True).clamp_min(1e-9))
    return p


def aggregate_over_ranges(
    probs: torch.Tensor, ranges: List[Tuple[int, int]]
) -> np.ndarray:
    """Mean attention over concatenated Q rows in `ranges`, mean over heads.

    probs: (B, H, N, N). Returns (N,) — mass received per K position.
    """
    parts = [probs[0, :, s:e, :] for (s, e) in ranges if e > s]
    if not parts:
        return np.zeros(probs.shape[-1])
    concat = torch.cat(parts, dim=1)  # (H, total_rows, N)
    return concat.mean(dim=(0, 1)).cpu().float().numpy()


def get_attention_probs(model, input_ids: torch.Tensor, layer_idx: int) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids, output_attentions=True)
    return out.attentions[layer_idx].to(torch.float32)


def decode_token(tokenizer, input_ids: torch.Tensor, pos: int, max_len: int = 12) -> str:
    tok_id = int(input_ids[0, pos].item())
    s = tokenizer.decode([tok_id])
    s = s.replace("\n", "\\n").replace("\t", "\\t")
    s_stripped = s.strip()
    display = s_stripped if s_stripped else s if s else f"<id{tok_id}>"
    if len(display) > max_len:
        display = display[: max_len - 1] + "…"
    return display


def find_sinks_in_zone(
    recv: np.ndarray,
    segments: List[Tuple[int, int]],
    zone_len: int,
    top_k_per_seg: int = 1,
    seg_indices: Tuple[int, ...] = MIDDLE_SEG_IDS,
) -> List[dict]:
    """For each segment in seg_indices, return top-k positions within its first
    `zone_len` tokens sorted by attention mass received.
    """
    sinks = []
    for i in seg_indices:
        s, e = segments[i]
        end_zone = min(s + zone_len, e)
        if end_zone <= s:
            continue
        zone_mass = recv[s:end_zone]
        k = min(top_k_per_seg, len(zone_mass))
        top_local = np.argsort(zone_mass)[-k:][::-1]
        for lp in top_local:
            sinks.append({
                "seg_label": SEGMENT_LABELS[i],
                "seg_idx": i,
                "global_pos": int(s + lp),
                "local_pos": int(lp),
                "mass": float(zone_mass[lp]),
            })
    return sinks


def draw_segment_backgrounds(ax, segments: List[Tuple[int, int]]):
    for i, (s, e) in enumerate(segments):
        color = SEGMENT_BG_COLORS[i % len(SEGMENT_BG_COLORS)]
        ax.axvspan(s - 0.5, e - 0.5, color=color, alpha=0.5, zorder=-3)
        ax.axvline(s - 0.5, color="#888", linestyle="-", linewidth=0.5, alpha=0.6, zorder=-2)


def draw_segment_labels(ax, segments: List[Tuple[int, int]]):
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for i, (s, e) in enumerate(segments):
        ax.text(
            (s + e) / 2, 0.965, SEGMENT_LABELS[i],
            ha="center", va="top", transform=trans,
            fontsize=11, fontweight="bold",
            color=SEGMENT_TEXT_COLORS[i % len(SEGMENT_TEXT_COLORS)],
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#555", alpha=0.92, linewidth=0.8),
            zorder=5,
        )


def annotate_sinks(ax, sinks: List[dict], tokenizer, input_ids: torch.Tensor):
    """Draw arrow + text box at each sink position."""
    # Stagger Y in axes fraction to reduce overlap
    y_slots = [0.78, 0.60, 0.42, 0.78, 0.60, 0.42]  # cycle
    for idx, sink in enumerate(sorted(sinks, key=lambda s: s["global_pos"])):
        pos = sink["global_pos"]
        tok = decode_token(tokenizer, input_ids, pos)
        label = (
            f"{sink['seg_label']}\n"
            f"pos={pos}\n"
            f"tok='{tok}'\n"
            f"mass={sink['mass']:.3f}"
        )
        y_frac = y_slots[idx % len(y_slots)]
        ax.annotate(
            label,
            xy=(pos, max(sink["mass"], 1e-4)),
            xycoords="data",
            xytext=(pos, y_frac),
            textcoords=blended_transform_factory(ax.transData, ax.transAxes),
            fontsize=8, ha="center", va="top",
            arrowprops=dict(arrowstyle="->", color="#c81717", lw=1.0, alpha=0.9),
            bbox=dict(boxstyle="round,pad=0.28", fc="#ffff99", ec="#c81717", alpha=0.95, linewidth=0.8),
            zorder=10,
        )


def plot_panel(
    ax, recv: np.ndarray, segments: List[Tuple[int, int]],
    title: str, bar_color: str, sinks: List[dict],
    tokenizer, input_ids: torch.Tensor,
):
    draw_segment_backgrounds(ax, segments)
    xs = np.arange(len(recv))
    ax.bar(
        xs, np.maximum(recv, 1e-6),
        color=bar_color, width=1.0, edgecolor="none", alpha=0.85, zorder=1,
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_xlim(-0.5, len(recv) - 0.5)
    ax.set_ylabel("attn mass (log)")
    ax.set_title(title, loc="left", fontsize=12, pad=8, fontweight="bold")
    ax.grid(axis="y", which="major", linestyle=":", alpha=0.35, zorder=0)
    ax.tick_params(axis="both", labelsize=9)

    draw_segment_labels(ax, segments)
    annotate_sinks(ax, sinks, tokenizer, input_ids)


def print_sink_table(scenario_name: str, sinks: List[dict], tokenizer, ids: torch.Tensor):
    print(f"\n▪ {scenario_name}")
    print(f"  {'seg':<5} {'global_pos':<11} {'local':<6} {'token':<16} {'mass':<8}")
    print(f"  {'---':<5} {'----------':<11} {'-----':<6} {'---------------':<16} {'-------':<8}")
    for s in sinks:
        tok = decode_token(tokenizer, ids, s["global_pos"])
        print(f"  {s['seg_label']:<5} {s['global_pos']:<11} {s['local_pos']:<6} {repr(tok):<16} {s['mass']:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/root/Qwen3")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--layer", type=int, default=-1,
                        help="Layer index. Default: middle layer (typically shows sink most clearly).")
    parser.add_argument("--sink-len", type=int, default=16)
    parser.add_argument("--top-k-per-seg", type=int, default=1,
                        help="Number of sink positions to annotate per segment (default 1).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="scripts/pic_attention_sink_demo.png")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    print(f"[info] device={device}, dtype={args.dtype}")

    print(f"[info] loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        attn_implementation="eager", trust_remote_code=True,
    ).to(device).eval()

    L = model.config.num_hidden_layers
    layer_idx = args.layer if args.layer >= 0 else L + args.layer
    layer_idx = max(0, min(L - 1, layer_idx))
    if args.layer == -1:
        layer_idx = L // 2
    print(f"[info] using layer {layer_idx}/{L}")

    ids_ns, segs_ns = tokenize_segments(tokenizer, SEGMENT_TEXTS, sink_len=0)
    ids_s, segs_s = tokenize_segments(tokenizer, SEGMENT_TEXTS, sink_len=args.sink_len)
    ids_ns, ids_s = ids_ns.to(device), ids_s.to(device)

    print(f"[info] no-sink : N={ids_ns.shape[1]}, segments={segs_ns}")
    print(f"[info] sink={args.sink_len:<3}: N={ids_s.shape[1]}, segments={segs_s}")

    print("[info] forward pass 1/2 (no sink) ...")
    probs_ns = get_attention_probs(model, ids_ns, layer_idx)
    print("[info] forward pass 2/2 (with sink) ...")
    probs_s = get_attention_probs(model, ids_s, layer_idx)

    p_A = probs_s  # full causal on SINK-AUGMENTED input (aligned with C)
    p_B = restrict_and_renormalize(
        probs_ns, make_seg_isolated_allowed(ids_ns.shape[1], segs_ns).to(device)
    )
    p_C = restrict_and_renormalize(
        probs_s, make_seg_isolated_allowed(ids_s.shape[1], segs_s).to(device)
    )

    # Q rows: middle segments (C1, C2, C3). For scenarios A & C (both on
    # sink-augmented input), aggregate over CONTENT rows only so A and C use
    # exactly the same Q rows -> per-K comparison is apples-to-apples.
    # Scenario B is on the no-sink input, so it uses full middle-segment ranges.
    q_ranges_ns = [segs_ns[i] for i in MIDDLE_SEG_IDS]
    q_ranges_s_content = [
        (segs_s[i][0] + args.sink_len, segs_s[i][1]) for i in MIDDLE_SEG_IDS
    ]

    recv_A = aggregate_over_ranges(p_A, q_ranges_s_content)
    recv_B = aggregate_over_ranges(p_B, q_ranges_ns)
    recv_C = aggregate_over_ranges(p_C, q_ranges_s_content)

    # Sinks:
    #   A: natural sink at pos 0 (on sink-augmented input, that's SYS's first sink token).
    #      Also probe first 2 positions of each middle seg (usually low under full causal).
    #   B: top-1 in each middle seg's first 2 positions (no-sink input) — the pollution.
    #   C: top-1 in each middle seg's first sink_len positions (sink-augmented input).
    sinks_A = [{
        "seg_label": SEGMENT_LABELS[0], "seg_idx": 0,
        "global_pos": 0, "local_pos": 0, "mass": float(recv_A[0]),
    }] + find_sinks_in_zone(recv_A, segs_s, zone_len=2, top_k_per_seg=args.top_k_per_seg)
    sinks_B = find_sinks_in_zone(recv_B, segs_ns, zone_len=2, top_k_per_seg=args.top_k_per_seg)
    sinks_C = find_sinks_in_zone(
        recv_C, segs_s, zone_len=args.sink_len, top_k_per_seg=args.top_k_per_seg
    )

    print("\n=== Attention sinks: which segment / which position / which token ===")
    print_sink_table(
        "A. Full recompute on sink-augmented input (aligned with C)",
        sinks_A, tokenizer, ids_s,
    )
    print_sink_table("B. PIC no sink prefix (on no-sink input, N smaller)", sinks_B, tokenizer, ids_ns)
    print_sink_table(f"C. PIC + {args.sink_len}-token sink prefix", sinks_C, tokenizer, ids_s)

    fig, axes = plt.subplots(3, 1, figsize=(18, 13), sharex=False)
    plot_panel(
        axes[0], recv_A, segs_s,
        "A. Full recompute (full causal) on SINK-AUGMENTED input   —   Q rows: C1+C2+C3 CONTENT",
        "#4C78A8", sinks_A, tokenizer, ids_s,
    )
    plot_panel(
        axes[1], recv_B, segs_ns,
        "B. PIC segment-isolated, NO sink prefix (input NOT augmented)   —   Q rows: C1+C2+C3",
        "#E45756", sinks_B, tokenizer, ids_ns,
    )
    plot_panel(
        axes[2], recv_C, segs_s,
        f"C. PIC segment-isolated + {args.sink_len}-token sink prefix   —   Q rows: C1+C2+C3 CONTENT",
        "#59A14F", sinks_C, tokenizer, ids_s,
    )
    axes[-1].set_xlabel("K position (token index)", fontsize=11)

    fig.suptitle(
        f"PIC attention-sink: WHICH document, WHICH position\n"
        f"model={args.model}   layer={layer_idx}/{L}   sink_len={args.sink_len}   "
        f"(bar height = attention mass on that K position, log scale)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n[done] saved plot to {out_path.resolve()}")


if __name__ == "__main__":
    main()
