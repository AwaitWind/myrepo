from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.compilation.piecewise_context_manager import is_in_piecewise_cuda_graph
from sglang.srt.environ import envs
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.communicator import get_attn_tp_context
from sglang.srt.layers.quantization.fp8_kernel import (
    fp8_dtype,
    per_tensor_quant_mla_fp8,
    per_token_group_quant_mla_deep_gemm_masked_fp8,
)
from sglang.srt.layers.utils.cp_utils import mla_use_prefill_cp
from sglang.srt.lora.deepseek_mla_correction import (
    apply_q_correction as apply_kv_b_lora_q_correction,
)
from sglang.srt.lora.deepseek_mla_correction import (
    apply_v_correction as apply_kv_b_lora_v_correction,
)
from sglang.srt.lora.deepseek_mla_correction import (
    is_kv_b_lora_active,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    get_attn_backend,
    get_token_to_kv_pool,
)
from sglang.srt.models.deepseek_common.utils import (
    FORWARD_ABSORB_CORE_ATTENTION_BACKENDS,
    _is_cpu,
    _is_cublas_ge_129,
    _is_cuda,
    _is_gfx95_supported,
    _is_hip,
    _is_musa,
    _use_aiter,
    _use_aiter_bpreshuffle_gfx95,
    _use_aiter_gfx95,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.state_capturer.indexer_topk import (
    maybe_capture_indexer_topk,
)
from sglang.srt.utils import BumpAllocator

_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

if TYPE_CHECKING:
    from sglang.srt.models.deepseek_v2 import DeepseekV2AttentionMLA

if _is_cuda:
    from sgl_kernel import bmm_fp8 as _raw_bmm_fp8

    from sglang.srt.utils.custom_op import register_custom_op

    # TODO(yuwei): remove this wrapper after sgl-kernel registers its own fake/meta impl
    # Wrap bmm_fp8 as a custom op so torch.compile does not trace into
    # torch.cuda.current_blas_handle() (which returns a non-Tensor).
    @register_custom_op(mutates_args=["out"])
    def _bmm_fp8_op(
        A: torch.Tensor,
        B: torch.Tensor,
        out: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
    ) -> None:
        _raw_bmm_fp8(A, B, A_scale, B_scale, out.dtype, out)

    def bmm_fp8(A, B, A_scale, B_scale, dtype, out=None):
        if out is None:
            out = torch.empty(
                (A.shape[0], A.shape[1], B.shape[2]),
                device=A.device,
                dtype=dtype,
            )
        _bmm_fp8_op(A, B, out, A_scale, B_scale)
        return out


if _use_aiter:
    # aiter ROCm/aiter#2958 renamed the public `fused_qk_rmsnorm` in
    # `aiter.ops.fused_qk_norm_rope_cache_quant` to a private `_fused_qk_rmsnorm`
    # and introduced a unified entry point in `aiter.ops.fused_qk_rmsnorm_group_quant`
    # with a different (in-place, kwarg-only, no-return) signature. Probe for the
    # new symbol first so SGLang works with both pre- and post-#2958 aiter without
    # requiring the docker pin to be bumped atomically.
    try:
        from aiter.ops.enum import QuantType as _AiterQuantType
        from aiter.ops.fused_qk_rmsnorm_group_quant import (
            fused_qk_rmsnorm as _aiter_fused_qk_rmsnorm_unified,
        )

        def fused_qk_rmsnorm_bf16(q, q_weight, q_eps, k, k_weight, k_eps):
            q_out = torch.empty_like(q)
            k_out = torch.empty_like(k)
            _aiter_fused_qk_rmsnorm_unified(
                q_out_quantized=q_out,
                k_out=k_out,
                q=q,
                q_weight=q_weight,
                q_epsilon=q_eps,
                k=k,
                k_weight=k_weight,
                k_epsilon=k_eps,
                quant_type=_AiterQuantType.No,
            )
            return q_out, k_out

    except ImportError:
        from aiter.ops.fused_qk_norm_rope_cache_quant import (
            fused_qk_rmsnorm as fused_qk_rmsnorm_bf16,
        )

    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
    )
if _use_aiter_gfx95:
    from aiter.ops.triton.fused_fp8_quant import (
        fused_flatten_fp8_group_quant,
        fused_rms_fp8_group_quant,
    )

    from sglang.srt.layers.quantization.rocm_mxfp4_utils import (
        batched_gemm_afp4wfp4_pre_quant,
        fused_flatten_mxfp4_quant,
        fused_rms_mxfp4_quant,
    )
    from sglang.srt.layers.rocm_linear_utils import fused_qk_rope_cat_and_cache_mla


class DeepseekMLAForwardMixin:
    def init_mla_forward(self: DeepseekV2AttentionMLA):
        self.flashinfer_mla_disable_ragged = (
            get_global_server_args().flashinfer_mla_disable_ragged
        )

    def forward_absorb_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        q_lora = None
        topk_indices = None
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]

            # overlap qk norm
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                if _use_aiter_gfx95 and self.q_b_proj.weight.dtype == torch.uint8:
                    q, _, k_nope, *_ = fused_rms_mxfp4_quant(
                        q,
                        self.q_a_layernorm.weight,
                        self.q_a_layernorm.variance_epsilon,
                        k_nope,
                        self.kv_a_layernorm.weight,
                        self.kv_a_layernorm.variance_epsilon,
                    )
                else:
                    q_lora = None
                    if (
                        _use_aiter_gfx95
                        and self.q_b_proj.weight.dtype == torch.float8_e4m3fn
                    ):
                        if self.use_dsa:
                            q_quanted, q_lora, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=True,
                                transpose_scale=_use_aiter_bpreshuffle_gfx95,
                            )
                            q = q_quanted
                        else:
                            q, _, k_nope, _ = fused_rms_fp8_group_quant(
                                q,
                                self.q_a_layernorm.weight,
                                self.q_a_layernorm.variance_epsilon,
                                k_nope,
                                self.kv_a_layernorm.weight,
                                self.kv_a_layernorm.variance_epsilon,
                                group_size=128,
                                dtype_quant=torch.float8_e4m3fn,
                                res1=None,
                                output_unquantized_inp1=False,
                                transpose_scale=_use_aiter_bpreshuffle_gfx95,
                            )

                    elif _use_aiter:
                        q, k_nope = fused_qk_rmsnorm_bf16(
                            q,
                            self.q_a_layernorm.weight,
                            self.q_a_layernorm.variance_epsilon,
                            k_nope,
                            self.kv_a_layernorm.weight,
                            self.kv_a_layernorm.variance_epsilon,
                        )
                    else:
                        q = self.q_a_layernorm(q)
                        # PIC split: kv_a_layernorm only for miss segments.
                        # Defensive shape check: pic_hit_pub_kv_loc must have
                        # the same length as the current forward's k_nope.
                        # In decode this tensor should be None (cleared by
                        # prepare_for_decode); the shape guard is belt-and-
                        # suspenders in case it leaks from continuous batching.
                        _pic_pub = getattr(forward_batch, "pic_hit_pub_kv_loc", None)
                        # pic_a3 / pic_cacheblend: skip the split-layernorm
                        # K-override branch. Under IMP_ONLY these two modes
                        # only put miss+imp positions into input_ids; hit
                        # positions never enter the forward, so there's no
                        # k_nope to override here.
                        _ln_pic_mode = getattr(forward_batch, "pic_mode", None)
                        _ln_skip_split = _ln_pic_mode in ("pic_a3", "pic_cacheblend")
                        if (
                            not _ln_skip_split
                            and _pic_pub is not None
                            and _pic_pub.shape[0] == k_nope.shape[0]
                            and (_pic_pub >= 0).any()
                        ):
                            _hit = _pic_pub >= 0
                            _miss = ~_hit
                            _buf = get_token_to_kv_pool().get_key_buffer(self.layer_id)
                            k_nope = k_nope.clone()
                            if _miss.any():
                                k_nope[_miss] = self.kv_a_layernorm(k_nope[_miss])
                            if _hit.any():
                                # Load post-layernorm k_nope from public slot.
                                # MLA KV buffer shape is (num_slots, 1, kv_cache_dim)
                                # (single KV head — MQA). Index dim=1 with 0 to
                                # collapse the KV-head dim, THEN slice the last
                                # dim to grab k_nope.
                                k_nope[_hit] = _buf[_pic_pub[_hit], 0, : self.kv_lora_rank].to(k_nope.dtype)
                        else:
                            k_nope = self.kv_a_layernorm(k_nope)

            # q_lora needed by indexer
            if self.use_dsa:
                if q_lora is None:
                    q_lora = q

            # overlap q_b_proj and indexer during decode
            if (
                self.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
            ):
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = self.q_b_proj(q)[0].view(
                        -1, self.num_local_heads, self.qk_head_dim
                    )
                # skip_topk (shared) layers carry no indexer weights in the
                # checkpoint, so they must reuse the carried topk and never run
                # the indexer. Do NOT widen this to `or prev_topk_indices is
                # None` (the upstream gate): that recomputes with an
                # uninitialized indexer whenever cross-layer propagation is
                # unavailable (e.g. the TBO op path drops topk_indices),
                # reintroducing the >index_topk garbling. The is_nextn clause is
                # the sole intentional fallback (layer 78 has its own weights).
                if not self.skip_topk or (self.is_nextn and prev_topk_indices is None):
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
                else:
                    # skip_topk reuses prev layer's indices; mirror into this
                    # layer's slot so the captured buffer matches what's used.
                    topk_indices = maybe_capture_indexer_topk(
                        self.layer_id, prev_topk_indices
                    )
                current_stream.wait_stream(self.alt_stream)
            else:
                k_nope = k_nope.unsqueeze(1)
                q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
                if q_lora is not None:
                    # See the skip_topk note above: shared layers have no
                    # indexer weights, so this gate must not fall back to
                    # computing when prev_topk_indices is None.
                    if not self.skip_topk or (
                        self.is_nextn and prev_topk_indices is None
                    ):
                        topk_indices = self.indexer(
                            x=hidden_states,
                            q_lora=q_lora,
                            positions=positions,
                            forward_batch=forward_batch,
                            layer_id=self.layer_id,
                        )
                    else:
                        topk_indices = maybe_capture_indexer_topk(
                            self.layer_id, prev_topk_indices
                        )
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            # PIC split: kv_a_proj_with_mqa only for miss segments.
            # Hit segments load k_nope (post-layernorm) directly from the public KV slot,
            # avoiding the projection and layernorm for those token positions.
            _pic_pub = getattr(forward_batch, "pic_hit_pub_kv_loc", None)
            # pic_a3 / pic_cacheblend new-path: skip the hit K load — layer 0-1
            # is fresh-recompute-full-length (see plan §3.5); layer 2+ hit KV
            # comes from PICache-populated public slots via req_to_token_pool.
            # Mirror of the gate at :244 on the fused branch (previously missing
            # here — see plan §1.4 / §3.5 Branch A').
            _pic_mode_nonfused = getattr(forward_batch, "pic_mode", None)
            _ln_skip_split_nonfused = _pic_mode_nonfused in ("pic_a3", "pic_cacheblend")
            if (
                not _ln_skip_split_nonfused
                and _pic_pub is not None
                and _pic_pub.shape[0] == hidden_states.shape[0]
                and (_pic_pub >= 0).any()
            ):
                _hit = _pic_pub >= 0
                _miss = ~_hit
                _buf = get_token_to_kv_pool().get_key_buffer(self.layer_id)
                kv_dim = self.kv_lora_rank + self.qk_rope_head_dim
                latent_cache = hidden_states.new_empty(hidden_states.shape[0], kv_dim)
                # KV projection only for miss positions (real compute saving!)
                if _miss.any():
                    latent_cache[_miss] = self.kv_a_proj_with_mqa(hidden_states[_miss])[0]
                # Hit positions: load from public slot
                # k_nope part is post-layernorm; k_pe part is post-RoPE at old positions
                # (the delta-RoPE block below will correct k_pe to current positions)
                if _hit.any():
                    # MLA buffer is (num_slots, 1, kv_cache_dim) — collapse dim 1.
                    latent_cache[_hit] = _buf[_pic_pub[_hit], 0].to(latent_cache.dtype)
                # Layernorm: only for miss positions (hit already post-layernorm)
                k_nope = latent_cache[..., : self.kv_lora_rank].clone()
                if _miss.any():
                    k_nope[_miss] = self.kv_a_layernorm(k_nope[_miss])
                # k_nope[hit] stays as the cached post-layernorm value
            else:
                latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
                k_nope = self.kv_a_layernorm(latent_cache[..., : self.kv_lora_rank])
            k_nope = k_nope.unsqueeze(1)

        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)

        _kvb_q = None
        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            # Fork the kv_b q-correction A-step onto the LoRA side stream to overlap the bmm.
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_q_prepare,
            )

            _kvb_q = kv_b_lora_q_prepare(self, q_nope)

        if self.use_deep_gemm_bmm:
            (
                q_nope_val,
                q_nope_scale,
                masked_m,
                expected_m,
                aligned_m,
            ) = per_token_group_quant_mla_deep_gemm_masked_fp8(q_nope.transpose(0, 1))
            q_nope_out = q_nope.new_empty(
                (self.num_local_heads, aligned_m, self.kv_lora_rank)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (q_nope_val, q_nope_scale),
                (self.w_kc, self.w_scale_k),
                q_nope_out,
                masked_m,
                expected_m,
            )
            q_nope_out = q_nope_out[:, :expected_m, :]
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_kc.dtype == torch.uint8:
                x = q_nope.transpose(0, 1)
                q_nope_out = torch.empty(
                    x.shape[0],
                    x.shape[1],
                    self.w_kc.shape[2],
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_kc.transpose(-2, -1),
                    self.w_scale_k.transpose(-2, -1),
                    torch.bfloat16,
                    q_nope_out,
                )
            else:
                if (_use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn) or (
                    get_is_capture_mode() and self.w_kc.dtype == torch.float8_e4m3fnuz
                ):
                    # fp8 Triton kernel: always on gfx950,
                    # cudagraph-only on gfx942 (hides launch overhead)
                    q_nope_out = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=q_nope,
                        WQ=self.w_kc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,  # allocate (B, M, N)
                        transpose_bm=False,  # (B, M, N)
                        transpose_bm_in=True,  # (M, B, K)
                        dtype=torch.bfloat16,
                    )

                else:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.w_kc.to(torch.bfloat16) * self.w_scale,
                    )

        elif self.w_kc.dtype == torch.float8_e4m3fn:
            if _is_cpu:
                q_nope_out = torch.bmm(
                    q_nope.to(torch.bfloat16).transpose(0, 1),
                    self.w_kc.to(torch.bfloat16) * self.w_scale,
                )
            else:
                # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
                q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                    q_nope.transpose(0, 1),
                    (
                        torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                q_nope_out = bmm_fp8(
                    q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
                )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

        q_nope_out = q_nope_out.transpose(0, 1)

        # head-level KV probe: dump w_kc (per-head K up-proj) so per-head K can be
        # decompressed offline from the shared MLA latent. Dumped on ALL TP ranks
        # (rank-tagged) → all heads reconstructable. Constant weight → written once
        # per (layer, rank). Gated by SGLANG_PIC_KDUMP_WKC=<dir>.
        try:
            import os as _os_wc

            _wcd = _os_wc.environ.get("SGLANG_PIC_KDUMP_WKC", "")
            if _wcd and hasattr(self, "w_kc") and torch.is_tensor(self.w_kc):
                from sglang.srt.distributed.parallel_state import (
                    get_tensor_model_parallel_rank as _tprk_wc,
                )

                _rk = int(_tprk_wc())
                _wf = f"{_wcd}/wkc_L{int(self.layer_id)}_r{_rk}.pt"
                if not _os_wc.path.exists(_wf):
                    _os_wc.makedirs(_wcd, exist_ok=True)
                    _save = {"w_kc": self.w_kc.detach().float().cpu(),
                             "dtype": str(self.w_kc.dtype)}
                    for _a in ("w_scale", "w_scale_k"):
                        _v = getattr(self, _a, None)
                        if torch.is_tensor(_v):
                            _save[_a] = _v.detach().float().cpu()
                    torch.save(_save, _wf)
        except Exception:
            pass
        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_q_apply,
            )

            q_nope_out = kv_b_lora_q_apply(self, q_nope, q_nope_out, _kvb_q)
        elif is_kv_b_lora_active(self):
            q_nope_out = apply_kv_b_lora_q_correction(self, q_nope, q_nope_out)

        skip_rope_for_dsa_tilelang_fused = self._skip_rope_for_dsa_tilelang_fused()
        skip_rope_for_aiter_fused_mla = self._skip_rope_for_aiter_fused_mla()
        if (
            self.rotary_emb is not None
            and (not self._fuse_rope_for_trtllm_mla(forward_batch))
            and (not skip_rope_for_dsa_tilelang_fused)
            and (not skip_rope_for_aiter_fused_mla)
            and (not _use_aiter or not _is_gfx95_supported or self.use_dsa)
        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        # PIC transition_rope: for hit segments, correct k_nope and k_pe using
        # cached public KV + delta-RoPE so the attention sees K at the current
        # positions without re-running the fused projection for those tokens.
        #
        # delta-RoPE correctness:
        #   k_pe_old[i] = W_pe*x[i] * RoPE(old_start + i)   (stored in public slot)
        #   k_pe_new[i] = W_pe*x[i] * RoPE(new_start + i)
        #              = apply_rope(k_pe_old[i], delta)        delta = new_start - old_start
        # The delta is constant for every token within one segment, so calling
        # self.rotary_emb with a constant position tensor applies the correct shift.
        #
        # k_nope is position-free (W_kv*x after layernorm), so we directly load
        # the cached value — identical to freshly computing it for the same tokens.
        _pub_kv_loc = getattr(forward_batch, "pic_hit_pub_kv_loc", None)
        # Defensive shape check: PIC K-override only makes sense when the
        # tensor length matches the current batch length. In decode the tensor
        # should be None (cleared by prepare_for_decode); the shape guard
        # protects against leaked prefill tensors under continuous batching.
        if _pub_kv_loc is not None and _pub_kv_loc.shape[0] != k_nope.shape[0]:
            _pub_kv_loc = None
        # pic_a3 / pic_cacheblend: skip the K-override — hit positions never
        # enter the forward under IMP_ONLY. They were pre-populated into the
        # KV pool by _pic_prepopulate_hit_slots before this call.
        _pic_mode = getattr(forward_batch, "pic_mode", None)
        _skip_k_override = _pic_mode in ("pic_a3", "pic_cacheblend")

        if _pub_kv_loc is not None and not _skip_k_override:
            _hit = _pub_kv_loc >= 0
            if _hit.any():
                _pub_slots = _pub_kv_loc[_hit]
                _delta = forward_batch.pic_hit_delta_pos[_hit]

                # Load cached k_nope and k_pe from the public KV pool.
                # MLA buffer is (num_slots, 1, kv_cache_dim) — collapse dim 1
                # via [_, 0, _] indexing to get a 2D view.
                _kv_buf = get_token_to_kv_pool().get_key_buffer(self.layer_id)
                # Shape: (n_hit, kv_lora_rank + qk_rope_head_dim)
                _cached = _kv_buf[_pub_slots, 0].to(k_nope.dtype)

                # k_nope: position-free, copy directly  (n_hit, 1, kv_lora_rank)
                _k_nope_hit = _cached[:, : self.kv_lora_rank].unsqueeze(1)

                # k_pe: apply delta-RoPE to shift from old position to new position
                # (n_hit, 1, qk_rope_head_dim)
                _k_pe_old = _cached[:, self.kv_lora_rank :].unsqueeze(1)
                if self.rotary_emb is not None and _delta.any():
                    _dummy = torch.zeros_like(_k_pe_old)
                    _, _k_pe_hit = self.rotary_emb(_delta, _dummy, _k_pe_old)
                else:
                    _k_pe_hit = _k_pe_old

                k_nope = k_nope.clone()
                k_pe = k_pe.clone()
                k_nope[_hit] = _k_nope_hit
                k_pe[_hit] = _k_pe_hit

        if dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch):
            # support allgather+rerrange
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

        # ── pic_a3 new-path: stash latent Q/K for imp selection ──
        # Fires at every A³ check layer (forward_batch.a3_check_layers; default
        # (1,) == the single-layer pic_a3). Keyed per-layer in
        # pic_a3_stash_by_layer; pic_a3_layer1_stash kept as an alias so the
        # single-layer boundary in deepseek_v2 keeps working unchanged.
        #   q_nope_out: (N, H, kv_lora_rank)  — absorbed Q (= q_nope · w_kc)
        #   k_nope:     (N, 1, kv_lora_rank)  — post-layernorm K latent
        #   k_pe:       (N, 1, qk_rope_head_dim)  — post-RoPE at fresh positions
        #   q_pe:       (N, H, qk_rope_head_dim)  — post-RoPE
        # Clone to detach from the graph so the layer's attention still runs
        # freely; imp selection consumes the stash after the layer returns.
        _a3_check_layers = getattr(forward_batch, "a3_check_layers", (1,))
        if (
            getattr(forward_batch, "pic_a3_new_path", False)
            and int(self.layer_id) in _a3_check_layers
        ):
            _stash = {
                "q_absorbed": q_nope_out.detach().clone(),
                "k_latent": k_nope.detach().clone(),
                "q_pe": q_pe.detach().clone(),
                "k_pe": k_pe.detach().clone(),
                "softmax_scale": float(self.scaling),
                "kv_lora_rank": int(self.kv_lora_rank),
                "layer_id": int(self.layer_id),
            }
            if getattr(forward_batch, "pic_a3_stash_by_layer", None) is None:
                forward_batch.pic_a3_stash_by_layer = {}
            forward_batch.pic_a3_stash_by_layer[int(self.layer_id)] = _stash
            if int(self.layer_id) == 1:
                forward_batch.pic_a3_layer1_stash = _stash

        # ── K-dump 观测台: dump the QUERY segment's summed-Q per layer ──
        # For the offline attention-weighted-KV-deviation plot. MLA's K is
        # MQA-shared, so the A3 merged-head score factorizes:
        #   score[q,k] = (Σ_heads [q_absorbed, q_pe][q,h]) · K576[k] * merged_scale
        # → we only need Σ_heads of the query rows' [q_absorbed,q_pe] (576-dim,
        # tiny). Env-gated (SGLANG_PIC_KDUMP_DIR + _ALL), rank0, measure-prefill,
        # PIC-family only (needs pic_segments for the query seg). Overwrites per
        # forward (last=correctness sample). Isolated + try/except → zero risk off.
        try:
            import os as _os_qd

            _qd = _os_qd.environ.get("SGLANG_PIC_KDUMP_DIR", "")
            if (
                _qd
                and _os_qd.environ.get("SGLANG_PIC_KDUMP_ALL", "0") == "1"
                and forward_batch.forward_mode.is_extend()
                and forward_batch.seq_lens is not None
                and int(forward_batch.seq_lens[0].item()) >= 3000
            ):
                from sglang.srt.distributed.parallel_state import (
                    get_tensor_model_parallel_rank as _tprk_q,
                )

                _reqs_q = getattr(forward_batch, "_reqs_ref", None)
                _segs_q = (
                    getattr(_reqs_q[0], "pic_segments", None) if _reqs_q else None
                )
                if _tprk_q() == 0 and _segs_q and positions is not None:
                    _qs, _qe = _segs_q[-1]
                    _qmask = (positions >= int(_qs)) & (positions < int(_qe))
                    if bool(_qmask.any()):
                        _H = int(q_nope_out.shape[1])
                        _qsum = (
                            torch.cat([q_nope_out, q_pe], dim=-1)[_qmask]
                            .sum(dim=1)
                        )
                        _dpe = int(q_pe.shape[-1])
                        _scale = 1.0 / ((_H * (int(self.kv_lora_rank) + _dpe)) ** 0.5)
                        _os_qd.makedirs(_qd, exist_ok=True)
                        _tag_q = _os_qd.environ.get("SGLANG_PIC_KDUMP_TAG", "modeX")
                        torch.save(
                            {
                                "q": _qsum.detach().float().cpu(),
                                "scale": float(_scale),
                                "qpos": positions[_qmask].detach().long().cpu(),
                            },
                            f"{_qd}/{_tag_q}_Q_L{int(self.layer_id)}.pt",
                        )
        except Exception:
            pass

        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
        )

    def project_latent_qk_from_normed(
        self: DeepseekV2AttentionMLA,
        normed_hidden: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> dict:
        """pic_a3_oracle Route-A: recompute latent Q/K from a normalized hidden
        state, for imp re-selection scoring ONLY. Returns the same dict shape as
        the forward_absorb_prepare stash (q_absorbed/k_latent/q_pe/k_pe/...).
        Forces bf16 for the w_kc absorption bmm (scoring is a topk, insensitive
        to quant error). No side effects: no KV write, no stash, no forward_batch
        mutation. Assumes the fused q_lora path (DeepSeek-V3 / GLM default).
        """
        assert (
            self.q_lora_rank is not None
        ), "Route-A projection assumes the fused q_lora path"
        latent = self.prepare_qkv_latent(normed_hidden, forward_batch)
        q, latent_cache = latent.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
        )
        # Full layernorm (no PIC split — Route-A always recomputes all rows).
        q = self.q_a_layernorm(q)
        k_nope = self.kv_a_layernorm(latent_cache[..., : self.kv_lora_rank]).unsqueeze(
            1
        )
        q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        k_pe = latent_cache[..., self.kv_lora_rank :].unsqueeze(1)
        # w_kc absorption forced bf16 (mirror the bf16 bmm branch at :442-467).
        if self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_nope_out = q_nope_out.transpose(0, 1)
        if self.rotary_emb is not None:
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        return {
            "q_absorbed": q_nope_out,
            "k_latent": k_nope,
            "q_pe": q_pe,
            "k_pe": k_pe,
            "softmax_scale": float(self.scaling),
            "kv_lora_rank": int(self.kv_lora_rank),
            "layer_id": int(self.layer_id),
        }

    def forward_absorb_core(
        self: DeepseekV2AttentionMLA,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
    ):
        save_kv_cache = True

        if self.current_attention_backend in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            if (
                self._skip_rope_for_dsa_tilelang_fused()
                and self.rotary_emb is not None
            ):
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache
                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )
                q_cat, _, k_pe_fused, _ = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    get_token_to_kv_pool().get_key_buffer(self.attn_mqa.layer_id),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )
                save_kv_cache = False
                # On decode, pass q_cat directly to attn_mqa with q_rope=None so
                # dsa_backend.forward_decode reuses q_cat as a zero-copy view
                # (`q.contiguous().view(...)` fast-path) instead of running the
                # redundant `concat_mla_absorb_q_general(q_nope_fused, q_pe_fused)`
                # that would otherwise rebuild a tensor byte-identical to q_cat.
                # On ROCm tilelang decode, this eliminates the
                # `CatArrayBatchedCopy<OpaqueType<1u>, ...>` kernel that used to
                # fire once per layer per decode step (~2.6 us / layer saved).
                # Prefill keeps the split form because dsa_backend.forward_extend
                # asserts `q_rope is not None`.
                if forward_batch.forward_mode.is_decode_or_idle():
                    if llama_4_scaling is not None:
                        # llama_4_scaling applies only to the q_nope portion;
                        # mutate in place via the slice view of q_cat.
                        q_cat[..., : self.kv_lora_rank] *= llama_4_scaling
                    attn_output = self.attn_mqa(
                        q_cat,
                        None,
                        None,
                        forward_batch,
                        q_rope=None,
                        k_rope=k_pe_fused,
                        save_kv_cache=save_kv_cache,
                        **(
                            dict(topk_indices=topk_indices)
                            if topk_indices is not None
                            else {}
                        ),
                    )
                else:
                    q_nope_fused = q_cat[..., : self.kv_lora_rank]
                    q_pe_fused = q_cat[..., self.kv_lora_rank :]
                    if llama_4_scaling is not None:
                        q_nope_fused *= llama_4_scaling
                    attn_output = self.attn_mqa(
                        q_nope_fused,
                        None,
                        None,
                        forward_batch,
                        q_rope=q_pe_fused,
                        k_rope=k_pe_fused,
                        save_kv_cache=save_kv_cache,
                        **(
                            dict(topk_indices=topk_indices)
                            if topk_indices is not None
                            else {}
                        ),
                    )
            else:
                extra_args = {}
                if self._fuse_rope_for_trtllm_mla(forward_batch):
                    extra_args = {
                        "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                        "is_neox": self.rotary_emb.is_neox_style,
                        "llama_4_scaling": llama_4_scaling,
                    }
                attn_output = self.attn_mqa(
                    q_nope_out,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=q_pe,
                    k_rope=k_pe,
                    **extra_args,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
        else:
            if _use_aiter_gfx95:
                cos = self.rotary_emb.cos_cache
                sin = self.rotary_emb.sin_cache

                kv_cache_dtype = (
                    fp8_dtype if self.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
                )

                q, _, _, k = fused_qk_rope_cat_and_cache_mla(
                    q_nope_out,
                    q_pe,
                    k_nope,
                    k_pe,
                    get_token_to_kv_pool().get_key_buffer(self.attn_mqa.layer_id),
                    forward_batch.out_cache_loc,
                    positions,
                    cos,
                    sin,
                    self.attn_mqa.k_scale,
                    self.rotary_emb.is_neox_style,
                    q_out_dtype=kv_cache_dtype,
                )

                save_kv_cache = False
            else:
                q = torch.cat([q_nope_out, q_pe], dim=-1)
                k = torch.cat([k_nope, k_pe], dim=-1)

            # Apply llama 4 scaling if provided
            if llama_4_scaling is not None:
                q *= llama_4_scaling

            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        _kvb_v = None
        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            # Fork the kv_b v-correction A-step onto the LoRA side stream to overlap the bmm.
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_v_prepare,
            )

            _kvb_v = kv_b_lora_v_prepare(self, attn_output)

        if self.use_deep_gemm_bmm:
            (
                attn_output_val,
                attn_output_scale,
                masked_m,
                expected_m,
                aligned_m,
            ) = per_token_group_quant_mla_deep_gemm_masked_fp8(
                attn_output.transpose(0, 1)
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif _is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            if _use_aiter_gfx95 and self.w_vc.dtype == torch.uint8:
                x = attn_output.transpose(0, 1)
                B_heads, M_batch = x.shape[0], x.shape[1]
                N_vdim = self.w_vc.shape[2]
                # Allocate in (batch, heads, dim) so the post-GEMM
                # transpose+flatten is a free view instead of a copy.
                _bmm_buf = torch.empty(
                    M_batch,
                    B_heads,
                    N_vdim,
                    device=x.device,
                    dtype=torch.bfloat16,
                )
                attn_bmm_output = _bmm_buf.transpose(0, 1)
                batched_gemm_afp4wfp4_pre_quant(
                    x,
                    self.w_vc.transpose(-2, -1),
                    self.w_scale_v.transpose(-2, -1),
                    torch.bfloat16,
                    attn_bmm_output,
                )
            else:
                _bmm_buf = None
                if _use_aiter_gfx95 and self.w_kc.dtype == torch.float8_e4m3fn:
                    attn_bmm_output = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
                        X=attn_output,
                        WQ=self.w_vc.transpose(-1, -2),
                        w_scale=self.w_scale,
                        group_size=128,
                        YQ=None,
                        transpose_bm=False,
                        transpose_bm_in=True,
                        dtype=torch.bfloat16,
                    )
                else:
                    attn_bmm_output = torch.bmm(
                        attn_output.to(torch.bfloat16).transpose(0, 1),
                        self.w_vc.to(torch.bfloat16) * self.w_scale,
                    )

            if _bmm_buf is not None:
                # _bmm_buf is already (batch, heads, dim) contiguous
                if self.o_proj.weight.dtype == torch.uint8:
                    attn_bmm_output = fused_flatten_mxfp4_quant(_bmm_buf)
                elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                    attn_bmm_output = fused_flatten_fp8_group_quant(
                        _bmm_buf, group_size=128, dtype_quant=torch.float8_e4m3fn
                    )
                else:
                    attn_bmm_output = _bmm_buf.flatten(1, 2)
            elif self.o_proj.weight.dtype == torch.uint8:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_mxfp4_quant(attn_bmm_output)
            elif self.o_proj.weight.dtype == torch.float8_e4m3fn:
                attn_bmm_output = attn_bmm_output.transpose(0, 1)
                attn_bmm_output = fused_flatten_fp8_group_quant(
                    attn_bmm_output, group_size=128, dtype_quant=torch.float8_e4m3fn
                )
            else:
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)

        elif self.w_vc.dtype == torch.float8_e4m3fn:
            if _is_cpu:
                attn_bmm_output = torch.bmm(
                    attn_output.to(torch.bfloat16).transpose(0, 1),
                    self.w_vc.to(torch.bfloat16) * self.w_scale,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
            else:
                attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                    attn_output.transpose(0, 1),
                    (
                        torch.zeros(
                            (1,), dtype=torch.float32, device=attn_output.device
                        )
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                attn_bmm_output = bmm_fp8(
                    attn_output_val,
                    self.w_vc,
                    attn_output_scale,
                    self.w_scale,
                    torch.bfloat16,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        elif _is_musa:
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1), self.w_vc
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            if is_in_piecewise_cuda_graph():
                # torch dynamo requires out= op was called where output tensor was non-contiguous
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                torch.bmm(
                    attn_output.transpose(0, 1),
                    self.w_vc,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_v_apply,
            )

            attn_bmm_output = kv_b_lora_v_apply(
                self, attn_output, attn_bmm_output, _kvb_v
            )
        elif is_kv_b_lora_active(self):
            attn_bmm_output = apply_kv_b_lora_v_correction(
                self, attn_output, attn_bmm_output
            )
        output, _ = self.o_proj(attn_bmm_output)

        if self.next_skip_topk is None:
            return output

        # Return topk_indices for the next layer when enabling index cache
        if not self.next_skip_topk:
            return output, None
        else:
            return output, topk_indices

    def _fuse_rope_for_trtllm_mla(
        self: DeepseekV2AttentionMLA, forward_batch: ForwardBatch
    ) -> bool:
        """
        Check if we should skip rope and do fused rope+quantize for TRTLLM MLA decode in fp8_e4m3 path.
        """
        if self.current_attention_backend in ("dsa", "nsa"):
            return (
                get_global_server_args().dsa_decode_backend == "trtllm"
                or get_global_server_args().dsa_prefill_backend == "trtllm"
            ) and get_attn_backend().kv_cache_dtype == torch.float8_e4m3fn

        return (
            self.current_attention_backend
            in ("trtllm_mla", "tokenspeed_mla", "cutedsl_mla")
            and (
                forward_batch.forward_mode.is_decode_or_idle()
                or forward_batch.forward_mode.is_target_verify()
            )
            and get_attn_backend().data_type == torch.float8_e4m3fn
        )

    def _skip_rope_for_dsa_tilelang_fused(self: DeepseekV2AttentionMLA) -> bool:
        """
        Check if we should skip rope and use fused rope+cache path for TileLang DSA on gfx95.
        """
        server_args = get_global_server_args()
        return (
            _use_aiter_gfx95
            and self.current_attention_backend in ("dsa", "nsa")
            and (
                server_args.dsa_decode_backend == "tilelang"
                or server_args.dsa_prefill_backend == "tilelang"
            )
        )

    def _skip_rope_for_aiter_fused_mla(self: DeepseekV2AttentionMLA) -> bool:
        """
        Skip rope in prepare and let the fused kernel in forward_absorb_core handle it,
        when running aiter-backend MLA on gfx95 (i.e., the `else` branch in forward_absorb_core
        that calls fused_qk_rope_cat_and_cache_mla).
        """
        return (
            _use_aiter_gfx95
            and self.current_attention_backend
            not in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
        )
