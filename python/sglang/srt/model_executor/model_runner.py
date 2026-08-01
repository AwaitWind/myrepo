# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ModelRunner runs the forward passes of the models."""

from __future__ import annotations

import contextlib
import datetime
import gc
import hashlib
import inspect
import logging
import os
import socket
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from sglang.jit_kernel.ngram_embedding import update_token_table_decode
from sglang.srt.compilation.piecewise_context_manager import (
    enable_piecewise_cuda_graph,
    set_forward_context,
)
from sglang.srt.configs import (
    BailingHybridConfig,
    FalconH1Config,
    GraniteMoeHybridConfig,
    InternS2PreviewConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    Lfm2Config,
    Lfm2MoeConfig,
    Lfm2VlConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
)
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    ModelImpl,
    get_num_indexer_layers,
)
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed import (
    get_default_distributed_backend,
    get_pp_group,
    get_tp_group,
    get_world_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    join_process_groups,
    try_recover_ranks,
)
from sglang.srt.elastic_ep.expert_backup_client import ExpertBackupClient
from sglang.srt.environ import envs
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionMetrics,
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    broadcast_global_expert_location_metadata,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.kv_canary.api import install_canary
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.kv_canary.token_oracle.install import install_token_oracle_from_env
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    attn_backend_wrapper,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_group,
    get_attention_tp_size,
    initialize_dp_attention,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.layers.utils.cp_utils import is_mla_prefill_cp_enabled
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import sanity_check_mm_pad_shift_value
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.breakable_cuda_graph_runner import (
    BreakableCudaGraphRunner,
)
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    CudaGraphBufferRegistry,
    build_decode_registry,
    build_prefill_registry,
)
from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    _allocate_decode_buffers,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.hook_manager import register_forward_hooks
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
    PiecewiseCudaGraphRunner,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.platforms import current_platform
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.state_capturer.base import TopkCaptureOutput
from sglang.srt.state_capturer.indexer_topk import (
    create_indexer_capturer,
    get_global_indexer_capturer,
    set_global_indexer_capturer,
)
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCapturer,
    get_global_experts_capturer,
    set_global_experts_capturer,
)
from sglang.srt.utils import (
    MultiprocessingSerializer,
    broadcast_pyobj,
    cpu_has_amx_support,
    dynamic_import,
    empty_context,
    enable_show_time_cost,
    get_available_gpu_memory,
    get_bool_env_var,
    get_cpu_ids_by_node,
    init_custom_process_group,
    is_hip,
    is_host_cpu_arm64,
    is_npu,
    log_info_on_rank0,
    monkey_patch_p2p_access_check,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_tp_gather,
    reserve_rope_cache_for_long_sequences,
    set_cuda_arch,
    slow_rank_detector,
)
from sglang.srt.utils.common import ceil_align, next_power_of_2, require_mlp_sync
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto
from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
from sglang.srt.utils.offloader import (
    create_offloader_from_server_args,
    get_offloader,
    set_offloader,
)
from sglang.srt.utils.patch_torch import (
    monkey_patch_torch_reductions,
    register_sgl_tp_rank,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_checker import WeightChecker
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()
elif current_platform.is_out_of_tree():
    current_platform.init_backend()

MLA_ATTENTION_BACKENDS = [
    "aiter",
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "flashmla",
    "cutedsl_mla",
    "cutlass_mla",
    "trtllm_mla",
    "tokenspeed_mla",
    "ascend",
    "dsa",
    "nsa",  # Deprecated alias for "dsa"
    "intel_xpu",
]

CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = [
    "flashinfer",
    "fa3",
    "fa4",
    "flashmla",
    "cutedsl_mla",
    "cutlass_mla",
    "trtllm_mla",
    "tokenspeed_mla",
]

TORCH_DTYPE_TO_KV_CACHE_STR = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.bfloat16: "bf16",
}


def add_mla_attention_backend(backend_name):
    if backend_name not in MLA_ATTENTION_BACKENDS:
        MLA_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to MLA_ATTENTION_BACKENDS.")


def add_chunked_prefix_cache_attention_backend(backend_name):
    if backend_name not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS:
        CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS.append(backend_name)
        logger.info(
            f"Added {backend_name} to CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS."
        )


# Detect stragger ranks in model loading
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing


logger = logging.getLogger(__name__)

_UNSET: Any = object()


def resolve_language_model(model: nn.Module) -> nn.Module:
    model_cls_name = model.__class__.__name__
    if model_cls_name == "Qwen3OmniMoeForConditionalGeneration":
        return model.thinker.model
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "language_model"):
        return model.language_model
    return model.model


class RankZeroFilter(logging.Filter):
    """Filter that only allows INFO level logs from rank 0, but allows all other levels from any rank."""

    def __init__(self, is_rank_zero):
        super().__init__()
        self.is_rank_zero = is_rank_zero

    def filter(self, record):
        if record.levelno == logging.INFO:
            return self.is_rank_zero
        return True


@dataclass
class ModelRunnerOutput:
    logits_output: Union[LogitsProcessorOutput, PPProxyTensors]
    can_run_graph: bool
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None
    routed_experts_output: Optional[TopkCaptureOutput] = None
    indexer_topk_output: Optional[TopkCaptureOutput] = None


@dataclass
class _EagerBufferRegistry:
    # Lazily-built eager input-buffer registry plus the capacity it was sized to.
    registry: Optional["CudaGraphBufferRegistry"] = None
    max_bs: int = 0
    max_num_tokens: int = 0


class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models."""

    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        attn_cp_rank: Optional[int] = None,
        moe_dp_rank: Optional[int] = None,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        draft_model_idx: Optional[int] = None,
    ):
        # Parse args
        self.mem_fraction_static = mem_fraction_static
        # Set on target by `_resolve_memory_pool_config`; passed in for draft
        # workers so they reuse target's resolved sizes (replaces legacy
        # `server_args._draft_pool_config` mutation hack).
        self.memory_pool_config = memory_pool_config
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.dp_rank = dp_rank
        self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.attn_cp_rank = attn_cp_rank
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dp_rank = moe_dp_rank
        self.moe_dp_size = server_args.moe_dp_size
        self.model_config = model_config
        self.dist_port = nccl_port
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.device_timer = None
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = getattr(
            model_config, "is_hybrid_swa_compress", False
        )
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size
        rope_scaling = getattr(
            model_config.hf_text_config, "rope_parameters", None
        ) or getattr(model_config.hf_text_config, "rope_scaling", {})
        self.model_is_mrope = (
            rope_scaling is not None and "mrope_section" in rope_scaling
        )
        self.enable_elastic_ep = server_args.elastic_ep_backend is not None
        self.forward_pass_id = 0
        self.init_new_workspace = False
        self._eager_decode_registry = _EagerBufferRegistry()
        self._eager_prefill_registry = _EagerBufferRegistry()
        self.draft_model_idx = draft_model_idx
        self.enable_hisparse = server_args.enable_hisparse

        self.remote_instance_transfer_engine = None
        self.remote_instance_transfer_engine_session_id = ""
        self.remote_instance_transfer_engine_weight_info = None

        self.msprobe_debugger = None
        if server_args.msprobe_dump_config is not None:
            self.init_msprobe()

        # auxiliary hidden capture mode. TODO: expose this to server args?
        self.eagle_use_aux_hidden_state = False
        self.dflash_use_aux_hidden_state = False
        self.dflash_target_layer_ids = None
        self.dflash_draft_num_layers = None
        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
            # load draft config
            draft_model_config = ModelConfig.from_server_args(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            self.eagle_use_aux_hidden_state = True

            try:
                # get the aux layer from draft model config
                eagle_config = getattr(
                    draft_model_config.hf_config, "eagle_config", None
                )
                self.eagle_use_aux_hidden_state = eagle_config.get(
                    "use_aux_hidden_state", True
                )
                self.eagle_aux_hidden_state_layer_ids = eagle_config[
                    "eagle_aux_hidden_state_layer_ids"
                ]
            except:
                # if there is no aux layer, set to None
                self.eagle_aux_hidden_state_layer_ids = None

        if self.spec_algorithm.is_dflash() and not self.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import (
                parse_dflash_draft_config,
            )

            # Select target layers to capture for building DFlash context features.
            draft_model_config = ModelConfig.from_server_args(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            dflash_draft_config = parse_dflash_draft_config(
                draft_hf_config=draft_model_config.hf_config
            )
            draft_num_layers = dflash_draft_config.require_num_layers()
            trained_target_layers = dflash_draft_config.num_target_layers

            target_num_layers = getattr(
                self.model_config.hf_text_config, "num_hidden_layers", None
            )
            if target_num_layers is None:
                raise ValueError(
                    "DFLASH requires target num_hidden_layers in config. "
                    f"Got target={target_num_layers}."
                )
            target_num_layers = int(target_num_layers)

            if (
                trained_target_layers is not None
                and trained_target_layers != target_num_layers
            ):
                logger.warning(
                    "DFLASH draft config num_target_layers=%s differs from runtime target num_hidden_layers=%s; "
                    "selecting capture layers based on the runtime target model.",
                    trained_target_layers,
                    target_num_layers,
                )

            self.dflash_use_aux_hidden_state = True
            self.dflash_draft_num_layers = int(draft_num_layers)
            self.dflash_target_layer_ids = dflash_draft_config.resolve_target_layer_ids(
                target_num_layers=int(target_num_layers),
                draft_num_layers=int(draft_num_layers),
            )

        # Apply the rank zero filter to logger
        if server_args.show_time_cost:
            enable_show_time_cost()

        # Model-specific adjustment
        self.model_specific_adjustment()

        # Set the global server_args in the scheduler process
        set_global_server_args_for_scheduler(server_args)
        global_server_args = get_global_server_args()

        # FIXME: hacky set `use_mla_backend`
        global_server_args.use_mla_backend = self.use_mla_backend

        # Init OpenMP threads binding for CPU
        if self.device == "cpu":
            self.init_threads_binding()

        # Get available memory before model loading
        pre_model_load_memory = self.init_torch_distributed()

        # Initialize MooncakeTransferEngine
        self.init_shared_mooncake_transfer_engine()

        # Init forward stream for overlap schedule
        self.forward_stream = torch.get_device_module(self.device).Stream()

        # CPU offload
        set_offloader(create_offloader_from_server_args(server_args, dp_rank=dp_rank))

        self._weight_checker = WeightChecker(model_runner=self)

        if envs.SGLANG_DETECT_SLOW_RANK.get():
            slow_rank_detector.execute()

        # Init mindspore running environment when model impl is "mindspore"
        self.init_mindspore_runner()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # For hisparse (must be set before initialize() so CUDA graph capture can see it)
        self.hisparse_coordinator = None

        self._linear_attn_registry_cache: Any = _UNSET

        # Initialize the model runner
        self.initialize(pre_model_load_memory)
        self.check_quantized_moe_compatibility()

        if (
            self.server_args.elastic_ep_backend is not None
            and self.server_args.elastic_ep_rejoin
        ):
            join_process_groups()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=True
                )
            )
            ElasticEPStateManager.instance().reset()

        if self.is_multimodal:
            sanity_check_mm_pad_shift_value(self.model_config.vocab_size)

        # Temporary cached values
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

        if self.pp_size > 1:
            assert (
                self.support_pp
            ), "Pipeline Parallel is not compatible with this model."

        # For weight updates
        self._model_update_group = {}
        self._weights_send_group = {}

    def init_msprobe(self):
        # Init the msprobe
        try:
            from msprobe.pytorch import PrecisionDebugger, seed_all
        except ImportError:
            logger.warning(
                "Please install msprobe for tensor data dump: pip install mindstudio-probe --pre, "
                "see https://gitcode.com/Ascend/msprobe for details."
            )
            return
        seed_all(mode=True)
        self.msprobe_debugger = PrecisionDebugger(
            config_path=self.server_args.msprobe_dump_config
        )

    def init_mindspore_runner(self):
        # Init the mindspore runner
        # for now, there is only some communication initialization work
        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE and _is_npu:
            from sglang.srt.model_executor.mindspore_runner import init_ms_distributed

            init_ms_distributed(
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                server_args=self.server_args,
                port=self.dist_port,
            )

    def initialize(self, pre_model_load_memory: float):
        server_args = self.server_args

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        if self.server_args.remote_instance_weight_loader_use_transfer_engine():
            self.remote_instance_init_transfer_engine()

        if not self.is_draft_worker:
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(
                    server_args=server_args,
                    model_config=self.model_config,
                    moe_ep_rank=self.moe_ep_rank,
                )
            )
            if self.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get():
                logger.info(
                    f"Initial expert_location_metadata: {get_global_expert_location_metadata()}"
                )

            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.tp_rank,
                )
            )

        # Expert parallelism
        self.eplb_manager = (
            EPLBManager(self)
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )
        self.expert_location_updater = ExpertLocationUpdater()

        if self.server_args.elastic_ep_backend:
            ElasticEPStateManager.init(self.server_args)
        self._token_oracle_manager = install_token_oracle_from_env(
            server_args=server_args,
            vocab_size=self.model_config.vocab_size,
        )
        # Load the model
        self.sampler = create_sampler()
        self.load_model()
        self._prepare_moe_topk()

        # Load the expert backup client
        self.expert_backup_client = (
            ExpertBackupClient(self.server_args, self)
            if (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
            else None
        )

        if (
            self.server_args.remote_instance_weight_loader_use_transfer_engine()
            # ModelExpress owns TransferEngine memory registration and metadata
            # publishing for backend=modelexpress. Re-registering here would
            # overlap the same weight buffers.
            and self.server_args.remote_instance_weight_loader_backend
            != RemoteInstanceWeightLoaderBackend.MODELEXPRESS
            and self.remote_instance_transfer_engine is not None
            and self.remote_instance_transfer_engine_weight_info is None
        ):
            # Register memory and upstream the transfer engine info to the bootstrap server
            self.remote_instance_transfer_engine_weight_info = register_memory_region(
                self.model, self.remote_instance_transfer_engine
            )
            self._register_to_engine_info_bootstrap()

        # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
        # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
        # determine the number of layers.
        # Some EAGLE3 drafts (e.g. nvidia/Kimi-K2.5-Thinking-Eagle3) carry the full DeepSeek-V3
        # config schema and explicitly set `num_nextn_predict_layers: 0`. Treat that the same as
        # the field being absent — otherwise the draft worker takes the MTP branch below with
        # model_num_layers=0, sizing the draft KV pool to zero and producing an IndexError on
        # the first forward (`set_mla_kv_buffer` -> `self.kv_buffer[layer_id - self.start_layer]`).
        _nnpl = self.model_config.num_nextn_predict_layers
        model_has_mtp_layers = _nnpl is not None and _nnpl > 0
        model_num_layers = (
            self.model_config.num_nextn_predict_layers
            if self.is_draft_worker and model_has_mtp_layers
            else max(
                self.model_config.num_hidden_layers,
                self.model_config.num_attention_layers,
            )
        )
        if self.model_config.hf_config.architectures[0] == "MiMoV2MTP":
            model_num_layers = 1
        elif self.model_config.hf_config.architectures[0] == "Step3p5MTP":
            model_num_layers = 1
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", model_num_layers)
        self.num_effective_layers = self.end_layer - self.start_layer

        self.adjust_hybrid_swa_layers_for_pp()

        # For LoopCoder models, each loop has its own layer_id, so we need to multiply by loop_num
        loop_num = getattr(self.model_config.hf_config, "loop_num", 1)
        if loop_num > 1:
            self.num_effective_layers = self.num_effective_layers * loop_num

        assert (
            (not model_has_mtp_layers)
            or (self.spec_algorithm.is_none())
            or (
                (not self.spec_algorithm.is_none())
                and (self.num_effective_layers == model_num_layers)
            )
        ), "PP is not compatible with MTP models."

        # Apply torchao quantization
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            apply_torchao_config_to_model(
                self.model, get_global_server_args().torchao_config
            )

        # Apply torch TP if the model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

        # Init lora
        if server_args.enable_lora:
            self.init_lora_manager()
            if not server_args.disable_cuda_graph:
                # Phase 1 of LoRA CUDA graph init: pre-allocate large MoE
                # intermediate buffers before init_memory_pool() so memory
                # profiling accounts for them.  Phase 2 (dense LoRA batch
                # metadata) is handled in CudaGraphRunner.__init__() via
                # lora_manager.init_cuda_graph_batch_info().
                self._init_lora_cuda_graph_moe_buffers()

        # Enable batch invariant mode
        if server_args.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

        # Deduce KV cache dtype
        self.configure_kv_cache_dtype()

        # Init memory pool and attention backends
        self.init_memory_pool(pre_model_load_memory)

        # Must be called AFTER init_memory_pool so the pool object exists for
        # canary to monkey-patch, and BEFORE init_device_graphs so warmup
        # forwards captured into the graph see the patched pool methods.
        self.canary_manager = install_canary(
            server_args=server_args,
            model_runner=self,
            token_oracle_manager=self._token_oracle_manager,
        )

        # Init ngram embedding token table
        self.maybe_init_ngram_embedding()

        # Init routed experts capturer
        self.init_routed_experts_capturer()

        self.init_indexer_capturer()

        # TODO: Refactor device-specific init branches into platform interface (separate PR).
        # Must be called BEFORE init_device_graphs() so CUDA graph capture
        # runs with aux hidden state capture enabled.
        self.init_aux_hidden_state_capture()

        if self.device == "cuda" or self.device == "musa":
            self.init_cublas()
            if self.enable_hisparse:
                from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
                from sglang.srt.mem_cache.sparsity import parse_hisparse_config

                hisparse_cfg = parse_hisparse_config(self.server_args)
                hisparse_top_k = getattr(
                    self.model_config.hf_text_config, "index_topk", hisparse_cfg.top_k
                )
                self.hisparse_coordinator = HiSparseCoordinator(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    top_k=hisparse_top_k,
                    device_buffer_size=hisparse_cfg.device_buffer_size,
                    device=self.device,
                    tp_group=(
                        self.attention_tp_group.cpu_group
                        if self.server_args.enable_dp_attention
                        else self.tp_group.cpu_group
                    ),
                    host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
                )
            self.init_attention_backend()
            self.kernel_warmup()
            self._pre_initialize_flashinfer_allreduce_workspace()
            self.init_device_graphs()
        elif self.device == "cpu":
            self.init_attention_backend()
            self.init_device_graphs()
        elif self.device == "npu":
            self.init_attention_backend()
            # lazy init for zbal with mix mode(before graph capture when enable_cuda_graph)
            if envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0 and not self.is_draft_worker:
                from sglang.srt.hardware_backend.npu.utils import lazy_init_zbal_gva_mem

                lazy_init_zbal_gva_mem(
                    self.device,
                    self.gpu_id,
                    get_world_group().rank_in_group,
                    get_world_group().world_size,
                    get_world_group().cpu_group,
                )
            self.init_device_graphs()
        elif current_platform.is_out_of_tree():
            self.init_attention_backend()
            if current_platform.support_cuda_graph():
                self.init_device_graphs()
            else:
                self.graph_runner = None
                self.graph_mem_usage = 0
        else:
            self.graph_runner = None
            self.graph_mem_usage = 0
            self.init_attention_backend()

        if server_args.forward_hooks:
            register_forward_hooks(self.model, server_args.forward_hooks)

        # Initialize piecewise CUDA graph
        self.init_piecewise_cuda_graphs()

        self.prealloc_symmetric_memory_pool()

        if self.canary_manager is not None and not self.is_draft_worker:
            self.canary_manager.mark_init_finished()

    def adjust_hybrid_swa_layers_for_pp(self):
        if not self.is_hybrid_swa:
            return

        if self.model_config.is_deepseek_v4_arch:
            return

        full_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "full_attention_layer_ids")
            and layer_idx in self.model_config.full_attention_layer_ids
        ]
        swa_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "swa_attention_layer_ids")
            and layer_idx in self.model_config.swa_attention_layer_ids
        ]
        self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
        self.model_config.full_attention_layer_ids = full_attention_layer_ids

    def init_routed_experts_capturer(self):
        if not self.server_args.disable_shared_experts_fusion and hasattr(
            self.model, "num_fused_shared_experts"
        ):
            num_fused_shared_experts = self.model.num_fused_shared_experts
        else:
            num_fused_shared_experts = 0

        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                enable=get_global_server_args().enable_return_routed_experts,
                model_config=self.model_config,
                num_fused_shared_experts=num_fused_shared_experts,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_indexer_capturer(self):
        enable = get_global_server_args().enable_return_indexer_topk
        # Producer wiring is CUDA-only (Indexer.forward_cuda + MLA skip_topk
        # path); other backends would create a capturer but never feed it.
        if enable and self.device != "cuda":
            logger.warning(
                "indexer-topk capture is CUDA-only; %s backend not yet wired. "
                "Disabling capturer.",
                self.device,
            )
            set_global_indexer_capturer(None)
            return

        hf_text_config = self.model_config.hf_text_config
        num_indexer_layers = get_num_indexer_layers(hf_text_config)
        index_topk = getattr(hf_text_config, "index_topk", 0)
        set_global_indexer_capturer(
            create_indexer_capturer(
                enable=enable,
                num_indexer_layers=num_indexer_layers,
                index_topk=index_topk,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    def init_aux_hidden_state_capture(self):
        """Configure auxiliary hidden state capture for speculative decoding.

        Must be called before CUDA graph capture so the captured graphs
        include aux hidden state output paths.
        """
        if self.eagle_use_aux_hidden_state:
            self.model.set_eagle3_layers_to_capture(
                self.eagle_aux_hidden_state_layer_ids
            )
        if self.dflash_use_aux_hidden_state:
            if not hasattr(self.model, "set_dflash_layers_to_capture"):
                raise ValueError(
                    f"Model {self.model.__class__.__name__} does not implement "
                    "set_dflash_layers_to_capture, which is required for DFLASH."
                )
            self.model.set_dflash_layers_to_capture(self.dflash_target_layer_ids)

    def remote_instance_init_transfer_engine(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake"
            )
            return
        self.remote_instance_transfer_engine = TransferEngine()
        local_ip = get_local_ip_auto()
        self.remote_instance_transfer_engine.initialize(
            local_ip,
            "P2PHANDSHAKE",
            envs.MOONCAKE_PROTOCOL.get(),
            envs.MOONCAKE_DEVICE.get(),
        )
        self.remote_instance_transfer_engine_session_id = NetworkAddress(
            local_ip, self.remote_instance_transfer_engine.get_rpc_port()
        ).to_host_port_str()

    def _register_to_engine_info_bootstrap(self):
        """Register transfer engine info with the EngineInfoBootstrapServer via HTTP PUT.

        The bootstrap server runs on node_rank==0. For multi-node setups, the
        host is derived from dist_init_addr. For single-node, use 127.0.0.1.
        """
        import requests as http_requests

        if self.server_args.dist_init_addr:
            # Multi-node: bootstrap server is on the head node (node_rank==0).
            # Derive host from dist_init_addr (shared across all nodes).
            bootstrap_host = (
                NetworkAddress.parse(self.server_args.dist_init_addr).resolved().host
            )
        else:
            bootstrap_host = "127.0.0.1"

        bootstrap_port = self.server_args.engine_info_bootstrap_port
        bootstrap_na = NetworkAddress(bootstrap_host, bootstrap_port)
        url = f"{bootstrap_na.to_url()}/register_transfer_engine_info"

        payload = {
            "tp_rank": self.tp_rank,
            "transfer_engine_info": {
                "session_id": self.remote_instance_transfer_engine_session_id,
                "weights_info_dict": self.remote_instance_transfer_engine_weight_info,
            },
        }

        try:
            resp = http_requests.put(url, json=payload, timeout=5)
            if resp.status_code == 200:
                logger.info(
                    f"Registered transfer engine info for tp_rank={self.tp_rank} "
                    f"with bootstrap server at {bootstrap_na}"
                )
            else:
                logger.error(
                    f"Failed to register transfer engine info for tp_rank={self.tp_rank}: "
                    f"{resp.status_code}, {resp.text}"
                )
        except Exception as e:
            logger.error(
                f"Failed to register transfer engine info for tp_rank={self.tp_rank}: {e}"
            )

    def model_specific_adjustment(self):
        server_args = self.server_args

        if self.is_multimodal:
            if not self.is_multimodal_chunked_prefill_supported:
                server_args.chunked_prefill_size = -1
                logger.info(
                    f"Automatically turn off --chunked-prefill-size as it is not supported for "
                    f"{self.model_config.hf_config.model_type}"
                )

        if (
            not self.use_mla_backend
            or server_args.attention_backend
            not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
        ):
            server_args.disable_chunked_prefix_cache = True

        if not server_args.disable_chunked_prefix_cache:
            log_info_on_rank0(logger, "Chunked prefix cache is turned on.")

    def check_quantized_moe_compatibility(self):
        if (
            quantization_config := getattr(
                self.model_config.hf_config, "quantization_config", None
            )
        ) is not None and (
            weight_block_size := quantization_config.get("weight_block_size", None)
        ) is not None:
            weight_block_size_n = weight_block_size[0]

            if self.tp_size % self.moe_ep_size != 0:
                raise ValueError(
                    f"tp_size {self.tp_size} must be divisible by ep_size {self.moe_ep_size}"
                )
            moe_tp_size = self.tp_size // self.moe_ep_size // self.moe_dp_size

            moe_intermediate_size = getattr(
                self.model_config.hf_text_config, "moe_intermediate_size", None
            )
            if moe_intermediate_size is None:
                return

            if moe_intermediate_size % moe_tp_size != 0:
                raise ValueError(
                    f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({self.tp_size}) divided by moe_ep_size ({self.moe_ep_size})."
                )

            if (
                not envs.SGLANG_SHARED_EXPERT_TP1.get()
                and (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0
                and not _use_aiter
            ):
                raise ValueError(
                    f"For quantized MoE models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                    f"where moe_tp_size is equal to tp_size ({self.tp_size}) divided by ep_size ({self.moe_ep_size}). "
                    f"You can fix this by setting arguments `--tp` and `--ep` correctly."
                )

    def init_torch_distributed(self):
        tic = time.perf_counter()
        logger.info("Init torch distributed begin.")

        try:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.tp_rank=} {self.tp_size=}"
            )
            raise

        backend = get_default_distributed_backend(self.device)
        if self.device == "cuda" and self.server_args.elastic_ep_backend == "mooncake":
            backend = "mooncake"
            if self.server_args.mooncake_ib_device:
                from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                    get_ib_devices_for_gpu,
                )

                ib_device_for_gpu = get_ib_devices_for_gpu(
                    self.server_args.mooncake_ib_device, self.gpu_id
                )
                mooncake_ib_device = (
                    ib_device_for_gpu.split(",") if ib_device_for_gpu else []
                )
                try:
                    from mooncake import ep as mooncake_ep

                    mooncake_ep.set_device_filter(mooncake_ib_device)
                except:
                    pass  # A warning will be raised in `init_distributed_environment`

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            monkey_patch_p2p_access_check()

        # Allow external orchestrators (e.g. trainpi) to override the distributed
        # init method.  When set to "env://", torch uses MASTER_ADDR/MASTER_PORT
        # env-vars and an externally-created TCPStore, completely avoiding port
        # conflicts with intra-host collocation.
        dist_init_method_override = envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
        if dist_init_method_override:
            dist_init_method = dist_init_method_override
        elif self.server_args.dist_init_addr:
            na = NetworkAddress.parse(self.server_args.dist_init_addr)
            dist_init_method = na.to_tcp()
        else:
            dist_init_method = NetworkAddress(
                self.server_args.host or "127.0.0.1", self.dist_port
            ).to_tcp()
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
        set_torch_symm_mem_all_reduce(self.server_args.enable_torch_symm_mem)

        if not self.is_draft_worker:
            if self.device == "cpu":
                if _is_cpu_amx_available or _is_cpu_arm64:
                    # Bind OpenMP threads to CPU cores
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

                    # Set local size to hint SGLang to use shared memory based AllReduce
                    os.environ["LOCAL_SIZE"] = str(self.tp_size)
                    torch.ops.sgl_kernel.initialize(self.tp_size, self.tp_rank)

                else:
                    logger.warning(
                        "init_cpu_threads_env and shared memory based AllReduce is disabled, only intel amx backend and arm64 are supported"
                    )

            # Only initialize the distributed environment on the target model worker.
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
                moe_a2a_backend=self.server_args.moe_a2a_backend,
                recovered_rank=self.server_args.elastic_ep_rejoin,
            )
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                attention_data_parallel_size=self.dp_size,
                pipeline_model_parallel_size=self.pp_size,
                expert_model_parallel_size=self.moe_ep_size,
                attention_context_model_parallel_size=self.attn_cp_size,
                moe_data_model_parallel_size=self.moe_dp_size,
                duplicate_tp_group=self.server_args.enable_pdmux,
                enable_symm_mem=self.server_args.enable_symm_mem,
                recovered_rank=self.server_args.elastic_ep_rejoin,
            )
            initialize_dp_attention(
                server_args=self.server_args,
                model_config=self.model_config,
            )
            if is_npu():
                register_sgl_tp_rank(self.gpu_id)

            # Pre-warm NCCL/RCCL to eliminate cold-start latency in first request
            # Controlled by --pre-warm-nccl flag (default: enabled on AMD GPUs)
            if self.server_args.pre_warm_nccl and (
                self.tp_size > 1 or self.pp_size > 1 or self.moe_ep_size > 1
            ):
                warmup_start = time.perf_counter()
                tp_group_handle = get_tp_group().device_group

                # Single warmup all_reduce to initialize NCCL/RCCL communicator
                warmup_tensor = torch.zeros(1, device=torch.cuda.current_device())
                dist.all_reduce(warmup_tensor, group=tp_group_handle)
                current_platform.synchronize()

                warmup_elapsed = time.perf_counter() - warmup_start
                logger.info(
                    f"NCCL/RCCL warmup completed in {warmup_elapsed:.3f}s "
                    f"(tp_size={self.tp_size}, pp_size={self.pp_size}, ep_size={self.moe_ep_size})"
                )

        pre_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        self.tp_group = get_tp_group()
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1 and not self.is_draft_worker:
            if pre_model_load_memory < local_gpu_memory * 0.9:
                msg = "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                msg += f"{pre_model_load_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                if envs.SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK.get():
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)

        logger.info(
            f"Init torch distributed ends. elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return pre_model_load_memory

    def init_shared_mooncake_transfer_engine(self):
        """
        Need MooncakeTransferEngine when:
        1) PD disaggregation uses mooncake for KV transfer (prefill/decode)
        2) HiCache uses mooncake storage backend
        3) Encoder disaggregation uses mooncake
        """
        use_mooncake_te = (
            (
                self.server_args.disaggregation_mode != "null"
                and self.server_args.disaggregation_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_hierarchical_cache
                and self.server_args.hicache_storage_backend == "mooncake"
                and envs.SGLANG_HICACHE_MOONCAKE_REUSE_TE.get()
            )
            or (
                self.server_args.encoder_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.language_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
        )

        if use_mooncake_te:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                init_mooncake_transfer_engine,
            )

            init_mooncake_transfer_engine(
                hostname=get_local_ip_auto(),
                gpu_id=self.gpu_id,
                ib_device=(
                    self.server_args.disaggregation_ib_device
                    or self.server_args.mooncake_ib_device
                ),
            )

    def load_model(self):
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        from sglang.srt.configs.modelopt_config import ModelOptConfig

        modelopt_config = ModelOptConfig(
            quant=self.server_args.modelopt_quant,
            checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
            checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
            export_path=self.server_args.modelopt_export_path,
            quantize_and_serve=self.server_args.quantize_and_serve,
        )

        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
            tp_rank=self.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_transfer_engine,
            remote_instance_weight_loader_transfer_engine_session_id=self.remote_instance_transfer_engine_session_id,
            modelexpress_url=self.server_args.modelexpress_url,
            modelexpress_transport=self.server_args.modelexpress_transport,
            modelopt_config=modelopt_config,
            rl_quant_profile=self.server_args.rl_quant_profile,
            draft_model_idx=self.draft_model_idx,
        )
        if self.device == "cpu":
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.tp_size
            )

        if (
            self.server_args.load_format == LoadFormat.REMOTE_INSTANCE
            and self.server_args.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            if self.tp_rank == 0:
                instance_ip = NetworkAddress.resolve_host(socket.gethostname())
                t = threading.Thread(
                    target=trigger_init_weights_send_group_for_remote_instance_request,
                    args=(
                        self.server_args.remote_instance_weight_loader_seed_instance_ip,
                        self.server_args.remote_instance_weight_loader_seed_instance_service_port,
                        self.server_args.remote_instance_weight_loader_send_weights_group_ports,
                        instance_ip,
                    ),
                )
                t.start()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        monkey_patch_vllm_parallel_state()

        enable_cpu_backup = self.server_args.enable_weights_cpu_backup or (
            self.is_draft_worker and self.server_args.enable_draft_weights_cpu_backup
        )
        with self.memory_saver_adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=enable_cpu_backup,
        ):
            self.loader = get_model_loader(
                load_config=self.load_config,
                model_config=self.model_config,
            )
            self.model = self.loader.load_model(
                model_config=self.model_config,
                device_config=DeviceConfig(self.device, self.gpu_id),
            )
            if hasattr(self.loader, "remote_instance_transfer_engine_weight_info"):
                self.remote_instance_transfer_engine_weight_info = (
                    self.loader.remote_instance_transfer_engine_weight_info
                )
        # Cache needs to be cleared after loading model weights (in the self.loader.load_model function).
        # To avoid conflict with memory_saver_adapter.region, empty_cache operation is now moved here.
        if _is_npu:
            torch.npu.empty_cache()
        monkey_patch_vllm_parallel_state(reverse=True)

        if not self.is_draft_worker:
            get_offloader().post_init()

        # Register model for layerwise NVTX profiling if enabled
        if self.server_args.enable_layerwise_nvtx_marker:
            pyt_hooks = PytHooks()
            pyt_hooks.register_hooks(self.model, module_prefix="model")

        if self.server_args.kv_cache_dtype == "fp8_e4m3":
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        self.server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        self.sliding_window_size = None
        if hasattr(self.model, "get_attention_sliding_window_size"):
            self.sliding_window_size = self.model.get_attention_sliding_window_size()
        elif (
            self.model_config.is_hybrid_swa
            and self.model_config.sliding_window_size is not None
        ):
            # sliding window field in model config may have different meaning for different kinds of models (e.g., dllm), here we only consider the sliding window in SWA model
            self.sliding_window_size = self.model_config.sliding_window_size
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size
            logger.info(
                f"Setting sliding_window_size to be attention_chunk_size: {self.sliding_window_size}"
            )

        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        # Get quantization config from ModelConfig
        # This handles both config.json (standard) and hf_quant_config.json (ModelOpt)
        quant_str = self.model_config.get_quantization_config_log_str()

        logger.info(
            f"Load weight end. "
            f"elapsed={time.perf_counter() - tic_total:.2f} s, "
            f"type={type(self.model).__name__}, "
            f"{quant_str + ', ' if quant_str else ''}"
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={self.weight_load_mem_usage:.2f} GB."
        )

        # TODO: Make sure all models have `quant_config` attribute, and all online quantization methods register which layers they actually quantize.
        # TODO: Move this online-quantization reporting out of ModelRunner.
        quantized_layers = getattr(
            getattr(self.model, "quant_config", None), "quantized_layers", None
        )
        if (
            self.server_args.quantization is not None
            and isinstance(quantized_layers, tuple)
            and len(quantized_layers) == 2
        ):
            layer_types, quantized_layers_count = quantized_layers
            logger.info(
                f"Online {self.server_args.quantization} quantization: quantized {quantized_layers_count} layers of types: {layer_types}"
            )

        if self.server_args.debug_tensor_dump_output_folder is not None:
            dump_folder = self.server_args.debug_tensor_dump_output_folder
            if self.spec_algorithm.is_eagle():
                role = "draft" if self.is_draft_worker else "target"
                dump_folder = os.path.join(dump_folder, role)
            register_forward_hook_for_model(
                self.model,
                dump_folder,
                self.server_args.debug_tensor_dump_layers,
                self.tp_size,
                self.tp_rank,
                self.pp_rank,
            )

        if dumper.may_enable:
            dumper.apply_source_patches()
            dumper.register_non_intrusive_dumper(self.model)

        # Pre-expand RoPE cache before CUDA Graph capture
        reserve_rope_cache_for_long_sequences(
            self.model,
            self.server_args,
            self.model_config,
            logger,
        )

        if self.server_args.elastic_ep_backend == "mooncake":
            # Mooncake does not support `monitored_barrier`
            dist.barrier(group=get_tp_group().cpu_group)
        else:
            # Handle the case where some ranks do not finish loading.
            try:
                dist.monitored_barrier(
                    group=get_tp_group().cpu_group,
                    timeout=datetime.timedelta(
                        seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S
                    ),
                    wait_all_ranks=True,
                )
            except RuntimeError:
                raise ValueError(
                    f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
                ) from None

    def _prepare_moe_topk(self):
        balancer_cls = None
        num_prepared = 0
        num_routed_experts = None
        for module in self.model.modules():
            if not isinstance(module, (TopK, HashTopK)):
                continue
            if (
                not module.enable_deepep_waterfill
                or module.deepep_waterfill_balancer is not None
            ):
                continue
            if num_routed_experts is None:
                num_routed_experts = getattr(
                    self.model_config.hf_config, "n_routed_experts", None
                )
                if num_routed_experts is None:
                    raise ValueError(
                        "DeepEP waterfill requires model config n_routed_experts."
                    )
            if balancer_cls is None:
                from sglang.srt.layers.moe.deepep_waterfill import (
                    DeepEPWaterfillBalancer,
                )

                balancer_cls = DeepEPWaterfillBalancer
            # Static EPLB remaps TopK ids to physical expert ids before Waterfill.
            # Redundant experts therefore need to be included in the per-rank
            # expert count used for Waterfill's shared-expert slot remapping.
            num_physical_routed_experts = (
                num_routed_experts + self.server_args.ep_num_redundant_experts
            )
            if isinstance(module, TopK):
                routed_scaling_factor = module.topk_config.routed_scaling_factor
            else:
                routed_scaling_factor = module.routed_scaling_factor
            module.deepep_waterfill_balancer = balancer_cls(
                num_routed_experts=num_physical_routed_experts,
                world_size=self.moe_ep_size,
                rank=self.moe_ep_rank,
                layer_id=module.layer_id,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor is not None else 1.0
                ),
            )
            num_prepared += 1
        if num_prepared:
            log_info_on_rank0(
                logger, f"Prepared {num_prepared} DeepEP waterfill TopK modules."
            )

    def update_expert_location(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        p2p_missing_logical_experts = self.expert_location_updater.update(
            self.model.routed_experts_weights_of_layer,
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=self.server_args.nnodes,
            rank=self.tp_rank,
        )

        if len(p2p_missing_logical_experts) > 0:
            # Load the missing expert weights from disk
            if callable(getattr(self.model, "generate_weight_name_filter", None)):
                # Filter and load only missing expert weights
                weight_name_filter = self.model.generate_weight_name_filter(
                    p2p_missing_logical_experts
                )
            else:
                # Do a full reload from disk/DRAM
                logger.info(
                    "[Elastic EP] Model does not implement generate_weight_name_filter. "
                    "Performing full weight reload."
                )
                weight_name_filter = None

            if (
                self.expert_backup_client is not None
                and self.expert_backup_client.use_backup
            ):
                # Load the missing weights from the DRAM backup
                self.expert_backup_client.update_weights(weight_name_filter)
            else:
                # Load the missing weights from disk
                self.update_weights_from_disk(
                    get_global_server_args().model_path,
                    get_global_server_args().load_format,
                    weight_name_filter=weight_name_filter,
                )

    def maybe_recover_ep_ranks(self):
        # TODO(perf): `active_ranks.all()` on a CUDA tensor triggers host-device
        # synchronization, and this function is on the forward-path.
        # This check only runs when `--elastic-ep-backend` is enabled, so the
        # synchronization overhead does not propagate to other configs.
        # Leave for future optimization of the elastic EP path.
        if self.tp_group.active_ranks.all() and self.tp_group.active_ranks_cpu.all():
            return

        tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
        tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
        tp_active_ranks &= tp_active_ranks_cpu
        # NOTE: `ranks_to_recover` uses indices in `tp_group`. For the current
        # Mooncake elastic EP implementation we assume `--pp-size=1`, so the
        # tp-group index is the same as the global rank index.
        ranks_to_recover = [
            i for i in range(len(tp_active_ranks)) if not tp_active_ranks[i]
        ]

        # try_recover_ranks polls peer state via Mooncake EP backend.
        # Mooncake's internal semantics guarantee that all ranks observe
        # consistent peer readiness state, so collective operations below
        # are safe even though polling appears local.
        if ranks_to_recover and try_recover_ranks(ranks_to_recover):
            self.forward_pass_id = 0
            self.eplb_manager.reset_generator()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=False
                )
            )
            ElasticEPStateManager.instance().reset()

            broadcast_pyobj(
                [self.server_args.random_seed],
                get_world_group().rank,
                get_world_group().cpu_group,
                src=get_world_group().ranks[0],
            )
            logger.info(f"recover ranks {ranks_to_recover} done")

    def _get_healthy_expert_location_src_rank(
        self, invoked_in_elastic_ep_rejoin_path: bool
    ) -> int:
        world_group = get_world_group()
        # NOTE: do not key off `self.server_args.elastic_ep_rejoin` here.
        # A rank that was started as a rejoin rank may later act as a healthy
        # rank in a subsequent recovery cycle.
        local_rejoin_flag = bool(invoked_in_elastic_ep_rejoin_path)
        gathered_rejoin_flags = world_group.all_gather_object(local_rejoin_flag)

        for rank_in_group, is_rejoin_rank in enumerate(gathered_rejoin_flags):
            if not is_rejoin_rank:
                return world_group.ranks[rank_in_group]

        raise RuntimeError(
            "No healthy rank found for broadcasting expert location metadata. "
            "All ranks are marked as elastic_ep_rejoin."
        )

    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        def model_load_weights(model, iter):
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config

        if recapture_cuda_graph and (
            self.device == "cuda"
            or self.device == "musa"
            or (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            )
        ):
            self.init_device_graphs()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    def init_weights_send_group_for_remote_instance(
        self,
        master_address,
        ports,
        group_rank,
        world_size,
        group_name,
        backend="nccl",
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        logger.info(
            f"init custom process group: tp_rank={self.tp_rank}, gpu_id={self.gpu_id}, master_address={master_address}, master_port={group_port}, "
            f"group_rank={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        current_platform.empty_cache()
        success = False
        message = ""
        try:
            na = NetworkAddress(master_address, group_port)
            self._weights_send_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=torch.device("cuda", self.gpu_id),
            )
            dist.barrier(group=self._weights_send_group[group_name])
            success = True
            message = f"Succeeded to init group through {na.to_host_port_str()} group."
        except Exception as e:
            message = f"Failed to init group: {e}."
            logger.error(message)

        current_platform.empty_cache()
        return success, message

    def send_weights_to_remote_instance(
        self,
        master_address,
        ports,
        group_name,
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        if self._weights_send_group[group_name] is not None:
            send_group = self._weights_send_group[group_name]
        else:
            message = f"Group {group_name} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first."
            logger.error(message)
            return False, message

        current_platform.empty_cache()
        success = False
        na = NetworkAddress(master_address, group_port)
        message = ""
        try:
            for _, weights in self.model.named_parameters():
                torch.distributed.broadcast(
                    weights,
                    src=0,
                    group=send_group,
                )
            success = True
            message = f"Succeeded to send weights through {na.to_host_port_str()} {group_name}."
        except Exception as e:
            message = f"Failed to send weights: {e}."
            logger.error(message)

        # destroy the process group after sending weights
        del self._weights_send_group[group_name]
        torch.distributed.distributed_c10d.destroy_process_group(send_group)
        current_platform.empty_cache()
        return success, message

    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            na = NetworkAddress(master_address, master_port)
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    def destroy_weights_update_group(self, group_name):
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def _update_bucketed_weights_from_distributed(
        self, names, dtypes, shapes, group_name
    ):
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (name, torch.empty(shape, dtype=target_dtype, device=self.device))
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.model.load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
        load_format: Optional[str] = None,
    ):
        monkey_patch_torch_reductions()
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        device_module = torch.get_device_module(self.device)
        infered_device = device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format in self.server_args.custom_weight_loader:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    def _update_weights_from_flattened_bucket(
        self,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            server_args=self.server_args,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
        )

    def _init_lora_cuda_graph_moe_buffers(self):
        """Phase 1 of LoRA CUDA graph init: pre-allocate MoE intermediate buffers.

        Must be called before init_memory_pool() so that memory profiling
        sees the reduced available memory and sizes KV cache correctly.
        All MoE LoRA layers share one set of buffers (managed by the
        lora_backend) since they execute sequentially during forward.

        Phase 2 (dense LoRA batch metadata) is handled later in
        CudaGraphRunner.__init__() via lora_manager.init_cuda_graph_batch_info(),
        because it needs capture-time parameters (max_bs, num_tokens_per_bs)
        that are only available at that stage.
        """
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        max_bs = self.server_args.cuda_graph_max_bs
        max_loras = self.server_args.max_loras_per_batch
        for module in self.model.modules():
            if isinstance(module, FusedMoEWithLoRA):
                self.lora_manager.init_cuda_graph_moe_buffers(
                    max_bs, max_loras, self.dtype, module
                )
                logger.info(
                    f"Pre-allocated shared MoE LoRA CUDA graph buffers "
                    f"(max_bs={max_bs}, max_loras={max_loras})"
                )
                break

    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface."""

        logger.info(
            f"LoRA adapter loading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.load_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter loading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    def load_lora_adapter_from_tensors(
        self, lora_ref: LoRARef, tensors, config_dict, added_tokens_config=None
    ):
        logger.info(f"LoRA adapter loading from tensors starts: {lora_ref}.")
        result = self.lora_manager.load_lora_adapter_from_tensors(
            lora_ref, tensors, config_dict, added_tokens_config
        )
        logger.info(f"LoRA adapter loading from tensors completes: {lora_ref}.")
        return result

    def unload_lora_adapter(self, lora_ref: LoRARef):
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading."""

        logger.info(
            f"LoRA adapter unloading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.unload_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter unloading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    @property
    def qwen3_next_config(self):
        config = self.model_config.hf_config
        if isinstance(config, Qwen3NextConfig):
            return config
        return None

    @property
    def hybrid_lightning_config(self):
        config = self.model_config.hf_config
        if isinstance(config, BailingHybridConfig):
            return config
        return None

    @property
    def hybrid_gdn_config(self):
        config = self.model_config.hf_config.get_text_config()
        if isinstance(
            config,
            Qwen3NextConfig
            | Qwen3_5Config
            | Qwen3_5MoeConfig
            | InternS2PreviewConfig
            | JetNemotronConfig
            | JetVLMConfig,
        ):
            return config
        return None

    @property
    def mamba2_config(self):
        config = self.model_config.hf_config
        if isinstance(config, NemotronHConfig) and self.is_draft_worker:
            # NemotronH MTP draft models have no Mamba layers (pattern like "*E")
            # so they shouldn't use HybridLinearAttnBackend
            pattern = getattr(config, "mtp_hybrid_override_pattern", None)
            if pattern is not None and "M" not in pattern:
                return None
        if isinstance(
            config,
            FalconH1Config
            | NemotronHConfig
            | Lfm2Config
            | Lfm2MoeConfig
            | Lfm2VlConfig,
        ):
            return config
        if isinstance(config, NemotronH_Nano_VL_V2_Config):
            return config.llm_config

        if isinstance(config, GraniteMoeHybridConfig):
            has_mamba = any(
                layer_type == "mamba"
                for layer_type in getattr(config, "layer_types", [])
            )
            if not has_mamba:
                return None
            else:
                return config

        return None

    @property
    def max_token_pool_size(self):
        """Return the max token pool size considering hybrid swa settings."""
        if self.is_hybrid_swa:
            return self.full_max_total_num_tokens
        else:
            return self.max_total_num_tokens

    @property
    def kimi_linear_config(self):
        config = self.model_config.hf_config
        if isinstance(config, KimiLinearConfig):
            return config
        return None

    def _get_linear_attn_registry_result(self):
        if self._linear_attn_registry_cache is _UNSET:
            self._linear_attn_registry_cache = get_linear_attn_config(
                self.model_config.hf_config
            )
        return self._linear_attn_registry_cache

    @property
    def linear_attn_model_spec(self):
        result = self._get_linear_attn_registry_result()
        return result[0] if result else None

    @property
    def mambaish_config(self):
        existing = (
            self.mamba2_config
            or self.hybrid_gdn_config
            or self.kimi_linear_config
            or self.hybrid_lightning_config
        )
        if existing:
            return existing
        result = self._get_linear_attn_registry_result()
        return result[1] if result else None

    def configure_kv_cache_dtype(self):
        if self.server_args.kv_cache_dtype == "auto":
            quant_config = getattr(self.model, "quant_config", None)
            kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
            if (
                isinstance(kv_cache_quant_algo, str)
                and kv_cache_quant_algo.upper() == "FP8"
            ):
                if _is_hip:
                    self.kv_cache_dtype = fp8_dtype
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
                else:
                    self.kv_cache_dtype = torch.float8_e4m3fn
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
            else:
                self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e4m3fn
        elif self.server_args.kv_cache_dtype in ("bf16", "bfloat16"):
            self.kv_cache_dtype = torch.bfloat16
        elif self.server_args.kv_cache_dtype == "fp4_e2m1":
            if hasattr(torch, "float4_e2m1fn_x2"):
                self.kv_cache_dtype = torch.float4_e2m1fn_x2
                logger.warning(f"FP4 (E2M1) KV Cache might lead to a accuracy drop!")
            else:
                logger.warning(
                    f"--kv-cache-dtype falls back to 'auto' because this torch version does not support torch.float4_e2m1fn_x2"
                )
                self.kv_cache_dtype = self.dtype
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    def init_attention_backend(self):
        """Init attention kernel backend."""
        if self.server_args.enable_pdmux:
            self.attn_backend = self._get_attention_backend(init_new_workspace=True)
            self.decode_attn_backend_group = []
            for _ in range(self.server_args.sm_group_num):
                self.decode_attn_backend_group.append(self._get_attention_backend())
            self.decode_attn_backend = self.decode_attn_backend_group[0]
        elif self.server_args.enable_two_batch_overlap and not self.is_draft_worker:
            self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
        else:
            self.attn_backend = self._get_attention_backend()

    def _get_attention_backend(self, init_new_workspace: bool = False):
        """Init attention kernel backend."""
        draft_attn_backend = self.server_args.speculative_draft_attention_backend
        if self.is_draft_worker and draft_attn_backend:
            logger.warning(
                f"Overriding draft attention backend to {draft_attn_backend}."
            )
            return self._get_attention_backend_from_str(
                draft_attn_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            self.prefill_attention_backend_str,
            self.decode_attention_backend_str,
        ) = self.server_args.get_attention_backends()

        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                self,
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            get_global_server_args().prefill_attention_backend,
            get_global_server_args().decode_attention_backend,
        ) = (self.prefill_attention_backend_str, self.decode_attention_backend_str)
        return attn_backend

    def _get_attention_backend_from_str(
        self, backend_str: str, init_new_workspace: bool = False
    ):
        if backend_str not in ATTENTION_BACKENDS:
            raise ValueError(f"Invalid attention backend: {backend_str}")
        self.init_new_workspace = init_new_workspace
        full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
        return attn_backend_wrapper(self, full_attention_backend)

    def kernel_warmup(self):
        """Warmup and tune kernels before cuda graph capture."""
        if self.device != "cuda":
            return

        if self._should_run_flashinfer_autotune():
            self._flashinfer_autotune()

        if (
            envs.SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP.get()
            and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and self.pp_size > 1
            and not self.spec_algorithm.is_speculative()
        ):
            from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
                pp_parallel_deep_gemm_warmup,
            )

            pp_parallel_deep_gemm_warmup(self)

    def _pre_initialize_flashinfer_allreduce_workspace(self):
        """Pre-initialize flashinfer allreduce fusion workspaces.

        Must run before CUDA graph capture to avoid collective operations
        (broadcasts, barriers) inside the graph capture context, which can
        deadlock with custom_all_reduce.register_graph_buffers.
        """
        if not self.server_args.enable_flashinfer_allreduce_fusion:
            return

        from sglang.srt.layers.communicator import FUSE_ALLREDUCE_MAX_BATCH_SIZE
        from sglang.srt.layers.flashinfer_comm_fusion import (
            pre_initialize_workspaces,
        )

        pre_initialize_workspaces(
            max_token_num=FUSE_ALLREDUCE_MAX_BATCH_SIZE,
            hidden_dim=self.model_config.hidden_size,
            dtype=self.dtype,
        )

    def _should_run_flashinfer_autotune(self) -> bool:
        """Check if flashinfer autotune should be run."""
        if self.server_args.disable_flashinfer_autotune:
            return False

        # CuteDSL v1 (cutedsl runner + deepep a2a) bypasses MoeRunner and must not
        # be autotuned -- its _dummy_run would dispatch more tokens per rank than
        # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, tripping a DeepEP assert.
        # Read server_args directly to avoid depending on initialize_moe_config()
        # having already populated the MoE backend globals.
        if (
            self.server_args.moe_runner_backend == "flashinfer_cutedsl"
            and self.server_args.moe_a2a_backend == "deepep"
        ):
            return False

        backend_str = self.server_args.moe_runner_backend

        # TODO smor- support other cases for flashinfer autotune, such as, mamba backend

        moe_needs_autotune = backend_str in [
            "flashinfer_trtllm",
            "flashinfer_trtllm_routed",
            "flashinfer_mxfp4",
            "flashinfer_cutedsl",
            "flashinfer_cutlass",
        ]

        from sglang.srt.layers.quantization.fp4_utils import (
            get_fp4_gemm_runner_backend,
        )

        model_uses_fp4 = self.model_config.quantization in (
            "modelopt_fp4",
            "modelopt_mixed",
        )
        fp4_gemm_needs_autotune = model_uses_fp4 and (
            get_fp4_gemm_runner_backend().is_flashinfer_cutlass()
            or get_fp4_gemm_runner_backend().is_flashinfer_cutedsl()
        )

        if not (moe_needs_autotune or fp4_gemm_needs_autotune):
            return False

        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            return False

        if self.spec_algorithm.is_speculative():
            return not self.is_draft_worker

        return True

    def _flashinfer_autotune(self):
        """Run flashinfer autotune."""
        from flashinfer.autotuner import autotune

        from sglang.srt.layers.logits_processor import autotune_dummy_run_mode

        cache_path = self._flashinfer_autotune_cache_path()
        if envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.get():
            autotune_cache = cache_path
            logger.info("Running FlashInfer autotune with cache: %s", autotune_cache)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            runs_dir = cache_path.parent / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            autotune_cache = (
                runs_dir / f"{cache_path.stem}.{timestamp}{cache_path.suffix}"
            )
            logger.info(
                "Running FlashInfer autotune (cache reuse DISABLED via "
                "SGLANG_FLASHINFER_AUTOTUNE_CACHE=0); writing fresh result to: %s",
                autotune_cache,
            )

        # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
        # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
        self.forward_stream.wait_stream(torch.cuda.current_stream())
        with torch.get_device_module(self.device).stream(self.forward_stream):
            with (
                torch.inference_mode(),
                autotune(True, cache=str(autotune_cache)),
                autotune_dummy_run_mode(),
            ):
                self._dummy_run(batch_size=self.req_to_token_pool.size)
        torch.cuda.current_stream().wait_stream(self.forward_stream)
        logger.info("FlashInfer autotune completed.")

    def _flashinfer_autotune_cache_path(self) -> Path:
        import flashinfer

        major, minor = torch.cuda.get_device_capability(self.device)
        arch = f"sm{major}{minor}"
        flashinfer_version = getattr(flashinfer, "__version__", "unknown")

        server_args = self.server_args
        model_key = "|".join(
            [
                str(server_args.model_path),
                str(self.dtype),
                str(server_args.quantization),
                str(server_args.moe_runner_backend),
                str(self.tp_size),
                str(self.pp_size),
                str(self.dp_size),
                str(self.moe_ep_size),
                str(self.model_config.hf_config.__class__.__name__),
            ]
        )
        cache_key = hashlib.sha256(model_key.encode()).hexdigest()[:16]
        cache_dir = (
            Path(envs.SGLANG_CACHE_DIR.get())
            / "flashinfer"
            / "autotune"
            / flashinfer_version
            / arch
            / cache_key
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return (
            cache_dir
            / f"rank_tp{self.tp_rank}_pp{self.pp_rank}_dp{self.dp_rank or 0}.json"
        )

    def _dummy_run(
        self,
        batch_size: int,
        run_ctx=None,
        forward_mode_override: Optional[ForwardMode] = None,
    ):
        """Run a dummy forward pass for warmup/profiling.

        ``forward_mode_override`` forces EXTEND/DECODE regardless of
        ``is_generation`` (used by the PP-parallel DeepGEMM warmup).
        """
        if forward_mode_override is not None:
            capture_forward_mode = forward_mode_override
        elif self.is_generation:
            capture_forward_mode = ForwardMode.DECODE
        else:
            capture_forward_mode = ForwardMode.EXTEND
        capture_hidden_mode = CaptureHiddenMode.NULL
        num_tokens_per_bs = 1
        if self.spec_algorithm.is_speculative():
            if self.is_draft_worker:
                if not self.spec_algorithm.supports_target_verify_for_draft():
                    raise RuntimeError("This should not happen")
            capture_forward_mode = ForwardMode.TARGET_VERIFY
            num_tokens_per_bs = (
                self.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                    self.server_args.speculative_num_draft_tokens, self.is_draft_worker
                )
            )

        if self.server_args.enable_return_hidden_states:
            capture_hidden_mode = CaptureHiddenMode.FULL

        num_tokens = batch_size * num_tokens_per_bs

        # Keep warmup aligned with scheduler MLP-sync padding.
        if require_mlp_sync(self.server_args):
            attn_tp_size = get_attention_tp_size()
            if attn_tp_size > 1 and num_tokens % attn_tp_size != 0:
                num_tokens = ceil_align(num_tokens, attn_tp_size)
                batch_size = num_tokens // num_tokens_per_bs

        seq_len_fill_value = self.attn_backend.get_cuda_graph_seq_len_fill_value()

        if self.server_args.enable_torch_compile:
            set_torch_compile_config()
            should_disable_torch_compile = not getattr(
                self.model, "_can_torch_compile", True
            )
            if should_disable_torch_compile:
                log_info_on_rank0(
                    logger,
                    "Transformers backend model reports it is not torch.compile "
                    "compatible (e.g. dynamic rope scaling). Disabling torch.compile.",
                )
                self.server_args.enable_torch_compile = False

        # NOTE: aux hidden state capture (eagle3/dflash) is already
        # configured by init_aux_hidden_state_capture() in initialize().

        require_mlp_tp_gather_ = require_mlp_tp_gather(self.server_args)
        if require_gathered_buffer(self.server_args):
            assert require_mlp_tp_gather_ or require_attn_tp_gather(self.server_args)

        buffers = _allocate_decode_buffers(
            device=self.device,
            max_bs=batch_size,
            max_num_token=num_tokens,
            hidden_size=self.model_config.hidden_size,
            vocab_size=self.model_config.vocab_size,
            dtype=self.model_config.dtype,
            dp_size=self.server_args.dp_size,
            pp_size=self.server_args.pp_size,
            is_encoder_decoder=self.model_config.is_encoder_decoder,
            require_mlp_tp_gather=require_mlp_tp_gather_,
            seq_len_fill_value=seq_len_fill_value,
            encoder_len_fill_value=(
                getattr(self.model_config.hf_config, "max_source_positions", 0)
                if self.model_config.is_encoder_decoder
                else 0
            ),
            num_tokens_per_bs=num_tokens_per_bs,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
            hc_hidden_size=getattr(self.model_config, "hc_hidden_size", None),
        )
        buffers.num_token_non_padded[...] = num_tokens

        # For extend mode
        if capture_forward_mode == ForwardMode.EXTEND:
            extend_prefix_lens_cpu = [0] * batch_size
            extend_seq_lens_cpu = [seq_len_fill_value] * batch_size
            extend_num_tokens = num_tokens
            extend_seq_lens = torch.full(
                (batch_size,), seq_len_fill_value, dtype=torch.int32, device=self.device
            )
            extend_prefix_lens = torch.zeros(
                (batch_size,), dtype=torch.int32, device=self.device
            )
            extend_start_loc = torch.arange(
                0, num_tokens, num_tokens_per_bs, dtype=torch.int32, device=self.device
            )
        else:
            extend_prefix_lens_cpu = None
            extend_seq_lens_cpu = None
            extend_num_tokens = None
            extend_seq_lens = None
            extend_prefix_lens = None
            extend_start_loc = None

        if self.server_args.pp_size > 1:
            # PP0 already cp-split hidden_states before send.
            pp_hidden_tokens = num_tokens
            if (
                capture_forward_mode == ForwardMode.EXTEND
                and self.pp_rank != 0
                and self.attn_cp_size > 1
            ):
                pp_hidden_tokens = num_tokens // self.attn_cp_size
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:pp_hidden_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if require_mlp_tp_gather_:
            global_num_tokens_cpu = [num_tokens] * self.server_args.dp_size
        elif require_attn_tp_gather(self.server_args):
            global_num_tokens_cpu = [num_tokens]
        else:
            global_num_tokens_cpu = None

        if global_num_tokens_cpu is not None:
            global_dp_buffer_len = sum(global_num_tokens_cpu)
            num_tokens_tensor = torch.tensor(
                global_num_tokens_cpu, dtype=torch.int32, device=self.device
            )
            buffers.global_num_tokens_gpu.copy_(num_tokens_tensor)
            buffers.global_num_tokens_for_logprob_gpu.copy_(num_tokens_tensor)
        else:
            global_dp_buffer_len = None
            global_num_tokens_cpu = None

        def get_spec_info():
            spec_info = None
            if self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone():
                from sglang.srt.speculative.eagle_info import EagleVerifyInput

                if self.is_draft_worker:
                    raise RuntimeError("This should not happen.")
                else:
                    spec_info = EagleVerifyInput(
                        draft_token=None,
                        custom_mask=buffers.custom_mask,
                        positions=None,
                        retrieve_index=None,
                        retrieve_next_token=None,
                        retrieve_next_sibling=None,
                        retrieve_cum_len=None,
                        spec_steps=self.server_args.speculative_num_steps,
                        topk=self.server_args.speculative_eagle_topk,
                        draft_token_num=self.server_args.speculative_num_draft_tokens,
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                        seq_lens_sum=None,
                        seq_lens_cpu=None,
                    )
            elif self.spec_algorithm.is_dflash():
                from sglang.srt.speculative.dflash_info import DFlashVerifyInput

                # Dummy warmup only needs shape metadata; avoid forcing custom-mask mode.
                spec_info = DFlashVerifyInput(
                    draft_token=None,
                    positions=None,
                    draft_token_num=self.server_args.speculative_num_draft_tokens,
                    custom_mask=None,
                    capture_hidden_mode=(
                        CaptureHiddenMode.NULL
                        if self.is_draft_worker
                        else CaptureHiddenMode.FULL
                    ),
                )

            elif self.spec_algorithm.is_ngram():
                from sglang.srt.speculative.ngram_info import NgramVerifyInput

                spec_info = NgramVerifyInput(
                    draft_token=None,
                    tree_mask=buffers.custom_mask,
                    positions=None,
                    retrieve_index=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    draft_token_num=num_tokens_per_bs,
                )
                spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

            return spec_info

        spec_info = get_spec_info()
        if capture_hidden_mode != CaptureHiddenMode.FULL:
            capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.server_args.enable_lora:
            lora_ids = [None] * batch_size
        else:
            lora_ids = None

        forward_batch = ForwardBatch(
            forward_mode=capture_forward_mode,
            batch_size=batch_size,
            input_ids=buffers.input_ids,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_cpu=buffers.seq_lens_cpu,
            next_token_logits_buffer=buffers.next_token_logits_buffer,
            orig_seq_lens=buffers.seq_lens,
            out_cache_loc=buffers.out_cache_loc,
            seq_lens_sum=buffers.seq_lens.sum().item(),
            encoder_lens=buffers.encoder_lens,
            return_logprob=False,
            positions=buffers.positions,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_cpu=global_num_tokens_cpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=buffers.mrope_positions,
            spec_algorithm=self.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=capture_forward_mode,
            lora_ids=lora_ids,
        )

        if lora_ids is not None:
            self.lora_manager.prepare_lora_batch(forward_batch)

        self.attn_backend.init_forward_metadata(forward_batch)

        def run_once():
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
                global_num_tokens_cpu,
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.server_args.pp_size > 1
                and "pp_proxy_tensors"
                in inspect.signature(self.model.forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if not self.is_generation:
                kwargs["get_embedding"] = True

            logits_output_or_pp_proxy_tensors = self.model.forward(
                buffers.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            return logits_output_or_pp_proxy_tensors

        torch.get_device_module(self.device).synchronize()
        self.tp_group.barrier()
        with forward_context(ForwardContext(attn_backend=self.attn_backend)):
            with torch.inference_mode(), run_ctx or empty_context():
                run_once()

    def maybe_init_ngram_embedding(self):
        self.use_ngram_embedding = self.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            from sglang.srt.layers.n_gram_embedding import NgramEmbedding

            # Sized to mirror req_to_token (indexed by req_pool_idx).
            self.token_table = torch.empty(
                self.req_to_token_pool.req_to_token.shape[0],
                self.model_config.context_len,
                dtype=torch.int32,
                device=self.device,
            )
            chunked_prefill_size = self.server_args.chunked_prefill_size
            assert (
                chunked_prefill_size is not None and chunked_prefill_size > 0
            ), "Ngram embedding requires chunked prefill to be enabled (chunked_prefill_size > 0)"
            for module in self.model.modules():
                if isinstance(module, NgramEmbedding):
                    module.init_buffers(
                        self.max_running_requests, chunked_prefill_size, self.device
                    )

    def maybe_update_ngram_token_table(
        self,
        next_token_ids: torch.Tensor,
        forward_batch: "ForwardBatch",
    ):
        """Update the ngram embedding token table after sampling."""
        ngram_embedding_info = forward_batch.ngram_embedding_info
        if ngram_embedding_info is None:
            return
        ngram_embedding_info.out_column_starts[: forward_batch.batch_size] = (
            forward_batch.seq_lens
        )
        ngram_embedding_info.out_req_lens[: forward_batch.batch_size] = 1
        update_token_table_decode(
            ne_token_table=ngram_embedding_info.token_table,
            tokens=next_token_ids.to(torch.int32),
            row_indices=forward_batch.req_pool_indices,
            column_starts=ngram_embedding_info.out_column_starts,
        )

    def init_device_graphs(self):
        """Capture device graphs."""
        self.graph_runner = None
        self.graph_mem_usage = 0

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
            return

        if self.device != "cpu" and self.server_args.disable_cuda_graph:
            return

        if self.device == "cpu" and not self.server_args.enable_torch_compile:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        graph_backend = defaultdict(
            lambda: f"{current_platform.device_name} graph",
            {
                "cuda": "cuda graph",
                "musa": "cuda graph",
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )
        logger.info(
            f"Capture {graph_backend[self.device]} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        if current_platform.is_out_of_tree():
            GraphRunnerCls = current_platform.get_graph_runner_cls()
            self.graph_runner = GraphRunnerCls(self)
        else:
            graph_runners = defaultdict(
                lambda: CudaGraphRunner,
                {
                    "cpu": CPUGraphRunner,
                    "npu": NPUGraphRunner,
                },
            )
            self.graph_runner = graph_runners[self.device](self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {graph_backend[self.device]} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_piecewise_cuda_graphs(self, force_for_draft_worker: bool = False):
        """Initialize piecewise CUDA graph runner."""
        self.piecewise_cuda_graph_runner = None

        if self.server_args.disable_piecewise_cuda_graph:
            logger.info(
                "Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set"
            )
            return

        # Draft models skip here during __init__; the eagle worker calls
        # this method explicitly (force_for_draft_worker=True) after
        # init_lm_head so graphs capture the final embedding weights.
        if self.is_draft_worker and not force_for_draft_worker:
            return

        # Disable piecewise CUDA graph for non-language models
        if not hasattr(self.model, "model"):
            logger.warning(
                "Disable piecewise CUDA graph because the model is not a language model"
            )
            return

        # Disable piecewise CUDA graph for non capture size
        if not self.server_args.piecewise_cuda_graph_tokens:
            logger.warning(
                "Disable piecewise CUDA graph because the capture size is not set"
            )
            return

        # Collect attention layers and moe layers from the model
        self.model.model = resolve_language_model(self.model)
        language_model = getattr(self.model, "language_model", self.model)

        # Resolve model with layers: handle CausalLM wrapper (.model.layers) and direct TextModel (.layers)
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            layer_model = language_model.model
        elif hasattr(language_model, "layers"):
            layer_model = language_model
        else:
            logger.warning(
                "Disable piecewise CUDA graph because the model does not have a 'layers' attribute"
            )
            return

        self.attention_layers = []
        self.moe_layers = []
        self.moe_fusions = []
        self.dsa_indexers = []
        for layer in layer_model.layers:
            attn_layer = None
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    attn_layer = layer.self_attn.attn
                elif hasattr(layer.self_attn, "attn_mqa"):
                    # For DeepSeek model
                    attn_layer = layer.self_attn.attn_mqa
                    if _is_hip and hasattr(layer.self_attn, "attn_mha"):
                        attn_layer._pcg_mha_companion = layer.self_attn.attn_mha
            # For hybrid model
            elif hasattr(layer, "attn"):
                attn_layer = layer.attn
            elif hasattr(layer, "linear_attn"):
                if hasattr(layer.linear_attn, "attn"):
                    attn_layer = layer.linear_attn.attn
                else:
                    attn_layer = layer.linear_attn
            # For InternVL model
            elif hasattr(layer, "attention"):
                if hasattr(layer.attention, "attn"):
                    attn_layer = layer.attention.attn
            # For NemotronH and similar hybrid models using 'mixer' attribute
            elif hasattr(layer, "mixer"):
                if hasattr(layer.mixer, "attn"):
                    attn_layer = layer.mixer.attn
                elif hasattr(layer, "_forward_mamba"):
                    # Mamba layer with split op support - store the layer itself
                    attn_layer = layer

            if attn_layer is not None:
                self.attention_layers.append(attn_layer)
            elif hasattr(layer, "mixer"):
                self.attention_layers.append(None)

            moe_block = None
            moe_fusion = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                moe_block = layer.mlp.experts
                moe_fusion = layer.mlp
            if hasattr(layer, "block_sparse_moe") and hasattr(
                layer.block_sparse_moe, "experts"
            ):
                moe_block = layer.block_sparse_moe.experts
                moe_fusion = layer.block_sparse_moe
            if hasattr(layer, "moe") and hasattr(layer.moe, "experts"):
                moe_block = layer.moe.experts
                moe_fusion = layer.moe
            # For NemotronH MoE layers using 'mixer' attribute
            if hasattr(layer, "mixer") and hasattr(layer.mixer, "experts"):
                moe_block = layer.mixer.experts
                moe_fusion = layer.mixer
            self.moe_layers.append(moe_block)
            self.moe_fusions.append(moe_fusion)
            # NSA indexers (None for layers without NSA)
            dsa_indexer = None
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "indexer"):
                dsa_indexer = layer.self_attn.indexer
            self.dsa_indexers.append(dsa_indexer)

        if len(self.attention_layers) < self.model_config.num_hidden_layers:
            # TODO(yuwei): support Non-Standard GQA
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because some layers do not apply Standard GQA",
            )
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture piecewise CUDA graph begin. avail mem={before_mem:.2f} GB"
        )

        if self.server_args.enable_breakable_cuda_graph:
            # Experimental feature
            self.piecewise_cuda_graph_runner = BreakableCudaGraphRunner(self)
        else:
            self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        mem_usage = before_mem - after_mem
        logger.info(
            f"Capture piecewise CUDA graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    def init_threads_binding(self):
        omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        if omp_cpuids == "all":
            assert self.tp_size <= n_numa_node, (
                f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
                f"tp_size {self.tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
                f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
                f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
                f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
                f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
                f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "
                f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
            )
            if self.tp_size < n_numa_node:
                logger.warning(
                    f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {self.tp_size}, so only {self.tp_size} numa nodes are used."
                )
            self.local_omp_cpuid = cpu_ids_by_node[self.tp_rank]
        else:
            threads_bind_list = omp_cpuids.split("|")
            assert self.tp_size == len(threads_bind_list), (
                f"SGLANG_CPU_OMP_THREADS_BIND setting must be aligned with TP size parameter ({self.tp_size}). "
                f"Please double check your settings."
            )
            self.local_omp_cpuid = threads_bind_list[self.tp_rank]
            if self.tp_size > n_numa_node:
                logger.warning(
                    f"TP size ({self.tp_size})is larger than numa node number ({n_numa_node}), "
                    f"in this case the available memory amount of each rank cannot be determined in prior. "
                    f"Please set proper `--max-total-tokens` to avoid the out-of-memory error."
                )

    def apply_torch_tp(self):
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.layers.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    def update_decode_attn_backend(self, stream_idx: int):
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

    def _ensure_eager_registry(
        self,
        cache: _EagerBufferRegistry,
        raw_bs: int,
        raw_num_tokens: int,
        build: Callable[[int, int], "CudaGraphBufferRegistry"],
    ) -> "CudaGraphBufferRegistry":
        # Built on first use and grown (next power of two) when a batch exceeds
        # the current capacity.
        if (
            cache.registry is not None
            and raw_bs <= cache.max_bs
            and raw_num_tokens <= cache.max_num_tokens
        ):
            return cache.registry
        cache.max_bs = next_power_of_2(max(raw_bs, cache.max_bs))
        cache.max_num_tokens = next_power_of_2(
            max(raw_num_tokens, cache.max_num_tokens)
        )
        cache.registry = build(cache.max_bs, cache.max_num_tokens)
        return cache.registry

    def _ensure_eager_decode_registry(
        self, raw_bs: int, raw_num_tokens: int
    ) -> "CudaGraphBufferRegistry":
        is_encoder_decoder = self.model_config.is_encoder_decoder
        return self._ensure_eager_registry(
            self._eager_decode_registry,
            raw_bs,
            raw_num_tokens,
            lambda bs, num_tokens: build_decode_registry(
                device=self.device,
                max_bs=bs,
                max_num_token=num_tokens,
                # Eager has no padding so this sentinel is never read; 0 avoids the
                # cuda-graph-only fill-value method that some backends lack.
                seq_len_fill_value=0,
                cache_loc_dtype=torch.int64,
                enable_mamba_track=(
                    self.server_args.enable_mamba_extra_buffer()
                    and self.spec_algorithm.is_none()
                ),
                is_encoder_decoder=is_encoder_decoder,
                encoder_len_fill_value=(
                    getattr(self.model_config.hf_config, "max_source_positions", 0)
                    if is_encoder_decoder
                    else 0
                ),
                enable_num_token_non_padded=False,
                register_global_num_tokens=False,
                require_gathered_buffer=False,
                require_mlp_tp_gather=False,
                dp_size=self.server_args.dp_size,
                share_pool=False,
                source=None,
            ),
        )

    def _ensure_eager_prefill_registry(
        self, raw_bs: int, raw_num_tokens: int
    ) -> "CudaGraphBufferRegistry":
        return self._ensure_eager_registry(
            self._eager_prefill_registry,
            raw_bs,
            raw_num_tokens,
            lambda bs, num_tokens: build_prefill_registry(
                device=self.device,
                max_bs=bs,
                max_num_token=num_tokens,
                cache_loc_dtype=torch.int64,
                is_multimodal=self.is_multimodal,
                enable_mamba_track=False,
                register_input_embeds=False,
                share_pool=False,
                source=None,
            ),
        )

    def _eager_fb_view(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> ForwardBatch:
        if envs.SGLANG_EAGER_INPUT_NO_COPY.get():
            return replace(forward_batch)
        raw_bs = forward_batch.batch_size
        raw_num_tokens = forward_batch.input_ids.shape[0]
        ensure = (
            self._ensure_eager_prefill_registry
            if forward_batch.forward_mode.is_extend(include_draft_extend_v2=True)
            else self._ensure_eager_decode_registry
        )
        registry = ensure(raw_bs, raw_num_tokens)
        registry.fill_from(
            forward_batch,
            raw_bs=raw_bs,
            padded_bs=raw_bs,
            raw_num_tokens=raw_num_tokens,
            padded_num_tokens=raw_num_tokens,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        return registry.extract_buffer(
            padded_bs=raw_bs,
            padded_num_tokens=raw_num_tokens,
            forward_batch_template=forward_batch,
        )

    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        if not self.server_args.enable_pdmux and self.device == "cuda":
            forward_batch = self._eager_fb_view(forward_batch, pp_proxy_tensors)
        # Set extra arguments
        pdmux_override = False
        if forward_batch.needs_forward_metadata_init():
            if hasattr(self.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                self.model.prepare_forward_batch(forward_batch)
            if self.server_args.enable_pdmux:
                self.decode_attn_backend.init_forward_metadata(forward_batch)
                # PDmux selects a per-stream backend; publish it to model-layer
                # readers via the active ForwardContext so RadixAttention etc.
                # dispatch against the right backend for this forward.
                pdmux_override = True
            else:
                self.attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors

        # Launch forward
        ctx = (
            self.device_timer.wrap(metadata={"category": "decode"})
            if self.device_timer
            else contextlib.nullcontext()
        )

        def _do_forward():
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

        with ctx:
            if pdmux_override:
                with forward_context(
                    ForwardContext(attn_backend=self.decode_attn_backend)
                ):
                    return _do_forward()
            return _do_forward()

    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors=None,
    ) -> Tuple[
        Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput], bool
    ]:
        # Setup extra arguments
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if (
            forward_batch.replace_embeds is not None
            and forward_batch.replace_positions is not None
        ):
            # Token embedding overrides: get base embeddings, scatter replacements
            if "input_embeds" not in kwargs:
                embed_layer = self.model.get_input_embeddings()
                kwargs["input_embeds"] = embed_layer(forward_batch.input_ids)
            kwargs["input_embeds"][forward_batch.replace_positions] = (
                forward_batch.replace_embeds.to(kwargs["input_embeds"].dtype)
            )
        if not self.is_generation:
            kwargs["get_embedding"] = True

        # Check piecewies cuda graph
        can_run_graph = (
            self.piecewise_cuda_graph_runner is not None
            and self.piecewise_cuda_graph_runner.can_run(forward_batch)
        )
        if can_run_graph:
            # TODO: device_timer.wrap is too broad here — it also includes
            # replay_prepare time. Move timing into the piecewise cuda graph
            # runner to capture only the model.forward part.
            ctx = (
                self.device_timer.wrap(metadata={"category": "extend"})
                if self.device_timer
                else contextlib.nullcontext()
            )
            with ctx:
                ret = self.piecewise_cuda_graph_runner.replay(forward_batch, **kwargs)
            return (ret, can_run_graph)

        if not self.server_args.enable_pdmux and self.device == "cuda":
            forward_batch = self._eager_fb_view(forward_batch, pp_proxy_tensors)

        # Launch model forward
        if forward_batch.needs_forward_metadata_init():
            if hasattr(self.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                self.model.prepare_forward_batch(forward_batch)
            self.attn_backend.init_forward_metadata(forward_batch)

        ctx = (
            self.device_timer.wrap(metadata={"category": "extend"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            if _is_hip and self.piecewise_cuda_graph_runner is not None:
                # AMD/HIP: when PCG is enabled but the batch exceeds max captured
                # size, run eagerly under enable_piecewise_cuda_graph() and
                # set_forward_context() so that (a) Dynamo guards on
                # _in_piecewise_cuda_graph stay consistent with the PCG-traced
                # graph (preventing runtime recompilation) and (b) PCG-specific
                # code paths (MoE, attention) can access their layer objects.
                with (
                    enable_piecewise_cuda_graph(),
                    set_forward_context(
                        forward_batch,
                        self.attention_layers,
                        getattr(self.model, "quant_config", None),
                        self.moe_layers,
                        self.moe_fusions,
                        dsa_indexers=self.dsa_indexers,
                    ),
                ):
                    ret = self.model.forward(
                        forward_batch.input_ids,
                        forward_batch.positions,
                        forward_batch,
                        **kwargs,
                    )
            else:
                ret = self.model.forward(
                    forward_batch.input_ids,
                    forward_batch.positions,
                    forward_batch,
                    **kwargs,
                )
        return (ret, can_run_graph)

    def _pic_prepopulate_hit_slots(self, forward_batch: "ForwardBatch") -> None:
        """Pre-populate hit segment private KV slots BEFORE the model forward.

        For each PIC cache-hit segment, copies the public (cached) KV to the
        newly-allocated private slot, applying delta-RoPE to the k_pe part so
        it reflects the segment's current sequence position rather than the
        position at which it was originally cached.

        This lets the model forward skip those tokens entirely (only miss segment
        tokens are passed as input), while the attention still sees correct K/V
        for all sequence positions via req_to_token_pool.

        Perf: stays in bf16 throughout (no fp32 round-trip), reuses a single
        dummy tensor across layers, and splits hit tokens into a delta=0 fast
        path (pure scatter-gather, no RoPE) vs a delta≠0 slow path. In the
        common quick_test_online.py setup, ~2/3 of hit tokens (SYS + C1) have
        delta=0 and take the fast path.
        """
        # pic_a3 new-path (§3.5, §3.9): layer 0-1 does FRESH full-length
        # forward with all tokens in input_ids; the private slots at hit
        # positions are l01_scratch that will be freed at the layer-2 boundary
        # (see _pic_a3_rewrite_req_to_token_pool_for_l2plus below). We do NOT
        # want to prepop stale public KV into l01_scratch — those slots must
        # be filled by the fresh layer 0-1 attention kernel writes. Short-
        # circuit here.
        if getattr(forward_batch, "pic_a3_new_path", False):
            return

        # For legacy pic mode (and pic_a3/pic_cacheblend .pt path) the pre-
        # population is LOAD-BEARING: hit tokens never enter the forward, so
        # every non-imp position at every layer must be filled here (attn_mqa
        # never writes them). Same story for plain `pic` mode.
        pub_slots  = getattr(forward_batch, "pic_all_hit_pub_slots",  None)
        priv_slots = getattr(forward_batch, "pic_all_hit_priv_slots", None)
        delta_pos  = getattr(forward_batch, "pic_all_hit_delta_pos",  None)

        if pub_slots is None or priv_slots is None or pub_slots.numel() == 0:
            return

        # Find rotary_emb from the model's first MLA attention layer.
        rotary_emb = None
        for module in self.model.modules():
            if hasattr(module, "rotary_emb") and module.rotary_emb is not None:
                rotary_emb = module.rotary_emb
                break

        # kv_lora_rank separates k_nope (position-free) from k_pe (position-encoded)
        kv_lora_rank = getattr(self.model_config, "kv_lora_rank", None)

        # Split hit tokens by whether they need delta-RoPE. Tokens with
        # delta_pos == 0 are at the same position as when cached → their k_pe
        # is already correct → pure gather-scatter, no RoPE compute.
        need_rope = (
            rotary_emb is not None
            and delta_pos is not None
            and kv_lora_rank is not None
        )
        if need_rope:
            rope_mask = delta_pos != 0
            has_any_rope = bool(rope_mask.any())
            has_any_direct = bool((~rope_mask).any())
            if has_any_rope:
                rope_pub  = pub_slots[rope_mask]
                rope_priv = priv_slots[rope_mask]
                rope_delta = delta_pos[rope_mask]
            if has_any_direct:
                direct_pub  = pub_slots[~rope_mask]
                direct_priv = priv_slots[~rope_mask]
            # Preallocate dummy zeros for rotary_emb once, reused across layers.
            # Shape is inferred from first-layer k_pe_old below.
            _dummy_rope: Optional[torch.Tensor] = None
        else:
            has_any_rope = False
            has_any_direct = True
            direct_pub, direct_priv = pub_slots, priv_slots

        for layer_id in range(self.start_layer, self.end_layer):
            buf = self.token_to_kv_pool.get_key_buffer(layer_id)

            # ── Fast path: direct scatter-gather for tokens with delta=0 ──
            # Single kernel call, no float promotion, no temporary allocation.
            if has_any_direct:
                buf[direct_priv] = buf[direct_pub]

            # ── Slow path: apply delta-RoPE to k_pe for shifted segments ──
            if has_any_rope:
                cached = buf[rope_pub]  # bf16 gather, one allocation
                k_pe_old = cached[..., kv_lora_rank:].contiguous()
                if _dummy_rope is None or _dummy_rope.shape != k_pe_old.shape:
                    _dummy_rope = torch.zeros_like(k_pe_old)
                _, k_pe_new = rotary_emb(rope_delta, _dummy_rope, k_pe_old)
                # In-place k_pe update on the gathered tensor (still bf16),
                # then scatter. Skips the redundant torch.cat + .to() round-trip.
                cached[..., kv_lora_rank:] = k_pe_new.to(cached.dtype)
                buf[rope_priv] = cached

        # DSA index-K prepopulation: page-level copy (unchanged; already bf16-safe).
        # 关键修复：DSA index-K buffer 是 *page 布局* —— shape (num_pages, page_bytes)，
        # 第一维是 page 索引(≈num_slots/page_size)，不是 token slot 索引！
        # 段已 64 对齐 → 命中段 slot 都是整 page，可安全提取 page 索引做整页复制。
        try:
            ps = self.token_to_kv_pool.page_size
            if ps > 1:
                if pub_slots.numel() % ps == 0 and priv_slots.numel() % ps == 0:
                    pub_pages = (pub_slots[::ps] // ps).long()
                    priv_pages = (priv_slots[::ps] // ps).long()
                    for layer_id in range(self.start_layer, self.end_layer):
                        dsa_buf = self.token_to_kv_pool.get_index_k_with_scale_buffer(  # type: ignore[union-attr]
                            layer_id=layer_id
                        )
                        dsa_buf[priv_pages] = dsa_buf[pub_pages]
            else:
                for layer_id in range(self.start_layer, self.end_layer):
                    dsa_buf = self.token_to_kv_pool.get_index_k_with_scale_buffer(  # type: ignore[union-attr]
                        layer_id=layer_id
                    )
                    dsa_buf[priv_slots] = dsa_buf[pub_slots]
        except (AttributeError, TypeError):
            pass  # DSA index-K copy is best-effort

    def _pic_writeback_mla_kv(self, forward_batch: "ForwardBatch") -> None:
        """PIC transition_rope: copy MLA KV from private slots to public slots.

        For miss-segment token positions (pic_public_out_loc >= 0), the model
        has just written latent KV to the private slot (out_cache_loc[i]).
        We copy that latent to the corresponding public slot so future cache
        hits can load position-free K from there and re-apply RoPE.
        """
        kv_pool = self.token_to_kv_pool

        # PIC writeback (all modes): copy fresh miss-segment K from priv slots
        # to pub slots so future requests can hit the PIC cache. The pair
        # (pic_public_out_loc, out_cache_loc) is aligned 1:1 by construction;
        # -1 entries in pic_public_out_loc mark last-segment tokens which are
        # never cached and are filtered out here.
        #
        # BUGFIX (pic_a3 new-path): the miss→public writeback must use the
        # dedicated l2plus tensors built by
        # _pic_a3_rewrite_req_to_token_pool_for_l2plus. Those are consumed here
        # AFTER the model forward, but forward_extend ran the model on an
        # _eager_fb_view COPY of forward_batch — so the rewrite's forward_batch
        # mutations (and the layer-1 pic_public_out_loc swap) are gone on this
        # original forward_batch. The rewrite therefore also stashes them on the
        # shared req (reached via _reqs_ref); read them from there. Without this,
        # any segment first cached under the new-path (a miss segment in a
        # request that also has a hit) kept ZERO public KV, so reusing it later
        # read zeros → garbage.
        _a3_new = getattr(forward_batch, "pic_a3_new_path", False)
        _a3_req = None
        if _a3_new:
            _reqs = getattr(forward_batch, "_reqs_ref", None)
            _a3_req = _reqs[0] if _reqs else None
            pub_loc = getattr(_a3_req, "pic_a3_l2plus_pub_out_loc", None)
        else:
            pub_loc = getattr(forward_batch, "pic_public_out_loc", None)
        if pub_loc is not None:
            out_loc = (
                _a3_req.pic_a3_l2plus_out_cache_loc
                if _a3_new
                else forward_batch.out_cache_loc
            )

            # Ensure pub_loc and out_loc have the same length.  They may differ if
            # attn_tp_scatter padded out_loc to a multiple of tp_size.
            n = min(pub_loc.shape[0], out_loc.shape[0])
            pub_loc = pub_loc[:n]
            out_loc = out_loc[:n]

            valid = pub_loc >= 0
            if valid.any():
                pub_slots = pub_loc[valid]       # (n_miss,)
                priv_slots = out_loc[valid]      # (n_miss,)
                for layer_id in range(self.start_layer, self.end_layer):
                    buf = kv_pool.get_key_buffer(layer_id)  # (total_slots, kv_dim)
                    buf[pub_slots] = buf[priv_slots]

    # ========================================================================
    # PIC + A³/CacheBlend new-path helpers (v1)
    # See scripts/pic_a3_full_recompute_plan.md §3.4, §3.6, §3.10.
    # ========================================================================
    def _pic_a3_pick_imp(self, forward_batch: "ForwardBatch") -> torch.Tensor:
        """Run A³ or CacheBlend imp selection using layer-1 latent stash.

        Reads forward_batch.pic_a3_layer1_stash (set by forward_mla under
        reuse_check_state=='checking'). For CacheBlend, also reads the
        layer-1 latent from PICache-populated public slots via
        pic_a3_hit_pub_slots. Sets forward_batch.pic_a3_imp_indices and
        forward_batch.pic_a3_q_positions_l2plus_per_req.

        v1: single-request batches only (aligned with _maybe_populate_reuse_fields).
        """
        assert forward_batch.batch_size == 1, (
            "pic_a3 new-path v1 supports single-request batches only "
            f"(got batch_size={forward_batch.batch_size})"
        )
        stash = forward_batch.pic_a3_layer1_stash
        assert stash is not None, (
            "pic_a3 layer-1 stash missing — forward_mla must run at "
            "CHECK_LAYER=1 with reuse_check_state='checking' before this call"
        )
        # Single-layer (Phase A) imp selection from the layer-1 stash. The
        # pic_a3_oracle mode does NOT go through here — it uses Phase B keepalive
        # windowed re-selection (_pic_a3_keepalive_window), where the stash is
        # already oracle-derived via _pic_a3_oracle_capture_or_inject.
        imp_indices = self._pic_a3_select_from_stash(forward_batch, stash)
        forward_batch.pic_a3_imp_indices = imp_indices
        # Q positions for layer 2+ = miss ∪ imp, sorted ascending.
        # NB: last_len covers ONLY the query region (last segment length,
        # e.g. 64 for the Q segment in quick_test_online). miss positions
        # like C2 (in the middle of the sequence) are NOT in last_indices —
        # they must be added explicitly. imp_indices from pick_imp_latent_*
        # only contains: topk(context) + last_indices(query region).
        # So we need: imp_indices ∪ miss_positions.
        reqs = getattr(forward_batch, "_reqs_ref", None)
        import logging as _lg_union
        import os as _os_union
        _log_u = _lg_union.getLogger(__name__)

        # DIAG: force imp = ALL positions (equivalent to full recompute at
        # layer 2+). If output is still wrong, plumbing is broken; if correct,
        # imp selection quality is the issue.
        _force_all_imp = _os_union.environ.get("PIC_A3_FORCE_ALL_IMP", "0") == "1"
        # DIAG: only miss (no hit-imp). Layer 2+ processes just miss+Q, hit
        # goes fully through prepop path. Tests if mixing fresh-imp K/V with
        # prepop'd K/V at hit-adjacent positions causes score inconsistency.
        _miss_only_imp = _os_union.environ.get("PIC_A3_MISS_ONLY_IMP", "0") == "1"

        if reqs is not None and len(reqs) >= 1:
            req = reqs[0]
            _miss_segs = getattr(req, "pic_miss_segments", [])
            _pic_segs = getattr(req, "pic_segments", None)
            _hit_segs = getattr(req, "pic_hit_segments", None)
            _log_u.warning(
                f"[PIC-A3-UNION-DBG] rid={getattr(req, 'rid', '?')[:8]} "
                f"reqs_len={len(reqs)} "
                f"pic_segs={_pic_segs} "
                f"pic_hit_segs_lens={[e - s for (s, e, _) in (_hit_segs or [])]} "
                f"miss_segs={_miss_segs} imp_before={imp_indices.numel()} "
                f"force_all={_force_all_imp}"
            )
            if _force_all_imp:
                # Bypass: all positions become imp → layer 2+ processes
                # everything → equivalent to full recompute at layer 2+
                full_len = int(getattr(req, "pic_a3_full_len", imp_indices.numel()))
                imp_indices = torch.arange(
                    0, full_len,
                    dtype=imp_indices.dtype, device=imp_indices.device,
                )
                forward_batch.pic_a3_imp_indices = imp_indices
                _log_u.warning(
                    f"[PIC-A3-UNION-DBG] FORCED all imp = {full_len}"
                )
            elif _miss_only_imp:
                # Bypass: imp = only miss positions (no topk from hit).
                # Layer 2+ processes just miss tokens; hit positions ALL go
                # through prepop path. Isolates whether mixing fresh vs prepop'd
                # at hit positions causes attention inconsistency.
                miss_positions = []
                for (s, e) in _miss_segs:
                    miss_positions.extend(range(s, e))
                if miss_positions:
                    imp_indices = torch.tensor(
                        miss_positions, dtype=imp_indices.dtype,
                        device=imp_indices.device,
                    ).sort()[0]
                    forward_batch.pic_a3_imp_indices = imp_indices
                _log_u.warning(
                    f"[PIC-A3-UNION-DBG] MISS_ONLY imp = {imp_indices.numel()}"
                )
            else:
                miss_positions = []
                for (s, e) in _miss_segs:
                    miss_positions.extend(range(s, e))
                if miss_positions:
                    miss_pos_tensor = torch.tensor(
                        miss_positions, dtype=imp_indices.dtype,
                        device=imp_indices.device,
                    )
                    imp_indices = torch.unique(
                        torch.cat([imp_indices, miss_pos_tensor])
                    )
                    forward_batch.pic_a3_imp_indices = imp_indices
                    _log_u.warning(
                        f"[PIC-A3-UNION-DBG] added {len(miss_positions)} miss pos, "
                        f"imp_after={imp_indices.numel()}"
                    )
                else:
                    _log_u.warning(
                        "[PIC-A3-UNION-DBG] no miss positions to add"
                    )
        else:
            _log_u.warning(
                f"[PIC-A3-UNION-DBG] reqs is None or empty: reqs={reqs}"
            )

        # PIC_A3_FORCE_HIT_HEAD_IMP=N: force the first N tokens of every hit
        # segment into imp at the layer-1 boundary too. Needed because
        # PIC_A3_CLIP_REAL_L1=1 (now default) routes layer 1 through pick_imp,
        # which bypasses _pic_a3_apply_imp_diag_env (where the deep check layers
        # apply the same head-force). Mirrors that helper. Default 0 = zero
        # regression.
        _hit_head = int(os.environ.get("PIC_A3_FORCE_HIT_HEAD_IMP", "0") or "0")
        if _hit_head > 0 and reqs is not None and len(reqs) >= 1:
            _hit_segs = getattr(reqs[0], "pic_hit_segments", None) or []
            _full_hh = int(getattr(reqs[0], "pic_a3_full_len", imp_indices.numel()))
            _heads = []
            for (s, e, _h) in _hit_segs:
                _end = min(int(s) + _hit_head, int(e), _full_hh)
                _heads.extend(range(int(s), _end))
            if _heads:
                _ht = torch.tensor(
                    _heads, dtype=imp_indices.dtype, device=imp_indices.device
                )
                imp_indices = torch.unique(torch.cat([imp_indices, _ht]))
                forward_batch.pic_a3_imp_indices = imp_indices
                _log_u.warning(
                    f"[PIC-A3-FORCE-HIT-HEAD] (pick_imp L1) N={_hit_head} "
                    f"hit_segs={len(_hit_segs)} forced={len(_heads)} "
                    f"imp_now={imp_indices.numel()} full_len={_full_hh}"
                )

        # === DIAGNOSTIC: log imp distribution per pic segment ===
        import logging as _lg
        _log = _lg.getLogger(__name__)
        try:
            reqs_diag = getattr(forward_batch, "_reqs_ref", None)
            if reqs_diag is not None and len(reqs_diag) >= 1:
                _req_diag = reqs_diag[0]
                _imp_set = set(int(x) for x in imp_indices.tolist())
                _dist = []
                for (s, e) in getattr(_req_diag, "pic_segments", []):
                    _in_seg = sum(1 for p in range(s, e) if p in _imp_set)
                    _dist.append(f"[{s},{e})={_in_seg}/{e-s}")
                _log.warning(
                    f"[PIC-A3-IMP-DIST] total imp={imp_indices.numel()} "
                    f"segs: {' '.join(_dist)}"
                )
                # Full imp position list — parsed by test/manual/pic_a3_imp_token_view.py.
                # Format: `[PIC-A3-IMP-LIST] rid=<8char> reuse=<method> N=<n_full> `
                # `last_len=<int> prefix_len=<int> n_imp=<int> positions=<comma-ints>`.
                # Emitted as a single WARNING line so the client script can regex out
                # positions and decode them via the tokenizer.
                _rid_short = getattr(_req_diag, "rid", "?")[:8]
                _pos_csv = ",".join(str(int(x)) for x in imp_indices.tolist())
                _log.warning(
                    f"[PIC-A3-IMP-LIST] rid={_rid_short} "
                    f"reuse={forward_batch.reuse_method or '?'} "
                    f"N={int(getattr(_req_diag, 'pic_a3_full_len', imp_indices.numel()))} "
                    f"last_len={_last_len} prefix_len={prefix_len} "
                    f"n_imp={imp_indices.numel()} positions={_pos_csv}"
                )
        except Exception:
            pass

        forward_batch.pic_a3_q_positions_l2plus_per_req = [imp_indices]

        # Free the stash — no longer needed after layer 1
        forward_batch.pic_a3_layer1_stash = None
        return imp_indices

    def _pic_a3_select_from_stash(
        self, forward_batch: "ForwardBatch", stash: dict, layer_id: Optional[int] = None
    ) -> torch.Tensor:
        """Core A³/CacheBlend imp selection from a single check-layer stash.

        Returns raw imp_indices (topk(context) ∪ query-region) BEFORE the
        miss-union. Shared by _pic_a3_pick_imp (real, layer-1 boundary) and
        _pic_a3_pick_imp_probe (read-only, other check layers in Phase A).
        """
        from sglang.srt.models.reuse_utils import (
            pick_imp_latent_a3,
            pick_imp_latent_a3_hit_only,
            pick_imp_latent_blend,
            read_layer_latent_from_pic_public,
            REUSE_A3,
            REUSE_BLEND,
        )

        # last_len defaults to miss_len (query region always kept as imp).
        _last_len = forward_batch.reuse_last_len
        if _last_len is None:
            req = self.req_pool[0] if hasattr(self, "req_pool") else None
            _last_len = int(getattr(req, "pic_a3_miss_len", 0)) if req else 0
            assert _last_len > 0, "cannot determine last_len for pic_a3 imp pick"

        recomp_ratio = float(forward_batch.recomp_ratio or 0.15)
        prefix_len = int(forward_batch.reuse_prefix_len or 0)

        reuse_method = forward_batch.reuse_method or ""
        # Per-segment imp budget when HIT_ONLY_IMP (always) or HYBRID_IMP at a
        # DEEP reselect layer (layer_id > 1). Global top-k otherwise (layer 1 in
        # hybrid → concentrate budget on the answer chunk). See env docstrings.
        _use_hit_only = envs.SGLANG_PIC_A3_HIT_ONLY_IMP.get() or (
            envs.SGLANG_PIC_A3_HYBRID_IMP.get()
            and layer_id is not None
            and int(layer_id) > 1
        )
        if REUSE_A3 in reuse_method:
            if _use_hit_only:
                # Restrict topk to hit-segment positions with per-segment budget.
                _reqs_hit = getattr(forward_batch, "_reqs_ref", None)
                _hit_segs_raw = []
                if _reqs_hit is not None and len(_reqs_hit) >= 1:
                    _hit_segs_raw = getattr(_reqs_hit[0], "pic_hit_segments", []) or []
                _hit_segs = [(int(s), int(e)) for (s, e, *_rest) in _hit_segs_raw]
                imp_indices = pick_imp_latent_a3_hit_only(
                    q_absorbed=stash["q_absorbed"],
                    k_latent=stash["k_latent"],
                    q_pe=stash["q_pe"],
                    k_pe=stash["k_pe"],
                    softmax_scale=stash["softmax_scale"],
                    last_len=_last_len,
                    recomp_ratio=recomp_ratio,
                    hit_segments=_hit_segs,
                    prefix_len=prefix_len,
                )
            else:
                imp_indices = pick_imp_latent_a3(
                    q_absorbed=stash["q_absorbed"],
                    k_latent=stash["k_latent"],
                    q_pe=stash["q_pe"],
                    k_pe=stash["k_pe"],
                    softmax_scale=stash["softmax_scale"],
                    last_len=_last_len,
                    recomp_ratio=recomp_ratio,
                    prefix_len=prefix_len,
                )
        elif REUSE_BLEND in reuse_method:
            # CacheBlend: need latent_old from PIC public cache
            hit_pub = forward_batch.pic_a3_hit_pub_slots
            assert hit_pub is not None, (
                "pic_cacheblend new-path needs pic_a3_hit_pub_slots on "
                "forward_batch (set by pic_alloc); got None"
            )
            layer_id = int(stash["layer_id"])
            kv_buf = self.token_to_kv_pool.get_key_buffer(layer_id)  # (slots, 1, D)
            latent_old = read_layer_latent_from_pic_public(kv_buf, hit_pub)
            latent_new = torch.cat([stash["k_latent"], stash["k_pe"]], dim=-1)
            imp_indices = pick_imp_latent_blend(
                latent_new=latent_new,
                latent_old=latent_old,
                kv_lora_rank=int(stash["kv_lora_rank"]),
                last_len=_last_len,
                recomp_ratio=recomp_ratio,
                prefix_len=prefix_len,
            )
        else:
            raise ValueError(
                f"pic_a3 unknown reuse_method: '{reuse_method}' "
                f"(expected substring '{REUSE_A3}' or '{REUSE_BLEND}')"
            )
        return imp_indices

    def _pic_a3_union_miss(
        self, forward_batch: "ForwardBatch", imp_indices: torch.Tensor
    ) -> torch.Tensor:
        """Union imp_indices with all miss positions (read-only; no fb mutation).

        Mirrors the plain-union branch of _pic_a3_pick_imp without the
        diagnostic env overrides. Used by the Phase-A probe.
        """
        reqs = getattr(forward_batch, "_reqs_ref", None)
        if not reqs:
            return imp_indices
        _miss_segs = getattr(reqs[0], "pic_miss_segments", []) or []
        miss_positions = []
        for (s, e) in _miss_segs:
            miss_positions.extend(range(int(s), int(e)))
        if not miss_positions:
            return imp_indices
        miss_pos_tensor = torch.tensor(
            miss_positions, dtype=imp_indices.dtype, device=imp_indices.device
        )
        return torch.unique(torch.cat([imp_indices, miss_pos_tensor]))

    def _pic_a3_pick_imp_probe(
        self, forward_batch: "ForwardBatch", layer_id: int
    ) -> None:
        """Phase-A read-only probe: run imp selection from the stash saved at
        `layer_id`, union with miss, and log the result + overlap with the
        applied layer-1 imp set. Does NOT mutate the applied selection.

        NB (Phase-A limitation): the production clip at the layer-1 boundary
        drops non-imp tokens, so a stash saved at layer>1 spans only the
        surviving (miss+imp) rows — this probe re-ranks within that set, not the
        full sequence. Full-sequence multi-layer selection is Phase B.
        """
        import logging as _lg

        _log = _lg.getLogger(__name__)
        stash_by_layer = getattr(forward_batch, "pic_a3_stash_by_layer", None)
        stash = stash_by_layer.get(int(layer_id)) if stash_by_layer else None
        if stash is None:
            return
        try:
            imp = self._pic_a3_select_from_stash(forward_batch, stash)
            imp = self._pic_a3_union_miss(forward_batch, imp)
        except Exception as _e:  # probe must never break the forward
            _log.warning(f"[PIC-A3-PROBE] layer={layer_id} selection failed: {_e}")
            return
        applied = getattr(forward_batch, "pic_a3_imp_indices", None)
        overlap = -1
        if applied is not None:
            _a = set(int(x) for x in applied.tolist())
            _b = set(int(x) for x in imp.tolist())
            overlap = len(_a & _b)
        _log.warning(
            f"[PIC-A3-PROBE] layer={layer_id} imp={int(imp.numel())} "
            f"overlap_l1={overlap} "
            f"applied_l1={int(applied.numel()) if applied is not None else -1}"
        )

    def _pic_a3_prepop_hit_slots_for_l2plus(
        self,
        forward_batch: "ForwardBatch",
        layer_start: Optional[int] = None,
        layer_end: Optional[int] = None,
        exclude_positions: Optional[set] = None,
    ) -> None:
        """Populate KV buffers at hit-position l01_scratch slots with
        delta-RoPE-corrected public cache values.

        Layer 0-1 wrote FRESH KV to l01_scratch slots (correct for those
        layers). Layer 2+ needs KV for hit-non-imp positions too, but doesn't
        recompute them. This method fills those buffers at those slot indices
        with the correct KV — same delta-RoPE trick as _pic_prepopulate_hit_slots
        but only for hit-position slots (not miss).

        Args (all optional — defaults reproduce the production single-boundary
        behavior, i.e. layers [2, N) over ALL hit positions):
          layer_start / layer_end: layer range [start, end) to fill. Phase B
            keep-alive passes a per-window range [check_layer+1, next_check).
          exclude_positions: set of positions to SKIP (the imp positions, which
            get fresh KV, not cached). Phase B keep-alive passes the current
            window's imp set so only NON-imp hit positions get cached.

        Called after _pic_a3_rewrite_req_to_token_pool_for_l2plus (production),
        or from _pic_a3_keepalive_window per window (Phase B).
        """
        # Get the single req for v1
        reqs = getattr(forward_batch, "_reqs_ref", None)
        assert reqs is not None and len(reqs) == 1
        req = reqs[0]

        # Collect flat (pub, priv, delta) tensors across all hit segments.
        # `priv_slots` here is the l01_scratch slice for the hit segment
        # (set in pic_alloc's pic_a3 branch to l01_slice.clone()).
        _pub_pieces: List[torch.Tensor] = []
        _priv_pieces: List[torch.Tensor] = []
        _delta_pieces: List[torch.Tensor] = []
        device = self.token_to_kv_pool.get_key_buffer(self.start_layer).device
        for (s, e), hit_info in req.pic_rope_hit_private_slots.items():
            priv_slots, pub_kv_slots, old_start = hit_info
            seg_len = e - s
            if exclude_positions is None:
                _keep = None
            else:
                _keep = [p - s for p in range(s, e) if p not in exclude_positions]
                if not _keep:
                    continue
            _pub = pub_kv_slots[:seg_len].to(device, non_blocking=True)
            _priv = priv_slots[:seg_len].to(device, non_blocking=True)
            _dlt = torch.full(
                (seg_len,), s - old_start, dtype=torch.int64, device=device
            )
            if _keep is not None:
                _kt = torch.tensor(_keep, dtype=torch.long, device=device)
                _pub, _priv, _dlt = _pub[_kt], _priv[_kt], _dlt[_kt]
            _pub_pieces.append(_pub)
            _priv_pieces.append(_priv)
            _delta_pieces.append(_dlt)
        if not _pub_pieces:
            return
        pub_slots = torch.cat(_pub_pieces)
        priv_slots = torch.cat(_priv_pieces)
        delta_pos = torch.cat(_delta_pieces)

        # Find rotary_emb (same pattern as _pic_prepopulate_hit_slots)
        rotary_emb = None
        for module in self.model.modules():
            if hasattr(module, "rotary_emb") and module.rotary_emb is not None:
                rotary_emb = module.rotary_emb
                break
        kv_lora_rank = getattr(self.model_config, "kv_lora_rank", None)
        need_rope = (
            rotary_emb is not None
            and delta_pos is not None
            and kv_lora_rank is not None
        )

        if need_rope:
            rope_mask = delta_pos != 0
            has_any_rope = bool(rope_mask.any())
            has_any_direct = bool((~rope_mask).any())
            if has_any_rope:
                rope_pub  = pub_slots[rope_mask]
                rope_priv = priv_slots[rope_mask]
                rope_delta = delta_pos[rope_mask]
            if has_any_direct:
                direct_pub  = pub_slots[~rope_mask]
                direct_priv = priv_slots[~rope_mask]
            _dummy_rope: Optional[torch.Tensor] = None
        else:
            has_any_rope = False
            has_any_direct = True
            direct_pub, direct_priv = pub_slots, priv_slots

        # Layer range: default [2, N) (production); Phase B keep-alive passes a
        # per-window sub-range. Layers 0-1 are skipped by default (fresh KV).
        _reuse_check_layer = 1  # aligned with reuse_utils.CHECK_LAYER
        start_layer_l2plus = (
            layer_start if layer_start is not None
            else max(self.start_layer, _reuse_check_layer + 1)
        )
        _end_layer_l2plus = layer_end if layer_end is not None else self.end_layer
        for layer_id in range(start_layer_l2plus, _end_layer_l2plus):
            buf = self.token_to_kv_pool.get_key_buffer(layer_id)

            if has_any_direct:
                buf[direct_priv] = buf[direct_pub]

            if has_any_rope:
                cached = buf[rope_pub]
                k_pe_old = cached[..., kv_lora_rank:].contiguous()
                if _dummy_rope is None or _dummy_rope.shape != k_pe_old.shape:
                    _dummy_rope = torch.zeros_like(k_pe_old)
                _, k_pe_new = rotary_emb(rope_delta, _dummy_rope, k_pe_old)
                cached[..., kv_lora_rank:] = k_pe_new.to(cached.dtype)
                buf[rope_priv] = cached

        # DSA index-K prepopulation for layers 2..N-1 (same page trick as
        # _pic_prepopulate_hit_slots — best-effort, wrapped in try/except).
        try:
            ps = self.token_to_kv_pool.page_size
            if ps > 1 and pub_slots.numel() % ps == 0 and priv_slots.numel() % ps == 0:
                pub_pages = (pub_slots[::ps] // ps).long()
                priv_pages = (priv_slots[::ps] // ps).long()
                for layer_id in range(start_layer_l2plus, _end_layer_l2plus):
                    dsa_buf = self.token_to_kv_pool.get_index_k_with_scale_buffer(  # type: ignore[union-attr]
                        layer_id=layer_id
                    )
                    dsa_buf[priv_pages] = dsa_buf[pub_pages]
            elif ps <= 1:
                for layer_id in range(start_layer_l2plus, _end_layer_l2plus):
                    dsa_buf = self.token_to_kv_pool.get_index_k_with_scale_buffer(  # type: ignore[union-attr]
                        layer_id=layer_id
                    )
                    dsa_buf[priv_slots] = dsa_buf[pub_slots]
        except (AttributeError, TypeError):
            pass

    def _pic_a3_oracle_capture_or_inject(
        self,
        forward_batch: "ForwardBatch",
        layer_id: int,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: Optional[torch.Tensor] = None,
    ):
        """pic_a3_oracle (SGLANG_PIC_A3_ORACLE) in-process capture + inject.

        Called at each Phase B check layer, BEFORE the layer runs, so the layer
        computes Q/K/V from the (possibly oracle-replaced) input hidden state.

        The residual stream entering this layer == hidden_states + residual (the
        fused add-RMSNorm input_layernorm sums them). So:
          • CAPTURE (warmup, isolated): the FIRST time a document segment is
            computed fresh (it is a MISS with no stored hidden yet), stash its
            per-layer input residual stream, keyed by PIC segment hash. Under the
            warmup-first flow (pic_w1/w2/w3 each carry ONE document) this is the
            segment's isolated-context representation. The last (query) segment
            is never captured.
          • INJECT (measure): for each HIT document segment with a stored hidden,
            overwrite its layer input with the isolated-warmup value
            (hidden_states[seg]=oracle_rs, residual[seg]=0 → input_layernorm sees
            residual+hidden == oracle_rs). The layer then re-picks imp and
            recomputes imp KV from accurate (undrifted) input (做法 1).

        Returns (hidden_states, residual), modified in place for injected
        segments. No-op (returns inputs unchanged) unless the oracle flag is on,
        the batch is single-request, and residual is present (layer >= 1).
        """
        if not envs.SGLANG_PIC_A3_ORACLE.get():
            return hidden_states, residual
        if residual is None or not torch.is_tensor(hidden_states):
            return hidden_states, residual
        if getattr(hidden_states, "_sglang_needs_allreduce_fusion", False):
            # MLP all-reduce is fused into the NEXT layer's input_layernorm, so
            # hidden_states here is a pre-all-reduce partial and
            # hidden_states+residual is NOT the materialized residual stream —
            # capturing/injecting it would be wrong. pic_a3_oracle runs with
            # --disable-cuda-graph where this fusion is off, so this is a
            # defensive skip (degrades to plain keepalive at this layer).
            import logging as _lg_orf

            _lg_orf.getLogger(__name__).warning(
                f"[PIC-A3-ORACLE] layer={int(layer_id)} skipped: MLP allreduce "
                f"fusion active (residual stream not materialized)"
            )
            return hidden_states, residual
        reqs = getattr(forward_batch, "_reqs_ref", None)
        if not reqs or len(reqs) != 1:
            return hidden_states, residual
        req = reqs[0]

        from sglang.srt.pic.hasher import segment_hash

        store = getattr(self, "_pic_a3_oracle_hidden", None)
        if store is None:
            store = self._pic_a3_oracle_hidden = {}
        L = int(layer_id)

        # Residual stream entering this layer. Computed pre-inject; miss ∩ hit
        # == ∅, so the miss slices read for capture are unaffected by the
        # hit-slice inject below.
        rs = hidden_states + residual

        # Route-A: after the first clip the main-loop tensors are clip-local
        # (rows = miss∪imp', NOT full_len). Absolute segment coords [s:e] are then
        # invalid, so INJECT/CAPTURE (indexed by abs coords) are skipped; only the
        # positions-masked query-segment stash (valid either way) runs.
        _full_len = int(getattr(req, "pic_a3_full_len", 0) or 0)
        _is_clipped = _full_len > 0 and int(hidden_states.shape[0]) != _full_len

        _last = req.pic_segments[-1] if getattr(req, "pic_segments", None) else None
        input_ids = forward_batch.input_ids
        # pic_a3_oracle Route-A: stash the query (last) segment's residual stream
        # for clip-time full-length re-projection. positions-mask works whether
        # hidden is full-length or clipped (positions carry abs values in both).
        if _last is not None and envs.SGLANG_PIC_A3_CLIP_CAPTURE.get():
            _qs, _qe = _last
            # Use the LOCAL positions (matches current hidden rows; clipped after
            # the first window). forward_batch.positions may be stale full-length.
            _pos = positions if positions is not None else forward_batch.positions
            if int(_pos.shape[0]) == int(rs.shape[0]):
                _qmask = (_pos >= int(_qs)) & (_pos < int(_qe))
                if bool(_qmask.any()):
                    forward_batch._pic_a3_last_seg_rs = rs[_qmask].detach().clone()

        # ── INJECT: hit segments → isolated-warmup oracle hidden (full-len only) ──
        n_inj = 0
        _inj_dbg = []       # (s,e,hash4) injected
        _inj_nostore = []   # (s,e,hash4) hit seg but no stored oracle at this layer
        _inj_shape = []     # (s,e,hash4) stored but length mismatch
        if not _is_clipped:
            for (s, e, seg_hash) in getattr(req, "pic_hit_segments", None) or []:
                _h4 = seg_hash.hex()[:4]
                per_layer = store.get(seg_hash)
                oracle_rs = per_layer.get(L) if per_layer is not None else None
                if oracle_rs is None:
                    _inj_nostore.append((s, e, _h4))
                    continue
                if int(oracle_rs.shape[0]) != int(e - s):
                    _inj_shape.append((s, e, _h4))
                    continue
                hidden_states[s:e] = oracle_rs.to(hidden_states.dtype)
                residual[s:e] = 0
                n_inj += 1
                _inj_dbg.append((s, e, _h4))

        # ── CAPTURE: first-seen miss document segments (isolated) ──
        n_cap = 0
        _cap_dbg = []       # (s,e,hash4) captured
        _cap_present = []   # (s,e,hash4) miss seg already stored at this layer
        _cap_miss = [] if _is_clipped else (getattr(req, "pic_miss_segments", None) or [])
        for (s, e) in _cap_miss:
            if _last is not None and (s, e) == _last:
                continue  # never capture the query (last) segment
            seg_hash = segment_hash(input_ids[s:e])
            _h4 = seg_hash.hex()[:4]
            per_layer = store.setdefault(seg_hash, {})
            if L in per_layer:
                _cap_present.append((s, e, _h4))
                continue  # capture-if-absent: keep the first (isolated) copy
            per_layer[L] = rs[s:e].detach().clone()
            n_cap += 1
            _cap_dbg.append((s, e, _h4))

        # Route-A: also capture HIT segments (capture-if-absent) so SYS and any
        # segment that is a hit at warmup (never a miss → skipped by the loop
        # above) still gets a stored isolated hidden. Reads pre-inject rs. Gated
        # on the clip flag → zero effect on the existing full-length keepalive.
        if envs.SGLANG_PIC_A3_CLIP_CAPTURE.get() and not _is_clipped:
            for (s, e, seg_hash) in getattr(req, "pic_hit_segments", None) or []:
                if _last is not None and (s, e) == (_last[0], _last[1]):
                    continue
                per_layer = store.setdefault(seg_hash, {})
                if L in per_layer:
                    continue
                per_layer[L] = rs[s:e].detach().clone()
                n_cap += 1
                _cap_dbg.append((s, e, seg_hash.hex()[:4]))

        # DIAG (rank 0 only): full per-segment breakdown so we can see why some
        # hit segments aren't injected (noStore / shapeBad) and what warmup
        # captured. Remove once 4/4 coverage is confirmed.
        from sglang.srt.distributed.parallel_state import (
            get_tensor_model_parallel_rank,
        )

        if (n_inj or n_cap or _inj_nostore or _inj_shape) and (
            get_tensor_model_parallel_rank() == 0
        ):
            import logging as _lg_orc

            _hit = [
                (s, e, h.hex()[:4])
                for (s, e, h) in (getattr(req, "pic_hit_segments", None) or [])
            ]
            _miss = list(getattr(req, "pic_miss_segments", None) or [])
            _lg_orc.getLogger(__name__).warning(
                f"[PIC-A3-ORACLE] layer={L} injected={n_inj} captured={n_cap} | "
                f"segs={getattr(req, 'pic_segments', None)} hit={_hit} miss={_miss} "
                f"| cap={_cap_dbg} capPresent={_cap_present} inj={_inj_dbg} "
                f"noStore={_inj_nostore} shapeBad={_inj_shape} "
                f"storeKeys={len(store)}"
            )
        return hidden_states, residual

    def _pic_a3_apply_imp_diag_env(
        self, imp_set: set, full_len: int, layer_id: int, forward_batch=None
    ) -> set:
        """Apply pic_a3 imp diagnostic env overrides (shared by keepalive_window
        and Route-A reselect). PIC_A3_FORCE_ALL_IMP → all positions imp (all
        fresh, no cached-K). PIC_A3_KEEPALIVE_FIXED_IMP → every check layer reuses
        the layer-1 imp split (no per-layer re-select).
        PIC_A3_FORCE_HIT_HEAD_IMP=N → force the FIRST N tokens of EVERY hit
        segment into imp (recomputed fresh from the in-context input), so each
        reused doc segment's head is never served from isolated cached-K — targets
        the isolated-K vs in-context-hidden mismatch that collapses the oracle on
        multi-doc data. N is clamped per segment (short segs forced whole) and
        should be a multiple of the 64 page size (128 in experiments). Applies at
        EVERY check layer incl. layer 1: on the default oracle run both
        _pic_a3_oracle_reselect_full and _pic_a3_keepalive_window route here.
        Default 0=off (zero regression)."""
        # Force the head of every HIT segment into imp (fresh recompute). Runs
        # BEFORE the other overrides so the forced heads are also captured by
        # KEEPALIVE_FIXED_IMP's layer-1 snapshot and survive FORCE_ALL_IMP.
        _hit_head = int(os.environ.get("PIC_A3_FORCE_HIT_HEAD_IMP", "0") or "0")
        if _hit_head > 0 and forward_batch is not None:
            reqs = getattr(forward_batch, "_reqs_ref", None)
            if reqs and len(reqs) >= 1:
                _hit_segs = getattr(reqs[0], "pic_hit_segments", None) or []
                imp_set = set(imp_set)
                _forced = 0
                for (s, e, _h) in _hit_segs:
                    _end = min(int(s) + _hit_head, int(e), int(full_len))
                    for p in range(int(s), _end):
                        imp_set.add(p)
                        _forced += 1
                if int(layer_id) == 1:
                    import logging as _lg_fh

                    _lg_fh.getLogger(__name__).warning(
                        f"[PIC-A3-FORCE-HIT-HEAD] N={_hit_head} "
                        f"hit_segs={len(_hit_segs)} forced={_forced} "
                        f"imp_now={len(imp_set)} full_len={full_len}"
                    )
        if os.environ.get("PIC_A3_FORCE_ALL_IMP") == "1":
            return set(range(int(full_len)))
        if os.environ.get("PIC_A3_KEEPALIVE_FIXED_IMP") == "1":
            if int(layer_id) == 1:
                self._pic_a3_ka_fixed_imp = set(imp_set)
            elif getattr(self, "_pic_a3_ka_fixed_imp", None) is not None:
                return set(self._pic_a3_ka_fixed_imp)
        return imp_set

    def _pic_a3_oracle_reselect_full(self, forward_batch, layer_id, layer):
        """pic_a3_oracle Route-A: re-select imp over the FULL sequence at check
        layer L, independent of the clipped main-loop rows. Assembles a full-len
        residual stream from the oracle capture (hit + miss-non-last segments via
        store[seg_hash][L]) plus the per-forward query-segment stash, runs
        input_layernorm → project_latent_qk_from_normed → pick_imp. Returns
        (imp_full sorted abs positions, ok). ok=False (→ caller falls back to
        full-length keepalive) if any required capture row is missing."""
        reqs = getattr(forward_batch, "_reqs_ref", None)
        if not reqs or len(reqs) != 1:
            return None, False
        req = reqs[0]
        full_len = int(getattr(req, "pic_a3_full_len", 0))
        store = getattr(self, "_pic_a3_oracle_hidden", None)
        if full_len <= 0 or not store:
            return None, False
        L = int(layer_id)
        device = forward_batch.input_ids.device
        input_ids = forward_batch.input_ids
        _last = req.pic_segments[-1] if getattr(req, "pic_segments", None) else None

        def _bail(reason):
            try:
                from sglang.srt.distributed.parallel_state import (
                    get_tensor_model_parallel_rank,
                )

                if get_tensor_model_parallel_rank() == 0:
                    import logging as _lgb

                    _lgb.getLogger(__name__).warning(
                        f"[PIC-A3-RESELECT-BAIL] layer={L} reason={reason} "
                        f"storeKeys={len(store)} hit={getattr(req,'pic_hit_segments',None)} "
                        f"miss={getattr(req,'pic_miss_segments',None)}"
                    )
            except Exception:
                pass
            return None, False

        rs_full = None
        def _place(s, e, src):
            nonlocal rs_full
            if src is None or int(src.shape[0]) != int(e - s):
                return False
            if rs_full is None:
                rs_full = src.new_empty((full_len, src.shape[-1]))
            rs_full[s:e] = src.to(rs_full.dtype)
            return True

        # hit segments (3-tuple, hash carried) → capture[hash][L]
        for (s, e, seg_hash) in getattr(req, "pic_hit_segments", None) or []:
            per = store.get(seg_hash)
            if not _place(s, e, per.get(L) if per is not None else None):
                return _bail(f"hit_seg({s},{e})_L{L}_perNone={per is None}")
        # miss segments: non-last → capture; last (query) → per-forward stash
        from sglang.srt.pic.hasher import segment_hash

        for (s, e) in getattr(req, "pic_miss_segments", None) or []:
            if _last is not None and (s, e) == _last:
                if not _place(s, e, getattr(forward_batch, "_pic_a3_last_seg_rs", None)):
                    _lr = getattr(forward_batch, "_pic_a3_last_seg_rs", None)
                    return _bail(f"lastseg({s},{e})_rs={None if _lr is None else tuple(_lr.shape)}")
            else:
                per = store.get(segment_hash(input_ids[s:e]))
                if not _place(s, e, per.get(L) if per is not None else None):
                    return _bail(f"miss_seg({s},{e})_L{L}_perNone={per is None}")
        if rs_full is None:
            return _bail("rs_full_None")

        # ── K-dump 观测台: dump the reconstructed FULL-LENGTH residual stream
        # rs_full — what oracle's deep re-selection actually uses at this check
        # layer, assembled from the isolated store — for ALL positions (incl. every
        # doc segment, "全有的"). Compared offline vs full_recompute's true hidden.
        # Gated by SGLANG_PIC_KDUMP_RS=1 (no _ALL needed). bf16, only check layers.
        try:
            import os as _os_rs

            if (
                _os_rs.environ.get("SGLANG_PIC_KDUMP_DIR", "")
                and _os_rs.environ.get("SGLANG_PIC_KDUMP_RS", "0") == "1"
            ):
                from sglang.srt.distributed.parallel_state import (
                    get_tensor_model_parallel_rank as _tprk_rs,
                )

                if _tprk_rs() == 0:
                    _rsd = _os_rs.environ["SGLANG_PIC_KDUMP_DIR"]
                    _os_rs.makedirs(_rsd, exist_ok=True)
                    _tag_rs = _os_rs.environ.get("SGLANG_PIC_KDUMP_TAG", "modeX")
                    torch.save(
                        {
                            "h": rs_full.detach().to(torch.bfloat16).cpu(),
                            "pos": torch.arange(int(full_len)),
                        },
                        f"{_rsd}/{_tag_rs}_RSFULL_L{int(L)}.pt",
                    )
        except Exception:
            pass

        normed = layer.input_layernorm(rs_full)
        if isinstance(normed, tuple):
            normed = normed[0]
        positions_full = torch.arange(full_len, device=device, dtype=torch.long)
        stash = layer.self_attn.project_latent_qk_from_normed(
            normed, positions_full, forward_batch
        )
        imp = self._pic_a3_select_from_stash(forward_batch, stash, layer_id=L)
        imp = self._pic_a3_union_miss(forward_batch, imp)
        imp_set = set(int(x) for x in imp.tolist())
        imp_set = self._pic_a3_apply_imp_diag_env(imp_set, full_len, L, forward_batch)
        imp_full = torch.tensor(sorted(imp_set), dtype=torch.long, device=device)
        forward_batch.pic_a3_imp_indices = imp_full
        forward_batch.pic_a3_q_positions_l2plus_per_req = [imp_full]
        return imp_full, True

    def _pic_a3_rebuild_rowset_for_window(
        self, forward_batch, layer_id, imp_new, h, r, pos, check_layers, num_layers,
        topk=None,
    ):
        """pic_a3_oracle Route-A clip: rebuild the main-loop tensors from the
        previous window's rowset to the new rowset (miss∪imp'). Kept rows are
        gathered from the current h/r; NEW rows (dropped by a prior window but
        re-selected now) are filled from the oracle capture (residual-stream:
        hidden=oracle_rs, residual=0). Also resets req_to_token→l01, re-routes it
        (rewrite), prepops non-imp cached K for the window, updates shape fields,
        sets postchecking + rebuilds DSA metadata. Returns (h_new, r_new, pos_new).
        """
        req = forward_batch._reqs_ref[0]
        full_len = int(req.pic_a3_full_len)
        device = h.device
        L = int(layer_id)
        new_pos_list = [int(x) for x in imp_new.tolist()]
        n_new = len(new_pos_list)
        old_map = {int(p): i for i, p in enumerate(pos.tolist())}

        # pos → (seg_hash, offset) for capture-fill of NEW rows. hit segs carry
        # the hash; the query (last) seg uses the per-forward last-seg stash.
        _last = req.pic_segments[-1] if getattr(req, "pic_segments", None) else None
        store = getattr(self, "_pic_a3_oracle_hidden", None) or {}

        h_new = h.new_empty((n_new, h.shape[-1]))
        r_new = r.new_empty((n_new, r.shape[-1])) if r is not None else None
        _need_cap = []  # (row_i, abs_pos)
        _old_local = []  # per new row: index into the OLD rowset (0 for new rows)
        for i, p in enumerate(new_pos_list):
            oi = old_map.get(p)
            _old_local.append(oi if oi is not None else 0)
            if oi is not None:
                h_new[i] = h[oi]
                if r_new is not None:
                    r_new[i] = r[oi]
            else:
                _need_cap.append((i, p))
        for (i, p) in _need_cap:
            src = None
            for (s, e, seg_hash) in getattr(req, "pic_hit_segments", None) or []:
                if s <= p < e:
                    per = store.get(seg_hash)
                    src = per.get(L) if per is not None else None
                    if src is not None:
                        src = src[p - s]
                    break
            if src is None and _last is not None and _last[0] <= p < _last[1]:
                _lr = getattr(forward_batch, "_pic_a3_last_seg_rs", None)
                src = _lr[p - _last[0]] if _lr is not None else None
            if src is None:
                # last resort: keep zeros (rare; logged by reselect coverage)
                h_new[i] = 0
            else:
                h_new[i] = src.to(h_new.dtype)
            if r_new is not None:
                r_new[i] = 0  # residual stream folded into h_new (oracle_rs)
        pos_new = imp_new.to(pos.dtype).to(device)

        # reset req_to_token to l01 baseline, then re-route for miss∪imp'.
        ridx = int(forward_batch.req_pool_indices[0].item())
        l01 = req.pic_a3_l01_scratch_slots.to(device)
        self.req_to_token_pool.req_to_token[ridx, :full_len] = l01
        self._pic_a3_rewrite_req_to_token_pool_for_l2plus(forward_batch)

        # window end = next check layer (or num_layers); prepop non-imp cached K.
        _next = int(num_layers)
        for cl in check_layers:
            if int(cl) > L:
                _next = int(cl)
                break
        self._pic_a3_prepop_hit_slots_for_l2plus(
            forward_batch,
            layer_start=L + 1,
            layer_end=min(_next + 1, int(num_layers)),
            exclude_positions=set(new_pos_list),
        )

        # shape fields (seq_lens unchanged — K pool full-len, invariant C).
        forward_batch.extend_num_tokens = n_new
        forward_batch.extend_seq_lens = torch.tensor(
            [n_new], dtype=torch.int32, device=forward_batch.seq_lens.device
        )
        forward_batch.extend_seq_lens_cpu = [n_new]
        forward_batch.out_cache_loc = forward_batch.pic_a3_l2plus_out_cache_loc
        forward_batch.pic_public_out_loc = forward_batch.pic_a3_l2plus_pub_out_loc
        forward_batch.reuse_check_state = "postchecking"
        _ab = self.attn_backend
        if hasattr(_ab, "reset_and_init_forward_metadata"):
            _ab.reset_and_init_forward_metadata(forward_batch)
        topk_new = None
        if topk is not None:
            topk_new = topk[
                torch.tensor(_old_local, dtype=torch.long, device=topk.device)
            ]
        return h_new, r_new, pos_new, topk_new

    def _pic_a3_keepalive_setup_miss_publish(self, forward_batch):
        """Route-A/keepalive WARMUP fix: keepalive never calls the l2plus rewrite,
        so req.pic_a3_l2plus_pub_out_loc stays unset and _pic_writeback_mla_kv
        SKIPS publishing first-seen MISS document segments to the public cache —
        the cached SegmentEntry then points at never-written (garbage) public
        slots, so a later request that hits the segment reads garbage K. Build the
        miss publish mapping (under keepalive routing miss==imp→real l01, so fresh
        K lives at l01) so the existing writeback copies l01→public for all layers.
        Idempotent per req; skips the query (last) segment (never cached)."""
        req = forward_batch._reqs_ref[0]
        if getattr(req, "_pic_a3_ka_miss_pub_done", False):
            return
        req._pic_a3_ka_miss_pub_done = True
        full_len = int(getattr(req, "pic_a3_full_len", 0))
        miss_slots = getattr(req, "pic_miss_segment_slots", None)
        l01 = getattr(req, "pic_a3_l01_scratch_slots", None)
        if not miss_slots or l01 is None or full_len <= 0:
            return
        _last = req.pic_segments[-1] if getattr(req, "pic_segments", None) else None
        device = self.req_to_token_pool.req_to_token.device
        l01 = l01.to(device)
        _out, _pub = [], []
        for (s, e), tup in miss_slots.items():
            if _last is not None and (int(s), int(e)) == (int(_last[0]), int(_last[1])):
                continue  # query segment is never cached
            pub = tup[1].to(device)  # (seg_len,) public slots
            for i, pos in enumerate(range(int(s), int(e))):
                _out.append(int(l01[pos].item()))
                _pub.append(int(pub[i].item()))
        if not _out:
            return
        req.pic_a3_l2plus_out_cache_loc = torch.tensor(
            _out, dtype=torch.long, device=device
        )
        req.pic_a3_l2plus_pub_out_loc = torch.tensor(
            _pub, dtype=torch.long, device=device
        )

    def _pic_a3_keepalive_window(
        self,
        forward_batch: "ForwardBatch",
        layer_id: int,
        check_layers: list,
        num_layers: int,
        layer=None,
    ) -> None:
        """Phase B keep-all-alive per-window boundary (accuracy-ceiling probe).

        Called after the forward of each A³ check layer. Re-selects imp from
        THIS layer's FULL-sequence stash (all tokens are alive → true
        multi-layer selection), then sets up KV routing for the window
        [layer_id+1, next_check):
          - out_cache_loc (full_len): imp -> real l01_scratch slot (fresh K
            lands where req_to_token reads); non-imp -> throwaway slot in the
            l2plus_imp pool (fresh K discarded).
          - non-imp hit positions' real slots get delta-RoPE-corrected cached K
            for the window's layers, so attention reads cached for them.
        req_to_token stays stable (all positions -> l01_scratch) and the DSA
        forward metadata (built once, full-length causal) is NOT rebuilt.
        """
        import logging as _lg

        reqs = getattr(forward_batch, "_reqs_ref", None)
        if not reqs or len(reqs) != 1:
            return
        req = reqs[0]
        # Route-A/keepalive warmup fix: publish first-seen miss doc segments to
        # the public cache (keepalive skips the l2plus rewrite → the default
        # writeback never publishes them → cached SegmentEntry = garbage K).
        self._pic_a3_keepalive_setup_miss_publish(forward_batch)
        full_len = int(getattr(req, "pic_a3_full_len", 0))
        if full_len <= 0:
            return
        device = self.req_to_token_pool.req_to_token.device

        # 1. Select imp from this layer's full-sequence stash + union miss.
        # Under pic_a3_oracle (SGLANG_PIC_A3_ORACLE) the hit segments' layer
        # input was replaced with the isolated-warmup oracle hidden BEFORE this
        # layer ran (_pic_a3_oracle_capture_or_inject), so this stash is already
        # oracle-derived (accurate, undrifted Q·K) — no separate swap needed.
        # Phase 0 (SGLANG_PIC_A3_CLIP_CAPTURE): re-select over the FULL sequence
        # via oracle-capture re-projection (Route-A), replacing the clipped-stash
        # select. Routing below is still full-length keepalive (no clip yet) — this
        # isolates/validates the projection chain. reselect_full already applies
        # the diag env and sets pic_a3_imp_indices.
        _imp_full = None
        if layer is not None and envs.SGLANG_PIC_A3_CLIP_CAPTURE.get():
            _imp_full, _ok = self._pic_a3_oracle_reselect_full(
                forward_batch, int(layer_id), layer
            )
            if not _ok:
                _imp_full = None
        if _imp_full is not None:
            imp_set = set(int(x) for x in _imp_full.tolist())
            # Phase-0 diagnostic: compare Route-A reselect vs the clipped-stash
            # imp to validate the re-projection chain (rank 0 only).
            _sbl = getattr(forward_batch, "pic_a3_stash_by_layer", None)
            _st = _sbl.get(int(layer_id)) if _sbl else None
            if _st is not None:
                try:
                    from sglang.srt.distributed.parallel_state import (
                        get_tensor_model_parallel_rank,
                    )

                    _si = self._pic_a3_union_miss(
                        forward_batch,
                        self._pic_a3_select_from_stash(
                            forward_batch, _st, layer_id=int(layer_id)
                        ),
                    )
                    _sset = set(int(x) for x in _si.tolist())
                    _inter = len(imp_set & _sset)
                    _uni = len(imp_set | _sset) or 1
                    if get_tensor_model_parallel_rank() == 0:
                        import logging as _lgov

                        _lgov.getLogger(__name__).warning(
                            f"[PIC-A3-RESELECT] layer={int(layer_id)} "
                            f"reproj={len(imp_set)} stash={len(_sset)} "
                            f"IoU={_inter/_uni:.3f} inter={_inter}"
                        )
                except Exception:
                    pass
        else:
            stash_by_layer = getattr(forward_batch, "pic_a3_stash_by_layer", None)
            stash = stash_by_layer.get(int(layer_id)) if stash_by_layer else None
            if stash is None:
                return
            imp = self._pic_a3_select_from_stash(
                forward_batch, stash, layer_id=int(layer_id)
            )
            imp = self._pic_a3_union_miss(forward_batch, imp)
            imp_set = set(int(x) for x in imp.tolist())
            imp_set = self._pic_a3_apply_imp_diag_env(
                imp_set, int(full_len), int(layer_id), forward_batch
            )
            forward_batch.pic_a3_imp_indices = imp  # for logging / inspection

        # 2. Window end = next check layer (or num_layers).
        _next = int(num_layers)
        for cl in check_layers:
            if int(cl) > int(layer_id):
                _next = int(cl)
                break

        # 3. Build full-length out_cache_loc: imp -> real, non-imp -> throwaway.
        l01 = req.pic_a3_l01_scratch_slots.to(device)          # (full_len,) real slots
        throwaway = req.pic_a3_l2plus_imp_slots_pool.to(device)  # (>=full_len,) keep-alive bump
        assert throwaway.numel() >= full_len, (
            f"keep-alive needs >= full_len throwaway slots; got "
            f"{throwaway.numel()} < {full_len} (SGLANG_PIC_A3_KEEP_ALIVE must be "
            f"set BEFORE pic_alloc so max_imp_len is bumped to full_len)"
        )
        out_loc = l01[:full_len].clone()
        non_imp = [p for p in range(full_len) if p not in imp_set]
        if non_imp:
            _nip = torch.tensor(non_imp, dtype=torch.long, device=device)
            out_loc[_nip] = throwaway[:full_len][_nip]
        forward_batch.out_cache_loc = out_loc
        forward_batch.pic_public_out_loc = torch.full(
            (full_len,), -1, dtype=torch.long, device=device
        )

        # 4. Prepop cached K into NON-imp hit real slots.
        # BUGFIX(2026-07-22 验证): range must INCLUDE the next check layer. The
        # next check layer forwards BEFORE its own keepalive_window runs, so it
        # still uses THIS window's out_cache_loc (non-imp -> throwaway). Without
        # filling its l01 buffer here, buffer(next)[l01_non_imp] holds stale /
        # garbage K -> attention reads garbage from layer 20/40/60 on -> FDT=0.
        _prepop_end = min(int(_next) + 1, int(num_layers))
        self._pic_a3_prepop_hit_slots_for_l2plus(
            forward_batch,
            layer_start=int(layer_id) + 1,
            layer_end=_prepop_end,
            exclude_positions=imp_set,
        )

        _lg.getLogger(__name__).warning(
            f"[PIC-A3-WINDOW] layer={layer_id} next={_next} full_len={full_len} "
            f"imp={len(imp_set)} non_imp={len(non_imp)}"
        )

    def _pic_a3_rewrite_req_to_token_pool_for_l2plus(
        self, forward_batch: "ForwardBatch"
    ) -> None:
        """Rewrite req_to_token_pool between layer 1 and layer 2 so that
        each position's slot pointer matches the layer-2+ K/V layout:
            hit(non-imp) → PICache public slot
            miss         → l2plus_miss_slot
            imp          → l2plus_imp_slot

        Preconditions:
        - _pic_a3_pick_imp has run (forward_batch.pic_a3_imp_indices set)
        - Per-req attributes pic_a3_l2plus_miss_slots / pic_a3_l2plus_imp_slots_pool
          / pic_rope_hit_private_slots populated by pic_alloc

        Also builds:
        - forward_batch.pic_a3_l2plus_row_indices    — clip mask into full_len
        - forward_batch.pic_a3_l2plus_out_cache_loc  — kernel write slot per row
        - forward_batch.pic_a3_l2plus_pub_out_loc    — public writeback slot per row
        """
        # Get the single req for v1
        req_pool_indices = forward_batch.req_pool_indices
        assert req_pool_indices is not None and req_pool_indices.numel() == 1, (
            f"pic_a3 v1 single-req only (got req_pool_indices={req_pool_indices})"
        )
        req_idx = int(req_pool_indices[0].item())

        # Access schedule_batch.reqs via ForwardBatch._reqs_ref (set at
        # ForwardBatch.init_new time — see forward_batch_info.py:749).
        reqs = getattr(forward_batch, "_reqs_ref", None)
        assert reqs is not None and len(reqs) == 1, (
            "pic_a3 hook needs forward_batch._reqs_ref (single req in v1); "
            f"got {reqs}"
        )
        req = reqs[0]

        full_len = int(req.pic_a3_full_len)
        imp_indices = forward_batch.pic_a3_imp_indices
        assert imp_indices is not None

        # Start from the current req_to_token_pool layout (l01_scratch slots
        # at all positions — set by pic_alloc). We only redirect miss and
        # hit-imp positions; hit-non-imp positions KEEP their l01_scratch
        # slot index (same slot used at layer 0-1). Layer 2..N-1 buffers at
        # those slots are populated by _pic_a3_prepop_hit_slots_for_l2plus
        # with delta-RoPE-corrected public KV.
        device = self.req_to_token_pool.req_to_token.device
        new_slots = self.req_to_token_pool.req_to_token[
            req_idx, :full_len
        ].clone()

        # 1. miss positions → l2plus_miss_slot (fresh KV, published to public)
        # l2plus_miss_slice is stashed per-segment in a SEPARATE dict
        # (pic_a3_l2plus_miss_slots_per_seg) to keep pic_miss_segment_slots
        # tuple shape compatible with picache.py:283-286 unpacking.
        _l2plus_miss_per_seg = getattr(
            req, "pic_a3_l2plus_miss_slots_per_seg", {}
        )
        miss_position_set = set()
        for (s, e) in req.pic_miss_segment_slots.keys():
            miss_position_set.update(range(s, e))
            l2plus_miss_slice = _l2plus_miss_per_seg.get((s, e))
            if l2plus_miss_slice is None:
                raise RuntimeError(
                    f"pic_a3 rewrite: missing l2plus_miss slot for seg [{s},{e})"
                )
            for local_i, pos in enumerate(range(s, e)):
                new_slots[pos] = l2plus_miss_slice[local_i].to(device)

        # 2. hit-imp positions → l2plus_imp_slot (fresh KV, NOT published).
        # These are hit-region positions selected as imp; layer 2+ recomputes
        # their KV. The over-allocated pool (max_imp_len size) is consumed
        # in imp_indices iteration order.
        hit_imp_positions = [
            int(p) for p in imp_indices.tolist() if int(p) not in miss_position_set
        ]
        l2plus_imp_pool = req.pic_a3_l2plus_imp_slots_pool
        assert len(hit_imp_positions) <= l2plus_imp_pool.numel(), (
            f"pic_a3 over-alloc insufficient: hit_imp={len(hit_imp_positions)} "
            f"> max_imp_len pool size={l2plus_imp_pool.numel()}"
        )
        for i, pos in enumerate(hit_imp_positions):
            new_slots[pos] = l2plus_imp_pool[i].to(device)

        # Sanity: no -1 remaining (l01_scratch layout has no -1 by construction)
        assert (new_slots >= 0).all(), (
            f"pic_a3 rewrite: {(new_slots < 0).sum().item()} positions unassigned "
            f"(full_len={full_len})"
        )

        # Write rewritten layout into req_to_token_pool
        self.req_to_token_pool.req_to_token[req_idx, :full_len] = new_slots

        # Build l2plus_row_indices, out_cache_loc, pub_out_loc — parallel arrays
        # for the (miss+imp) Q rows (ascending position order matches
        # pic_a3_q_positions_l2plus_per_req).
        _q_pos = imp_indices  # already sorted ascending; contains miss ∪ imp
        n_q = int(_q_pos.numel())

        _out_cache_loc = torch.empty(n_q, dtype=torch.int64, device=device)
        _pub_out_loc = torch.full((n_q,), -1, dtype=torch.int64, device=device)

        # For each q row, look up the slot we just wrote to req_to_token_pool
        _q_pos_dev = _q_pos.to(device)
        _out_cache_loc[:] = new_slots[_q_pos_dev]

        # Miss position writeback slots (from pic_a3_l2plus_miss_pub_slots,
        # aligned with miss position order under pic_alloc). Fill only for
        # miss (non-last) positions.
        # v1 shortcut: only miss positions in the last segment (query region)
        # in the typical test setup — that IS the last segment, so no writeback.
        # Full correctness: iterate the miss segments in order and match up.
        _miss_pub_slots = req.pic_a3_l2plus_miss_pub_slots  # (miss_non_last,)
        if _miss_pub_slots.numel() > 0:
            # Build a position→pub_slot map for miss-non-last positions
            miss_pos_to_pub: dict = {}
            _pub_off = 0
            miss_segs = list(req.pic_miss_segment_slots.keys())
            for _seg_i, (s, e) in enumerate(miss_segs):
                is_last = (s, e) == req.pic_segments[-1]
                if is_last:
                    continue
                seg_len = e - s
                for local_i, pos in enumerate(range(s, e)):
                    miss_pos_to_pub[pos] = int(_miss_pub_slots[_pub_off + local_i].item())
                _pub_off += seg_len
            # Fill _pub_out_loc for miss positions that need writeback
            for row_i, pos in enumerate(_q_pos.tolist()):
                p = int(pos)
                if p in miss_pos_to_pub:
                    _pub_out_loc[row_i] = miss_pos_to_pub[p]

        forward_batch.pic_a3_l2plus_row_indices = _q_pos.to(device)
        forward_batch.pic_a3_l2plus_out_cache_loc = _out_cache_loc
        forward_batch.pic_a3_l2plus_pub_out_loc = _pub_out_loc
        # Also persist the miss→public writeback tensors on the shared req.
        # forward_extend runs the model on an _eager_fb_view COPY of
        # forward_batch (dataclasses.replace under SGLANG_EAGER_INPUT_NO_COPY),
        # so these forward_batch attribute mutations are LOST by the time the
        # post-forward _pic_writeback_mla_kv runs on the original forward_batch.
        # req (reached via _reqs_ref) IS shared with the copy, so stash the
        # tensors here and have the writeback read them back from req.
        req.pic_a3_l2plus_pub_out_loc = _pub_out_loc
        req.pic_a3_l2plus_out_cache_loc = _out_cache_loc

        # Per-Q kstart for layer 2+ segment-isolated attention. Gather from
        # the per-position kstart tensor (built in schedule_batch per PIC
        # segments): hit-segment imp Q gets its segment start, last-segment
        # (query) miss Q gets 0. Same PIC/CacheBlend rule as layer 0-1.
        _kstart_per_pos = getattr(
            forward_batch, "pic_layer_kstart_flat", None
        )
        if _kstart_per_pos is not None:
            # v1 batch_size=1: kstart_per_pos is the flat per-position
            # tensor of the single request (length = full_len).
            _q_pos_dev_i64 = _q_pos.to(_kstart_per_pos.device).long()
            forward_batch.pic_a3_l2plus_kstart_flat = _kstart_per_pos[
                _q_pos_dev_i64
            ].to(dtype=torch.int32, device=device)
        else:
            forward_batch.pic_a3_l2plus_kstart_flat = None

        # ── Diagnostic: dump the final per-layer kv_pool state at every
        # request-slot position (NOT just extend tokens) when
        # SGLANG_PIC_POOL_DUMP_DIR is set. Runs AFTER all forwards +
        # writeback so the pool reflects what future attention would read —
        # for pic modes that includes pub K + delta-RoPE at hit positions,
        # plus K-override at postchecking layers for pic_a3 / pic_cacheblend.
        # Runs for ALL modes (including full_recompute) so
        # test/manual/pic_mode_kv_error_diag.py can do apples-to-apples
        # cross-mode comparison. Best-effort; capture failure must not break
        # the request.
        _dump_dir = os.environ.get("SGLANG_PIC_POOL_DUMP_DIR")
        if _dump_dir and forward_batch.forward_mode.is_extend():
            try:
                from sglang.srt.distributed.parallel_state import (
                    get_tensor_model_parallel_rank,
                )
                _rank = get_tensor_model_parallel_rank()
                # Take the first request only (diagnostic sends bs=1 requests).
                _req_idx = int(forward_batch.req_pool_indices[0].item())
                _seq_len = int(forward_batch.seq_lens[0].item())
                _r2t = self.req_to_token_pool.req_to_token
                _slots = _r2t[_req_idx, :_seq_len].long()  # (seq_len,) kv_pool slot ids
                for layer_id in range(self.start_layer, self.end_layer):
                    _buf = kv_pool.get_key_buffer(layer_id)
                    _lat = _buf[_slots].detach().to(torch.bfloat16).cpu()
                    # MLA key buffer is (T, 1, kv_lora_rank+qk_rope_head_dim);
                    # squeeze the num_kv_heads=1 dim so the .pt is a clean 2D
                    # (T, kv_dim) and pic_mode_kv_error_diag.py can slice
                    # directly. Non-MLA backends (already 2D) fall through.
                    while _lat.ndim > 2:
                        _squeezed = False
                        for _d in range(_lat.ndim):
                            if _lat.shape[_d] == 1:
                                _lat = _lat.squeeze(_d)
                                _squeezed = True
                                break
                        if not _squeezed:
                            break
                    torch.save(
                        _lat,
                        os.path.join(
                            _dump_dir,
                            f"rank{_rank}_layer{layer_id}_latent.pt",
                        ),
                    )
            except Exception as _e:  # noqa: BLE001
                import logging as _l
                _l.getLogger(__name__).warning(
                    "PIC pool dump failed: %s", _e
                )

    def forward_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # In DP Attention, IDLE batches may be padded (batch_size > 0) for MLP
        # sync. Reinit metadata for the padded case so attention kernels see
        # the right batch_size (e.g. DSA Indexer). For the unpadded case
        # (batch_size == 0) explicitly drop any stale forward_metadata left
        # over from the previous forward — without this, attention layers
        # called from the idle path can re-read a prior batch's req_pool
        # indices and trigger SWA mapping use-after-free.
        if forward_batch.batch_size > 0:
            if not self.server_args.enable_pdmux and self.device == "cuda":
                forward_batch = self._eager_fb_view(forward_batch, pp_proxy_tensors)
            self.attn_backend.init_forward_metadata(forward_batch)
        else:
            self.attn_backend.forward_metadata = None

        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        ctx = (
            self.device_timer.wrap(metadata={"category": "idle"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        ctx = (
            self.device_timer.wrap(metadata={"category": "split_prefill"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            ret = self.model.forward_split_prefill(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                (forward_batch.split_index, next_split_index),
            )
        forward_batch.split_index = next_split_index
        return ret

    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: Optional[bool] = None,  # deprecated
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # Deprecated kwarg: pre-planners mark the batch themselves now.
        forward_batch.apply_deprecated_skip_attn_backend_init(skip_attn_backend_init)

        self.forward_pass_id += 1

        # Try msprob debugger
        if self.msprobe_debugger is not None:
            rank_id = (
                self.gpu_id if self.dp_size is not None and self.dp_size > 1 else None
            )
            self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

        # Step span
        step_span_ctx = (
            torch.profiler.record_function(_build_step_span_name(forward_batch))
            if torch.autograd._profiler_enabled()
            else contextlib.nullcontext()
        )

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if not self.is_draft_worker and ((c := self.canary_manager) is not None)
            else contextlib.nullcontext()
        )

        with (
            canary_ctx,
            step_span_ctx,
            get_global_expert_distribution_recorder().with_forward_pass(
                self.forward_pass_id,
                forward_batch,
            ) as recorder_outputs,
        ):
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            if self.enable_elastic_ep:
                output = self._maybe_rebalance_after_rank_fault(
                    output,
                    forward_batch,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
        output.expert_distribution_metrics = recorder_outputs.get("metrics")

        no_copy_to_cpu = not self.server_args.disable_overlap_schedule
        if (experts_capturer := get_global_experts_capturer()) is not None:
            output.routed_experts_output = experts_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if (indexer_capturer := get_global_indexer_capturer()) is not None:
            output.indexer_topk_output = indexer_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        if dumper.may_enable:
            dumper.step()

        if self.msprobe_debugger is not None:
            self.msprobe_debugger.stop()
            self.msprobe_debugger.step()

        if self.server_args.elastic_ep_backend is not None:
            self.maybe_recover_ep_ranks()

        return output

    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # Honor an outer-published context (spec workers wrap each per-step
        # draft forward with the i-th child backend); otherwise publish this
        # runner's own attn_backend for the forward.
        if has_forward_context():
            ctx_mgr = contextlib.nullcontext()
        else:
            ctx_mgr = forward_context(ForwardContext(attn_backend=self.attn_backend))
        with ctx_mgr:
            mode_check = (
                forward_batch.forward_mode.is_cpu_graph
                if self.device == "cpu"
                else forward_batch.forward_mode.is_cuda_graph
            )
            can_run_graph = bool(
                mode_check()
                and self.graph_runner
                and self.graph_runner.can_run(forward_batch)
            )

            # Hisparse coordinator — backends now read it from self.model_runner.
            if (
                forward_batch.forward_mode.is_decode()
                and self.hisparse_coordinator is not None
            ):
                self.hisparse_coordinator.wait_for_pending_backup()
                self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

            # Replay cuda graph if applicable
            if can_run_graph:
                ret = self.graph_runner.replay(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

            # For MLP sync
            if forward_batch.global_num_tokens_cpu is not None:
                forward_batch.prepare_mlp_sync_batch(self)
            else:
                forward_batch.prepare_attn_tp_scatter_input(self)

            # Normalize num_token_non_padded to be local to this attention TP rank if needed.
            # The skip is scoped to DSACPLayerCommunicator-style CP (DSA, MLA): those
            # flavors already feed a zigzag-split rank-local layout whose token count
            # should not be further divided by attn_tp_size. MHA-arch prefill CP
            # (Qwen3/Qwen2 MoE) keeps the attn_tp-replicated layout and wants the
            # adjustment to run — see docs/design/prefill-cp-mla.md §Phase 5.
            if (
                forward_batch.num_token_non_padded is not None
                and forward_batch.global_num_tokens_gpu is not None
                and require_gathered_buffer(self.server_args)
                and not is_dsa_enable_prefill_cp()
                and not is_mla_prefill_cp_enabled()
            ):
                forward_batch.adjust_num_token_non_padded_for_attn_tp(
                    server_args=self.server_args,
                )

            # Hisparse coordinator — backends now read it from self.model_runner.
            if self.hisparse_coordinator is not None:
                self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

            # Forward without cuda graph
            if forward_batch.forward_mode.is_decode():
                ret = self.forward_decode(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            elif forward_batch.forward_mode.is_split_prefill():
                ret = self.forward_split_prefill(
                    forward_batch,
                    reinit_attn_backend=reinit_attn_backend,
                    forward_count=split_forward_count,
                )
            elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
                # PIC: pre-populate hit segment private KV slots before the model
                # forward so that the forward only processes miss segment tokens.
                self._pic_prepopulate_hit_slots(forward_batch)

                # pic_a3 new-path: model needs handles to model_runner (for
                # _pic_a3_pick_imp / _pic_a3_rewrite_req_to_token_pool_for_l2plus)
                # to call from the deepseek_v2 hook. Also expose reqs list via
                # forward_batch._reqs_ref (already set by ForwardBatch.init_new).
                _pic_a3_active_this_fwd = getattr(
                    forward_batch, "pic_a3_new_path", False
                )
                _inner_model = None
                if _pic_a3_active_this_fwd or os.environ.get(
                    "SGLANG_PIC_KDUMP_DIR", ""
                ):
                    # Also expose self when K-dump is on, so the probe can read the
                    # pools for ANY mode (e.g. full_recompute reference dump).
                    _inner_model = getattr(self.model, "model", None)
                    if _inner_model is not None:
                        _inner_model._pic_a3_model_runner = self

                ret, can_run_graph = self.forward_extend(
                    forward_batch,
                    pp_proxy_tensors=pp_proxy_tensors,
                )

                # Clean up the pic_a3 handles after forward (serial forward
                # so no race). Not strictly required — self is a stable
                # reference — but keeps the model instance clean.
                if _inner_model is not None:
                    _inner_model._pic_a3_model_runner = None

                self._pic_writeback_mla_kv(forward_batch)

                # ── Diagnostic dump (fires for ALL extend modes incl.
                # full_recompute, unlike the pic_a3-only dump in
                # _pic_a3_rewrite_req_to_token_pool_for_l2plus). Env-gated by
                # SGLANG_PIC_POOL_DUMP_DIR. Used by scripts/pic_a3_layer01_error.py
                # to do cross-mode K comparison. Best-effort; failure must not
                # break the request.
                #
                # Slot selection:
                # - For pic_a3 at layer 0-1: read from `pic_a3_l01_scratch_flat`
                #   because pic_a3's layer 0-1 fresh forward wrote K/V to those
                #   slots. Post-rewrite req_to_token for miss/imp positions
                #   points at l2plus_miss / l2plus_imp slots which layer 0-1
                #   never touched (would read zeros → misleading err vs
                #   full_recompute). l01_scratch has the actual fresh layer 0-1
                #   K for all full_len positions.
                # - Otherwise: read via req_to_token (post-rewrite for pic_a3
                #   layer 2+ reflects the actual runtime K that future
                #   attention would see).
                _dump_dir_v2 = os.environ.get("SGLANG_PIC_POOL_DUMP_DIR")
                if _dump_dir_v2:
                    try:
                        from sglang.srt.distributed.parallel_state import (
                            get_tensor_model_parallel_rank,
                        )
                        _rank_v2 = get_tensor_model_parallel_rank()
                        _req_idx_v2 = int(forward_batch.req_pool_indices[0].item())
                        _seq_len_v2 = int(forward_batch.seq_lens[0].item())
                        _r2t_v2 = self.req_to_token_pool.req_to_token
                        _slots_r2t = _r2t_v2[_req_idx_v2, :_seq_len_v2].long()
                        # pic_a3 layer 0-1 override slots (may be None for
                        # non-pic_a3 modes)
                        _l01_flat = getattr(
                            forward_batch, "pic_a3_l01_scratch_flat", None
                        )
                        _slots_l01 = None
                        if _l01_flat is not None:
                            _slots_l01 = _l01_flat[:_seq_len_v2].long().to(
                                _slots_r2t.device
                            )
                        _kv_pool_v2 = self.token_to_kv_pool
                        _CHECK_LAYER = 1  # pic_a3 layer 0-1 boundary
                        for _lid_v2 in range(self.start_layer, self.end_layer):
                            # For pic_a3 layer 0-1: use l01_scratch (actual
                            # fresh K); else use req_to_token (post-rewrite).
                            _use_l01 = (
                                _slots_l01 is not None
                                and _lid_v2 <= _CHECK_LAYER
                            )
                            _slots_v2 = _slots_l01 if _use_l01 else _slots_r2t
                            _buf_v2 = _kv_pool_v2.get_key_buffer(_lid_v2)
                            _lat_v2 = (
                                _buf_v2[_slots_v2].detach().to(torch.bfloat16).cpu()
                            )
                            while _lat_v2.ndim > 2:
                                _sq_v2 = False
                                for _d_v2 in range(_lat_v2.ndim):
                                    if _lat_v2.shape[_d_v2] == 1:
                                        _lat_v2 = _lat_v2.squeeze(_d_v2)
                                        _sq_v2 = True
                                        break
                                if not _sq_v2:
                                    break
                            torch.save(
                                _lat_v2,
                                os.path.join(
                                    _dump_dir_v2,
                                    f"rank{_rank_v2}_layer{_lid_v2}_latent.pt",
                                ),
                            )
                    except Exception as _e_v2:  # noqa: BLE001
                        import logging as _l_v2
                        _l_v2.getLogger(__name__).warning(
                            "PIC pool dump (post-writeback) failed: %s", _e_v2
                        )
            elif forward_batch.forward_mode.is_idle():
                ret = self.forward_idle(
                    forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )
            else:
                raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

            if (
                forward_batch.global_num_tokens_cpu is not None
                and self.pp_group.is_last_rank
            ):
                forward_batch.post_forward_mlp_sync_batch(ret)

            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.

        # Calculate logits bias and apply it to next_token_logits.
        sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Release the vocab_mask GPU tensor immediately after it has been applied
        # to the logits. In overlap scheduling, the sampling_info (and its
        # vocab_mask) can be kept alive by the delay_sample_func closure and
        # batch_record_buf until the next iteration, causing a steady VRAM leak
        # when structured output (grammar) is used.
        sampling_info.vocab_mask = None

    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Sample the next tokens
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        self.maybe_update_ngram_token_table(next_token_ids, forward_batch)
        return next_token_ids

    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """
        Compute token_ids_logprobs without performing sampling.

        Optimized path for prefill-only requests that need token_ids_logprobs but don't
        require next token generation. Skips expensive sampling operations
        while still providing requested probability information.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
        """
        if not forward_batch.token_ids_logprobs:
            return

        # Preprocess logits (same as in sample method)
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

    def save_remote_model(self, url: str):
        from sglang.srt.model_loader.loader import RemoteModelLoader

        logger.info(f"Saving model to {url}")
        RemoteModelLoader.save_model(self.model, self.model_config.model_path, url)

    def save_sharded_model(
        self, path: str, pattern: Optional[str] = None, max_size: Optional[int] = None
    ):
        from sglang.srt.model_loader.loader import ShardedStateLoader

        logger.info(
            f"Save sharded model to {path} with pattern {pattern} and max_size {max_size}"
        )
        ShardedStateLoader.save_model(self.model, path, pattern, max_size)

    def check_weights(self, action: str):
        return self._weight_checker.handle(action=action)

    def update_weights_from_ipc(self, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)

    def prealloc_symmetric_memory_pool(self):
        # PyTorch mempools never de-fragment memory in OOM scenarios, so we need to pre-allocate a large chunk of memory to limit fragmentation.
        if (
            self.is_draft_worker
            or not self.server_args.enable_symm_mem
            or envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() <= 0
        ):
            return

        # Memory allocation is tied to a cuda stream, use the forward stream
        with torch.get_device_module(self.device).stream(self.forward_stream):
            logger.info(
                f"Pre-allocating symmetric memory pool with {envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get()} GiB"
            )
            with use_symmetric_memory(get_tp_group()):
                torch.empty(
                    (envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() * 1024 * 1024 * 1024,),
                    dtype=torch.uint8,
                    device=self.device,
                )

    def _maybe_rebalance_after_rank_fault(
        self,
        output: ModelRunnerOutput,
        forward_batch: ForwardBatch,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool,
        split_forward_count: int,
    ) -> ModelRunnerOutput:
        elastic_ep_state = ElasticEPStateManager.instance()
        if elastic_ep_state is not None and not elastic_ep_state.is_active_equal_last():
            elastic_ep_state.snapshot_active_to_last()
            elastic_ep_state.sync_active_to_cpu()
            logging.info("EPLB due to rank faults")
            gen = self.eplb_manager.rebalance()
            while True:
                try:
                    next(gen)
                except StopIteration:
                    break
            output = self._forward_raw(
                forward_batch,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
        return output


def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


def _build_step_span_name(forward_batch: ForwardBatch) -> str:
    """Build a profile-trace span name for one forward step."""
    mode = forward_batch.forward_mode
    bs = forward_batch.batch_size
    if mode == ForwardMode.EXTEND:
        ext_toks = forward_batch.extend_num_tokens or 0
        return f"step[EXTEND bs={bs} toks={ext_toks}]"
    return f"step[{mode.name} bs={bs}]"


@dataclass
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
