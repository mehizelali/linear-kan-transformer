from __future__ import annotations

import functools
import os
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn import Module


SHARDING = {
    "full_shard": ShardingStrategy.FULL_SHARD,
    "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
    "no_shard": ShardingStrategy.NO_SHARD,
    "hybrid_shard": getattr(ShardingStrategy, "HYBRID_SHARD", ShardingStrategy.FULL_SHARD),
}


def _fsdp_block_types(model: Module | None = None) -> set[type]:
    """Transformer block classes eligible for per-block FSDP wrap / checkpointing.

    Prefer types already present on ``model`` so unused backends (e.g. ViG/fla)
    are not imported into the training process.
    """
    types: set[type] = set()
    if model is not None:
        for m in model.modules():
            name = type(m).__name__
            if name.endswith("Block") or name in ("MixerBlock", "SwinTransformerBlock"):
                types.add(type(m))
        if types:
            return types

    from models.KATT import KATTBlock
    from models.LKAT import LKATBlock
    from models.vit5 import Block as ViT5Block

    types = {KATTBlock, LKATBlock, ViT5Block}
    # timm baselines (optional imports)
    try:
        from timm.models.vision_transformer import Block as TimmVitBlock

        types.add(TimmVitBlock)
    except Exception:
        pass
    try:
        from timm.models.swin_transformer import SwinTransformerBlock

        types.add(SwinTransformerBlock)
    except Exception:
        pass
    try:
        from timm.models.mlp_mixer import MixerBlock

        types.add(MixerBlock)
    except Exception:
        pass
    try:
        from timm.models.pvt_v2 import Block as PvtBlock

        types.add(PvtBlock)
    except Exception:
        pass
    return types


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if not is_dist_avail_and_initialized():
        return
    # device_id was set at init_process_group; plain barrier is fine after that
    dist.barrier()


def init_distributed(cfg: dict[str, Any]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for FSDP training.")

    env_world = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    num_gpus = int(cfg.get("num_gpus", env_world))
    cfg["num_gpus"] = max(num_gpus, env_world)
    cfg["world_size"] = env_world
    cfg["rank"] = env_rank
    cfg["local_rank"] = local_rank
    cfg["device"] = "cuda"
    cfg["distributed"] = env_world > 1 or bool(cfg.get("fsdp", True))

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if not dist.is_initialized():
        backend = str(cfg.get("dist_backend", "nccl"))
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(cfg.get("master_port", 29500))
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(max(env_world, 1))
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(env_rank)
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = str(local_rank)

        dist.init_process_group(
            backend=backend,
            init_method=cfg.get("dist_url", "env://"),
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"]),
            device_id=device,  # silences NCCL / barrier device warnings
        )
        env_world = dist.get_world_size()
        env_rank = dist.get_rank()
        cfg["world_size"] = env_world
        cfg["rank"] = env_rank

    barrier()
    return {"device": device, "local_rank": local_rank, "world_size": env_world, "rank": env_rank}


def wrap_fsdp(model: Module, cfg: dict[str, Any], local_rank: int) -> Module:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: training on CPU is not supported.")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    model = model.to(device)

    # Single-GPU: plain CUDA model (AMP via autocast). Multi-GPU: FSDP.
    use_fsdp = bool(cfg.get("fsdp", True)) and get_world_size() > 1
    if not use_fsdp:
        if next(model.parameters()).device.type != "cuda":
            raise RuntimeError("Model failed to move to CUDA.")
        # Still allow activation checkpointing on single GPU for memory.
        if bool(cfg.get("activation_checkpointing", False)):
            block_types = _fsdp_block_types(model)
            apply_activation_checkpointing(
                model,
                checkpoint_wrapper_fn=functools.partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                ),
                check_fn=lambda m: isinstance(m, tuple(block_types)),
            )
        return model

    block_types = _fsdp_block_types(model)

    # Checkpoint before FSDP wrap so each block is a checkpoint + shard unit.
    use_act_ckpt = bool(cfg.get("activation_checkpointing", False))
    if use_act_ckpt:
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            ),
            check_fn=lambda m: isinstance(m, tuple(block_types)),
        )

    auto_wrap_policy = None
    if bool(cfg.get("fsdp_auto_wrap_blocks", True)):
        # After AC, blocks are CheckpointWrapper; otherwise wrap the block classes.
        wrap_cls = {CheckpointWrapper} if use_act_ckpt else block_types
        auto_wrap_policy = ModuleWrapPolicy(wrap_cls)

    shard_name = str(cfg.get("fsdp_sharding_strategy", "full_shard")).lower()
    sharding_strategy = SHARDING.get(shard_name, ShardingStrategy.FULL_SHARD)
    mp = None
    if bool(cfg.get("amp", True)):
        dtype = (
            torch.bfloat16
            if str(cfg.get("amp_dtype", "fp16")).lower() in ("bf16", "bfloat16")
            else torch.float16
        )
        mp = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=sharding_strategy,
        device_id=local_rank,
        mixed_precision=mp,
        use_orig_params=bool(cfg.get("fsdp_use_orig_params", True)),
        sync_module_states=bool(cfg.get("fsdp_sync_module_states", True)),
        limit_all_gathers=bool(cfg.get("fsdp_limit_all_gathers", True)),
    )


@torch.no_grad()
def fsdp_state_dict(model: Module) -> dict[str, Any]:
    if not isinstance(model, FSDP):
        return model.state_dict()
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    return get_model_state_dict(model, options=opts)


def fsdp_load_state_dict(model: Module, state_dict: dict[str, Any]) -> None:
    if not isinstance(model, FSDP):
        model.load_state_dict(state_dict)
        return
    opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
    set_model_state_dict(model, state_dict, options=opts)


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        barrier()
        dist.destroy_process_group()
