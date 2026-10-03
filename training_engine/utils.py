from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int, deterministic: bool = True) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >=1.8
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0).item() * 100.0 / batch_size for k in topk]


def save_checkpoint(state: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def append_log(output_dir: str | Path, stats: dict[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "log.txt").open("a") as f:
        f.write(json.dumps(stats) + "\n")


def cosine_lr(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    lr: float,
    min_lr: float,
    warmup_lr: float,
) -> float:
    if epoch < warmup_epochs:
        new_lr = warmup_lr + (lr - warmup_lr) * epoch / max(warmup_epochs, 1)
    else:
        t = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        new_lr = min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * t))
    for group in optimizer.param_groups:
        group["lr"] = new_lr
    return new_lr


def linear_scaled_lr(base_lr: float, batch_size: int, world_size: int = 1) -> float:
    return sqrt_scaled_lr(base_lr, batch_size, world_size)


def sqrt_scaled_lr(base_lr: float, batch_size: int, world_size: int = 1) -> float:
    return base_lr * math.sqrt(batch_size * world_size / 512.0)


def interpolate_abs_pos_embed(
    pos_embed: torch.Tensor,
    *,
    num_patches: int,
    num_extra: int,
) -> torch.Tensor:
    """Bicubic-resize absolute ``pos_embed`` for a different input resolution.

    LKAT / KATT layout: ``[patch_tokens | register_tokens | optional cls]``.
    Extra tokens are copied unchanged; only the square patch grid is resized.
    Target length is ``num_patches + num_extra`` (do not read FSDP-sharded
    ``model.pos_embed.shape``).
    """
    if not isinstance(pos_embed, torch.Tensor) or pos_embed.ndim != 3:
        raise ValueError(f"pos_embed must be [1, N, C], got {type(pos_embed)} "
                         f"shape={getattr(pos_embed, 'shape', None)}")
    if pos_embed.shape[0] != 1:
        raise ValueError(f"pos_embed batch dim must be 1, got {tuple(pos_embed.shape)}")

    num_patches = int(num_patches)
    num_extra = int(num_extra)
    if num_patches <= 0 or num_extra < 0:
        raise ValueError(f"invalid num_patches={num_patches} num_extra={num_extra}")

    target_len = num_patches + num_extra
    if int(pos_embed.shape[1]) == target_len:
        return pos_embed

    src_tokens = int(pos_embed.shape[1])
    src_patches = src_tokens - num_extra
    if src_patches <= 0:
        raise ValueError(
            f"ckpt pos_embed length {src_tokens} incompatible with num_extra={num_extra}"
        )

    src_gs = int(round(math.sqrt(src_patches)))
    dst_gs = int(round(math.sqrt(num_patches)))
    if src_gs * src_gs != src_patches or dst_gs * dst_gs != num_patches:
        raise ValueError(
            f"non-square patch grids: src_patches={src_patches} dst_patches={num_patches}"
        )

    patch_pos = pos_embed[:, :src_patches, :]
    extra_pos = pos_embed[:, src_patches:, :]
    if extra_pos.shape[1] != num_extra:
        raise ValueError(
            f"extra token count mismatch: ckpt={extra_pos.shape[1]} model={num_extra}"
        )

    # [1, N, C] → [1, C, H, W] → interpolate → [1, N', C]
    patch_pos = patch_pos.reshape(1, src_gs, src_gs, -1).permute(0, 3, 1, 2).float()
    patch_pos = torch.nn.functional.interpolate(
        patch_pos, size=(dst_gs, dst_gs), mode="bicubic", align_corners=False
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, num_patches, -1).to(
        dtype=pos_embed.dtype
    )
    return torch.cat([patch_pos, extra_pos], dim=1)


def abs_pos_embed_target_from_cfg(cfg: dict[str, Any]) -> tuple[int, int] | None:
    """Return ``(num_patches, num_extra)`` for LKAT/KATT abs pos, or None if disabled."""
    if not bool(cfg.get("use_abs_pos", True)):
        return None
    img = int(cfg.get("input_size", 224))
    patch = int(cfg.get("patch_size", 16))
    if img % patch != 0:
        raise ValueError(f"input_size={img} must be divisible by patch_size={patch}")
    num_patches = (img // patch) ** 2
    num_registers = int(cfg.get("num_registers", 4))
    use_cls = bool(cfg.get("use_cls_token", True))
    num_extra = num_registers + (1 if use_cls else 0)
    return num_patches, num_extra

