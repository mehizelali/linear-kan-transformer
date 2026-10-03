from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn


def _add_weight_decay(model: nn.Module, weight_decay: float, skip_list: Iterable[str] = ()):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": no_decay, "weight_decay": 0.0},
        {"params": decay, "weight_decay": weight_decay},
    ]


def create_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    opt_name = str(cfg.get("opt", "adamw")).lower()
    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 0.0))
    eps = float(cfg.get("opt_eps", 1e-8))
    betas = cfg.get("opt_betas", None)
    if betas is None:
        betas = (0.9, 0.999)

    param_groups = _add_weight_decay(model, weight_decay)

    if opt_name == "adamw":
        kwargs = dict(lr=lr, eps=eps, betas=tuple(betas))
        if bool(cfg.get("fused_optim", False)) and torch.cuda.is_available():
            try:
                return torch.optim.AdamW(param_groups, fused=True, **kwargs)
            except (TypeError, RuntimeError):
                pass
        return torch.optim.AdamW(param_groups, **kwargs)
    if opt_name == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=float(cfg.get("momentum", 0.9)),
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {opt_name}")
