from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm

from .utils import AverageMeter, accuracy

try:
    from torch.amp import GradScaler as _AmpGradScaler

    GradScaler = _AmpGradScaler  # type: ignore
except Exception:  # pragma: no cover
    pass

try:
    from torch.amp import autocast as _amp_autocast

    def _autocast(enabled: bool, dtype: torch.dtype | None = None):
        if dtype is None:
            return _amp_autocast("cuda", enabled=enabled)
        return _amp_autocast("cuda", enabled=enabled, dtype=dtype)

except Exception:  # pragma: no cover
    from torch.cuda.amp import autocast as _cuda_autocast

    def _autocast(enabled: bool, dtype: torch.dtype | None = None):
        return _cuda_autocast(enabled=enabled)


def build_criterion(cfg: dict[str, Any], mixup_active: bool) -> nn.Module:
    if mixup_active:
        return SoftTargetCrossEntropy()
    smoothing = float(cfg.get("smoothing", 0.1))
    if smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    return nn.CrossEntropyLoss()


def build_mixup(cfg: dict[str, Any], num_classes: int) -> Optional[Mixup]:
    mixup_alpha = float(cfg.get("mixup", 0.0))
    cutmix_alpha = float(cfg.get("cutmix", 0.0))
    cutmix_minmax = cfg.get("cutmix_minmax", None)
    if mixup_alpha <= 0 and cutmix_alpha <= 0 and cutmix_minmax is None:
        return None
    return Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        cutmix_minmax=cutmix_minmax,
        prob=float(cfg.get("mixup_prob", 1.0)),
        switch_prob=float(cfg.get("mixup_switch_prob", 0.5)),
        mode=str(cfg.get("mixup_mode", "batch")),
        label_smoothing=float(cfg.get("smoothing", 0.1)),
        num_classes=num_classes,
    )


def _scaler_enabled(loss_scaler: Optional[GradScaler], use_amp: bool) -> bool:
    if not use_amp or loss_scaler is None:
        return False
    return bool(loss_scaler.is_enabled())


def _model_module(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler: Optional[GradScaler],
    mixup_fn: Optional[Mixup] = None,
    model_ema: Optional[ModelEma] = None,
    clip_grad: Optional[float] = None,
    use_amp: bool = True,
    amp_dtype: torch.dtype | None = None,
    show_pbar: bool = True,
    kan_prune_reg: float = 0.0,
) -> dict[str, float]:
    model.train()
    use_scaler = _scaler_enabled(loss_scaler, use_amp)
    do_clip = clip_grad is not None
    mixup_on = mixup_fn is not None

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter() if not mixup_on else None

    data_iter = iter(data_loader)
    copy_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    pbar = tqdm(
        total=len(data_loader),
        desc=f"train {epoch:03d}",
        leave=False,
        disable=not show_pbar,
        dynamic_ncols=True,
    )

    def _pull():
        try:
            imgs, tgts = next(data_iter)
        except StopIteration:
            return None
        if copy_stream is None:
            return imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
        with torch.cuda.stream(copy_stream):
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
        return imgs, tgts

    batch = _pull()
    while batch is not None:
        images, targets = batch
        if copy_stream is not None:
            torch.cuda.current_stream(device).wait_stream(copy_stream)

        nxt = _pull()

        soft = targets.ndim > 1
        hard_targets = None if soft else targets

        optimizer.zero_grad(set_to_none=True)
        with _autocast(enabled=use_amp, dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, targets)
            if kan_prune_reg > 0.0:
                base = _model_module(model)
                if hasattr(base, "kan_pruning_loss"):
                    loss = loss + kan_prune_reg * base.kan_pruning_loss()

        if use_scaler:
            assert loss_scaler is not None
            loss_scaler.scale(loss).backward()
            if do_clip:
                loss_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            loss_scaler.step(optimizer)
            loss_scaler.update()
        else:
            loss.backward()
            if do_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        bs = images.shape[0]
        loss_meter.update(loss.item(), bs)
        if acc1_meter is not None and hard_targets is not None:
            acc1 = accuracy(outputs, hard_targets, topk=(1,))[0]
            acc1_meter.update(acc1, bs)

        if show_pbar:
            pbar.set_postfix(loss=f"{loss_meter.avg:.3f}", refresh=False)
        pbar.update(1)
        batch = nxt

    pbar.close()
    stats = {"loss": loss_meter.avg, "lr": optimizer.param_groups[0]["lr"]}
    if acc1_meter is not None:
        stats["acc1"] = acc1_meter.avg
    return stats


@torch.inference_mode()
def evaluate(
    data_loader,
    model: nn.Module,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype | None = None,
    show_pbar: bool = True,
    epoch: int | None = None,
) -> dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()

    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    data_iter = iter(data_loader)
    copy_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
    desc = f"eval  {epoch:03d}" if epoch is not None else "eval"
    pbar = tqdm(
        total=len(data_loader),
        desc=desc,
        leave=False,
        disable=not show_pbar,
        dynamic_ncols=True,
    )

    def _pull():
        try:
            imgs, tgts = next(data_iter)
        except StopIteration:
            return None
        if copy_stream is None:
            return imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
        with torch.cuda.stream(copy_stream):
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
        return imgs, tgts

    batch = _pull()
    while batch is not None:
        images, targets = batch
        if copy_stream is not None:
            torch.cuda.current_stream(device).wait_stream(copy_stream)
        nxt = _pull()

        with _autocast(enabled=use_amp, dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        bs = images.shape[0]
        loss_meter.update(loss.item(), bs)
        acc1_meter.update(acc1, bs)
        acc5_meter.update(acc5, bs)

        pbar.update(1)
        batch = nxt

    pbar.close()
    _sync_meters(loss_meter, acc1_meter, acc5_meter, device=device)
    return {"loss": loss_meter.avg, "acc1": acc1_meter.avg, "acc5": acc5_meter.avg}


def _sync_meters(*meters, device: torch.device) -> None:
    """All-reduce AverageMeter sum/count so distributed eval is global."""
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    for meter in meters:
        if meter is None:
            continue
        t = torch.tensor([meter.sum, float(meter.count)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        meter.sum = float(t[0].item())
        meter.count = int(t[1].item())
        meter.avg = meter.sum / max(meter.count, 1)
