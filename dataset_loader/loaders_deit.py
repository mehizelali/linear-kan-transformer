"""DeiT-specific data loaders and transforms.

Includes:
- RandomResizedCrop(scale=(0.08, 1.0), interpolation=BICUBIC)
- RepeatAugSampler(num_repeats=3) for distributed training
- Standard RandAugment, Mixup, CutMix, RandomErasing, and LabelSmoothing
"""
from __future__ import annotations

import os
from typing import Any, Optional, Tuple

import torch
import torchvision.transforms as transforms
from timm.data.auto_augment import rand_augment_transform
from timm.data.distributed_sampler import RepeatAugSampler
from timm.data.mixup import Mixup
from torch.utils.data import DataLoader, DistributedSampler

from .collate import MixupCollate
from .loaders import _build_datasets, _cuda_blind_worker_init
from .paths import DATASET_SPECS, NORMALIZATION_STATS, normalize_dataset_name
from .transforms import build_mixup

_DEIT_CFG: dict[str, Any] = {}


def set_deit_config(cfg: dict[str, Any]) -> None:
    global _DEIT_CFG
    _DEIT_CFG = dict(cfg)


def get_deit_config() -> dict[str, Any]:
    return _DEIT_CFG


def build_transforms_deit(
    *,
    img_size: int,
    mean: list[float],
    std: list[float],
    mixup: float,
    cutmix: float,
    num_classes: int,
    rrc_scale: tuple[float, float] = (0.08, 1.0),
) -> tuple[transforms.Compose, transforms.Compose, float, Mixup | None]:
    label_smoothing = 0.1
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=rrc_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            rand_augment_transform("rand-m9-mstd0.5-inc1", {}),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ]
    )
    mixup_fn = build_mixup(
        mixup=mixup,
        cutmix=cutmix,
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(
                int(img_size / 0.875),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform_train, transform_test, label_smoothing, mixup_fn


def get_data_loaders_deit(
    dataset_name: str = "imagenet100",
    batch_size: int = 128,
    num_workers: int = 4,
    img_size: int = 224,
    aug_type: str = "deit",
    data_path: Optional[str] = None,
    mixup: float = 0.8,
    cutmix: float = 1.0,
    **kwargs,
) -> Tuple[DataLoader, DataLoader, int, float, Optional[Mixup]]:
    cfg = get_deit_config()
    repeated_aug = int(cfg.get("repeated_aug", 3))
    rrc_scale = tuple(cfg.get("rrc_scale", (0.08, 1.0)))

    dataset_name = normalize_dataset_name(dataset_name)
    spec = DATASET_SPECS[dataset_name]
    norm_stats = NORMALIZATION_STATS[dataset_name]
    root = data_path or spec["root"]

    transform_train, transform_test, label_smoothing, mixup_fn = build_transforms_deit(
        img_size=img_size,
        mean=norm_stats["mean"],
        std=norm_stats["std"],
        mixup=mixup,
        cutmix=cutmix,
        num_classes=spec["num_classes"],
        rrc_scale=rrc_scale,
    )

    trainset, testset = _build_datasets(dataset_name, root, transform_train, transform_test)

    train_sampler = None
    val_sampler = None
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            if repeated_aug > 0:
                train_sampler = RepeatAugSampler(
                    trainset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=True,
                    num_repeats=repeated_aug,
                )
            else:
                train_sampler = DistributedSampler(trainset, shuffle=True, drop_last=True)
            val_sampler = DistributedSampler(testset, shuffle=False, drop_last=False)
    except Exception:
        train_sampler = None
        val_sampler = None

    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs.update(
            persistent_workers=True,
            prefetch_factor=4,
            multiprocessing_context="spawn",
            worker_init_fn=_cuda_blind_worker_init,
        )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=MixupCollate(mixup_fn) if mixup_fn is not None else None,
        **worker_kwargs,
    )
    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        **worker_kwargs,
    )
    trainloader.sampler_obj = train_sampler  # type: ignore[attr-defined]

    return trainloader, testloader, spec["num_classes"], label_smoothing, mixup_fn
