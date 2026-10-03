import os
from typing import Optional, Tuple

import torchvision
from timm.data.mixup import Mixup
from torch.utils.data import DataLoader, DistributedSampler

from .cifar10 import use_fast_mirror
from .cifar100 import Cifar100
from .collate import MixupCollate
from .imagenet100 import ImageNet100, imagefolder_ready, load_splits
from .paths import DATASET_SPECS, NORMALIZATION_STATS, normalize_dataset_name
from .transforms import build_transforms


def _build_datasets(
    dataset_name: str,
    root: str,
    transform_train,
    transform_test,
):
    if dataset_name == "cifar10":
        use_fast_mirror()
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test
        )
        return trainset, testset

    if dataset_name == "cifar100":
        trainset = Cifar100(root=root, train=True, download=True, transform=transform_train)
        testset = Cifar100(root=root, train=False, download=True, transform=transform_test)
        return trainset, testset

    if imagefolder_ready(root):
        trainset = torchvision.datasets.ImageFolder(
            root=os.path.join(root, "train"), transform=transform_train
        )
        testset = torchvision.datasets.ImageFolder(
            root=os.path.join(root, "val"), transform=transform_test
        )
        return trainset, testset

    splits = load_splits(root)
    trainset = ImageNet100(splits["train"], transform=transform_train)
    testset = ImageNet100(splits["validation"], transform=transform_test)
    return trainset, testset


def _cuda_blind_worker_init(_worker_id: int) -> None:
    """Prevent DataLoader workers from initializing CUDA (avoids FSDP/NCCL hangs)."""
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def get_data_loaders(
    dataset_name: str = "cifar10",
    batch_size: int = 128,
    num_workers: int = 4,
    img_size: int = 224,
    aug_type: str = "deit",
    data_path: Optional[str] = None,
    mixup: float = 0.8,
    cutmix: float = 1.0,
) -> Tuple[DataLoader, DataLoader, int, float, Optional[Mixup]]:
    dataset_name = normalize_dataset_name(dataset_name)
    spec = DATASET_SPECS[dataset_name]
    norm_stats = NORMALIZATION_STATS[dataset_name]
    root = data_path or spec["root"]

    transform_train, transform_test, label_smoothing, mixup_fn = build_transforms(
        aug_type=aug_type,
        img_size=img_size,
        mean=norm_stats["mean"],
        std=norm_stats["std"],
        mixup=mixup,
        cutmix=cutmix,
        num_classes=spec["num_classes"],
    )

    trainset, testset = _build_datasets(dataset_name, root, transform_train, transform_test)

    train_sampler = None
    val_sampler = None
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            train_sampler = DistributedSampler(trainset, shuffle=True, drop_last=True)
            val_sampler = DistributedSampler(testset, shuffle=False, drop_last=False)
    except Exception:
        train_sampler = None
        val_sampler = None

    # spawn + hide CUDA in workers: fork/forkserver after cuda init (or torch import
    # opening /dev/nvidia*) pegs util at 100% and hangs FSDP/NCCL after step 1.
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
