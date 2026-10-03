#!/usr/bin/env python3
"""ImageNet-100 from clane9/imagenet-100 (Hugging Face parquet).

Training auto-downloads on first use. Prefetch only:

    python -m dataset_loader.imagenet100
"""

from __future__ import annotations

import glob
import io
import os
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from .paths import HF_IMAGENET100_REPO, IMAGENET100_ROOT


def parquet_files(root: str, split: str) -> list[str]:
    return sorted(glob.glob(os.path.join(root, "**", f"{split}-*.parquet"), recursive=True))


def is_downloaded(root: str = IMAGENET100_ROOT) -> bool:
    return bool(parquet_files(root, "train") and parquet_files(root, "validation"))


def imagefolder_ready(root: str) -> bool:
    train = os.path.join(root, "train")
    val = os.path.join(root, "val")
    return os.path.isdir(train) and os.path.isdir(val) and any(os.scandir(train)) and any(os.scandir(val))


def download(root: str = IMAGENET100_ROOT) -> str:
    if is_downloaded(root):
        print(f"[imagenet100] already present at {root}")
        return root
    os.makedirs(root, exist_ok=True)
    from huggingface_hub import snapshot_download

    print(f"[imagenet100] downloading {HF_IMAGENET100_REPO} -> {root} (~8.4 GB)")
    snapshot_download(
        repo_id=HF_IMAGENET100_REPO,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=["data/*.parquet", "*.parquet"],
    )
    if not is_downloaded(root):
        raise FileNotFoundError(f"Download finished but parquet splits missing under {root}")
    n_train = len(parquet_files(root, "train"))
    n_val = len(parquet_files(root, "validation"))
    print(f"[imagenet100] ready ({n_train} train shards, {n_val} val shards)")
    return root


def load_splits(root: str = IMAGENET100_ROOT) -> Any:
    download(root)
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "ImageNet-100 parquet loading requires the Hugging Face datasets package. "
            "Install with:  uv pip install datasets"
        ) from e
    return load_dataset(
        "parquet",
        data_files={
            "train": parquet_files(root, "train"),
            "validation": parquet_files(root, "validation"),
        },
    )


def to_pil_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and image.get("bytes") is not None:
        with Image.open(io.BytesIO(image["bytes"])) as im:
            return im.convert("RGB")
    return Image.open(image).convert("RGB")


class ImageNet100(Dataset):
    def __init__(self, split, transform=None):
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, index: int):
        row = self.split[int(index)]
        img = to_pil_rgb(row["image"])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])


if __name__ == "__main__":
    download()
