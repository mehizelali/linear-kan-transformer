import glob
import io
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .paths import HF_CIFAR100_REPO


def find_parquet(root: str, train: bool) -> Optional[str]:
    split = "train" if train else "test"
    candidates = [
        os.path.join(root, f"{split}-00000-of-00001.parquet"),
        os.path.join(root, "cifar100", f"{split}-00000-of-00001.parquet"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    hits = sorted(glob.glob(os.path.join(root, "**", f"{split}-*.parquet"), recursive=True))
    return hits[0] if hits else None


def ensure_downloaded(root: str) -> None:
    if find_parquet(root, True) and find_parquet(root, False):
        return
    os.makedirs(root, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_CIFAR100_REPO,
        repo_type="dataset",
        local_dir=root,
        allow_patterns=["cifar100/*.parquet", "*.parquet"],
    )


def read_parquet(path: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "CIFAR-100 parquet loading requires pyarrow. "
            "Install with:  uv pip install 'pyarrow>=14.0'"
        ) from e

    table = pq.read_table(path, columns=["img", "fine_label"])
    records = table.column("img").to_pylist()
    data = np.empty((len(records), 32, 32, 3), dtype=np.uint8)
    for i, rec in enumerate(records):
        with Image.open(io.BytesIO(rec["bytes"])) as im:
            data[i] = np.asarray(im.convert("RGB"), dtype=np.uint8)
    targets = np.asarray(table.column("fine_label").to_numpy(), dtype=np.int64)
    return data, targets


class Cifar100(Dataset):
    def __init__(self, root: str, train: bool = True, transform=None, download: bool = True):
        if download:
            ensure_downloaded(root)
        path = find_parquet(root, train)
        if path is None:
            raise FileNotFoundError(
                f"CIFAR-100 parquet not found under {root}. "
                f"Download {HF_CIFAR100_REPO} or set download=True."
            )
        self.data, self.targets = read_parquet(path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        img = Image.fromarray(self.data[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[index])
