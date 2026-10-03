import os
from typing import TypedDict

_LOADER_DIR = os.path.dirname(os.path.abspath(__file__))

CIFAR10_ROOT = os.path.join(_LOADER_DIR, "cifar10")
CIFAR100_ROOT = os.path.join(_LOADER_DIR, "cifar100")
IMAGENET100_ROOT = os.path.join(_LOADER_DIR, "imagenet100")

HF_CIFAR100_REPO = "uoft-cs/cifar100"
HF_IMAGENET100_REPO = "clane9/imagenet-100"

CIFAR10_MIRROR = (
    "https://github.com/Digital-Media/cv_data/releases/download/cifar-10/cifar-10-python.tar.gz"
)

NORMALIZATION_STATS = {
    "cifar10": {"mean": [0.4914, 0.4822, 0.4465], "std": [0.2023, 0.1994, 0.2010]},
    "cifar100": {"mean": [0.5071, 0.4865, 0.4409], "std": [0.2673, 0.2564, 0.2762]},
    "imagenet100": {"mean": [0.4850, 0.4560, 0.4060], "std": [0.2290, 0.2240, 0.2250]},
}


class DatasetSpec(TypedDict):
    num_classes: int
    root: str


DATASET_SPECS: dict[str, DatasetSpec] = {
    "cifar10": {"num_classes": 10, "root": CIFAR10_ROOT},
    "cifar100": {"num_classes": 100, "root": CIFAR100_ROOT},
    "imagenet100": {"num_classes": 100, "root": IMAGENET100_ROOT},
}


def normalize_dataset_name(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key in ("imagenet_100", "in100"):
        return "imagenet100"
    if key not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset {name!r}. Choose from: {sorted(DATASET_SPECS)}")
    return key
