import torchvision
import torchvision.datasets.cifar as cifar_mod

from .paths import CIFAR10_MIRROR


def use_fast_mirror() -> None:
    for cls in (
        getattr(torchvision.datasets, "CIFAR10", None),
        getattr(cifar_mod, "CIFAR10", None),
    ):
        if cls is not None:
            cls.url = CIFAR10_MIRROR
