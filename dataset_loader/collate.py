from timm.data.mixup import Mixup
from torch.utils.data import default_collate


class MixupCollate:
    """Top-level collate so DataLoader workers can pickle under Python 3.14 forkserver."""

    def __init__(self, mixup_fn: Mixup):
        self.mixup_fn = mixup_fn

    def __call__(self, batch):
        images, targets = default_collate(batch)
        return self.mixup_fn(images, targets)
