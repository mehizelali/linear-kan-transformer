import torchvision.transforms as transforms
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup


def build_mixup(
    *,
    mixup: float,
    cutmix: float,
    label_smoothing: float,
    num_classes: int,
) -> Mixup | None:
    if mixup <= 0 and cutmix <= 0:
        return None
    return Mixup(
        mixup_alpha=mixup,
        cutmix_alpha=cutmix,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode="batch",
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )


def build_transforms(
    *,
    aug_type: str,
    img_size: int,
    mean: list[float],
    std: list[float],
    mixup: float,
    cutmix: float,
    num_classes: int,
) -> tuple[transforms.Compose, transforms.Compose, float, Mixup | None]:
    label_smoothing = 0.1 if aug_type == "deit" else 0.0
    mixup_fn = None

    if aug_type == "deit":
        transform_train = transforms.Compose(
            [
                transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(img_size, padding=4, padding_mode="reflect"),
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
    else:
        transform_train = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.RandomCrop(img_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.Resize(int(img_size / 0.875), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transform_train, transform_test, label_smoothing, mixup_fn
