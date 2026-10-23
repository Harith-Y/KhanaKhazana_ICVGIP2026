import torchvision.transforms as T

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _norm(cfg):
    """Normalisation stats; defaults to ImageNet, override via augmentation.mean/std (PE-Core uses 0.5/0.5)."""
    aug = cfg["augmentation"]
    return T.Normalize(aug.get("mean", _IMAGENET_MEAN), aug.get("std", _IMAGENET_STD))


def train_transforms(cfg):
    aug = cfg["augmentation"]
    cj  = aug["color_jitter"]
    scale = tuple(aug["random_resized_crop_scale"])
    size  = aug["image_size"]
    return T.Compose([
        T.RandomResizedCrop(size, scale=scale),
        T.RandomHorizontalFlip(),
        T.ColorJitter(
            brightness=cj["brightness"],
            contrast=cj["contrast"],
            saturation=cj["saturation"],
            hue=cj["hue"],
        ),
        T.RandomRotation(15),
        T.ToTensor(),
        _norm(cfg),
        T.RandomErasing(
            p=aug["random_erasing_prob"],
            scale=tuple(aug["random_erasing_scale"]),
        ),
    ])


def val_transforms(cfg):
    size = cfg["augmentation"]["image_size"]
    return T.Compose([
        T.Resize(int(size * 256 / 224)),
        T.CenterCrop(size),
        T.ToTensor(),
        _norm(cfg),
    ])
