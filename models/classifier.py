import torch
import torch.nn as nn
import torchvision.models as tvm


_RESNET_BACKBONES = {
    "resnet18":  (tvm.resnet18,  tvm.ResNet18_Weights.IMAGENET1K_V1,  512),
    "resnet50":  (tvm.resnet50,  tvm.ResNet50_Weights.IMAGENET1K_V2,  2048),
    "resnet101": (tvm.resnet101, tvm.ResNet101_Weights.IMAGENET1K_V2, 2048),
}

_ALL_BACKBONES = list(_RESNET_BACKBONES) + [
    "efficientnet_b3", "efficientnet_b4", "efficientnet_v2_s",
    "convnext_tiny",
    "mobilenet_v2", "mobilenet_v3_large",
    "densenet121",
    "vit_s_16", "vit_b_16",
    "dinov3_s16", "dinov3_b16", "pe_core_b16", "pe_core_l14",
]

# Frozen self-supervised / vision-language encoders used as linear probes.
# name -> (kind, hub id / PE config name)
_PROBE_ENCODERS = {
    "dinov3_s16":  ("dinov3", "facebook/dinov3-vits16-pretrain-lvd1689m"),
    "dinov3_b16":  ("dinov3", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
    "pe_core_b16": ("pe",     "PE-Core-B16-224"),
    "pe_core_l14": ("pe",     "PE-Core-L14-336"),
}


class _DinoV3Encoder(nn.Module):
    """HF DINOv3: returns the layer-normed CLS token (pooler_output)."""
    def __init__(self, hub_id: str, pretrained: bool):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if pretrained:
            self.m = AutoModel.from_pretrained(hub_id)
        else:  # weights come from the checkpoint being loaded
            self.m = AutoModel.from_config(AutoConfig.from_pretrained(hub_id))
        self.feat_dim = self.m.config.hidden_size

    def forward(self, x):
        return self.m(pixel_values=x).pooler_output


class _PEEncoder(nn.Module):
    """Perception Encoder (PE-Core): image tower of the CLIP model, projected embedding."""
    def __init__(self, config_name: str, pretrained: bool):
        super().__init__()
        import core.vision_encoder.pe as pe
        clip = pe.CLIP.from_config(config_name, pretrained=pretrained)
        self.visual = clip.visual          # drop the text tower (keeps checkpoints small)
        self.feat_dim = self.visual.output_dim

    def forward(self, x):
        return self.visual(x)              # same as CLIP.encode_image(x)


class _FrozenProbe(nn.Module):
    """Frozen encoder + one trainable linear head (linear probe)."""
    def __init__(self, encoder: nn.Module, num_classes: int):
        super().__init__()
        self.encoder = encoder
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.head = nn.Linear(encoder.feat_dim, num_classes)

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()   # frozen encoder stays deterministic
        return self

    def forward(self, x):
        with torch.no_grad():
            f = self.encoder(x)
        return self.head(f.float())


def build_model(
    backbone: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    if backbone not in _ALL_BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {_ALL_BACKBONES}")
    if backbone in _PROBE_ENCODERS:   # always a frozen linear probe
        kind, ref = _PROBE_ENCODERS[backbone]
        enc = _DinoV3Encoder(ref, pretrained) if kind == "dinov3" else _PEEncoder(ref, pretrained)
        return _FrozenProbe(enc, num_classes)
    if freeze_backbone and backbone not in ("vit_s_16", "vit_b_16"):
        raise ValueError(f"freeze_backbone is only supported for ViT backbones, got backbone='{backbone}'")

    # ── ResNet family ──────────────────────────────────────────────────────────
    if backbone in _RESNET_BACKBONES:
        factory, weights, feat_dim = _RESNET_BACKBONES[backbone]
        model = factory(weights=weights if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(feat_dim, num_classes))
        return model

    # ── EfficientNet-B3 (torchvision) ──────────────────────────────────────────
    if backbone == "efficientnet_b3":
        w = tvm.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b3(weights=w)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1536, num_classes),
        )
        return model

    # ── EfficientNet-B4 (torchvision) ─────────────────────────────────────────
    if backbone == "efficientnet_b4":
        w = tvm.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_b4(weights=w)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1792, num_classes),
        )
        return model

    # ── EfficientNet-V2-S (torchvision) ───────────────────────────────────────
    if backbone == "efficientnet_v2_s":
        w = tvm.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.efficientnet_v2_s(weights=w)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),
        )
        return model

    # ── MobileNet-V2 (torchvision) ────────────────────────────────────────────
    if backbone == "mobilenet_v2":
        w = tvm.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.mobilenet_v2(weights=w)
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(1280, num_classes))
        return model

    # ── MobileNet-V3-Large (torchvision) ──────────────────────────────────────
    if backbone == "mobilenet_v3_large":
        w = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = tvm.mobilenet_v3_large(weights=w)
        model.classifier[-1] = nn.Linear(1280, num_classes)
        model.classifier[-2] = nn.Dropout(p=dropout)
        return model

    # ── ConvNeXt-Tiny (torchvision) ───────────────────────────────────────────
    if backbone == "convnext_tiny":
        w = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.convnext_tiny(weights=w)
        # Keep LayerNorm2d + Flatten from pretrained classifier, replace only head
        model.classifier[-1] = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(768, num_classes))
        return model

    # ── DenseNet-121 (torchvision) ────────────────────────────────────────────
    if backbone == "densenet121":
        w = tvm.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.densenet121(weights=w)
        model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(1024, num_classes))
        return model

    # ── ViT-B/16 (torchvision) ────────────────────────────────────────────────
    if backbone == "vit_b_16":
        w = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        model = tvm.vit_b_16(weights=w)
        model.heads = nn.Linear(768, num_classes)
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("heads"):
                    param.requires_grad = False
        return model

    # ── ViT-S/16 (timm) ───────────────────────────────────────────────────────
    if backbone == "vit_s_16":
        import timm
        model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
        if freeze_backbone:
            for name, param in model.named_parameters():
                if not name.startswith("head"):
                    param.requires_grad = False
        return model
