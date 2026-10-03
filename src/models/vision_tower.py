"""MobileNetV3-Small morphology classifier (Pipeline 1B, Rohan's lane).

timm ``mobilenetv3_small_100`` feature extractor (ImageNet-pretrained,
``num_classes=0``, ``global_pool='avg'``) with a projection head::

    backbone -> [B, feat_dim] -> Linear(feat_dim->256) -> LayerNorm -> ReLU
                              -> Linear(256->5) -> [B, 5]

Classes (AGENT.md 3.1): sphere / fragment / fiber / film / foam.

FEATURE DIMENSION NOTE
``mobilenetv3_small_100`` keeps its 1x1 ``conv_head`` (576 -> 1024 channels)
even with ``num_classes=0``, so the backbone emits 1024 features. The head is
sized from ``backbone.num_features`` at runtime via ``self.feat_dim`` — never
hardcode this value (an empirical fallback guards against timm API drift).
"""

from __future__ import annotations

import timm
import torch
from torch import nn


class VisionTower(nn.Module):
    """MobileNetV3-Small classifier over particle-morphology classes."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = getattr(self.backbone, "num_features", None)
        if feat_dim is None:  # empirical fallback — robust to timm API drift
            with torch.no_grad():
                feat_dim = self.backbone(torch.zeros(1, 3, 224, 224)).shape[1]
        self.feat_dim = int(feat_dim)
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        if pretrained:
            self.freeze_backbone(freeze=True)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze (``freeze=True``) or unfreeze every backbone parameter."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def unfreeze_last_blocks(self, n: int = 4) -> None:
        """Unfreeze the last ``n`` backbone blocks (Guide Day 5, run 2)."""
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

    def unfreeze_all(self) -> None:
        """Unfreeze the whole network (Guide Day 6, run 3)."""
        self.freeze_backbone(freeze=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``[B, num_classes]`` raw morphology logits."""
        feats = self.backbone(x)  # [B, feat_dim]
        return self.head(feats)  # [B, num_classes]
