"""MobileNetV3-Small morphology classifier (Pipeline 1B)."""

from __future__ import annotations

import warnings

import timm
import torch
from torch import nn

#: timm backbone used for the morphology classifier.
BACKBONE_NAME = "mobilenetv3_small_100"
#: Side length the backbone is probed with to measure its pooled feature width.
PROBE_SIZE = 224


class VisionTower(nn.Module):
    """MobileNetV3-Small classifier over particle-morphology classes."""

    def __init__(self, num_classes: int = 5, pretrained: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = timm.create_model(
            BACKBONE_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        # ``backbone.num_features`` is unreliable: on timm 1.0.x
        # ``mobilenetv3_small_100`` advertises 576 while the pooled output is
        # actually 1024, which silently builds a head with the wrong input size
        # and blows up on the first forward pass. Measure instead of trusting.
        self.feat_dim = self._measure_feat_dim()
        declared = getattr(self.backbone, "num_features", None)
        if declared is not None and int(declared) != self.feat_dim:
            warnings.warn(
                f"{BACKBONE_NAME}: backbone.num_features={declared} disagrees "
                f"with measured output {self.feat_dim}; using measured value.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.head = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        if pretrained:
            self.freeze_backbone(freeze=True)

    def _measure_feat_dim(self) -> int:
        """Run one dummy forward pass to learn the pooled feature width."""
        was_training = self.backbone.training
        self.backbone.eval()
        try:
            with torch.no_grad():
                probe = torch.zeros(1, 3, PROBE_SIZE, PROBE_SIZE)
                return int(self.backbone(probe).shape[1])
        finally:
            self.backbone.train(was_training)

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze (``freeze=True``) or unfreeze every backbone parameter."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def unfreeze_last_blocks(self, n: int = 4) -> None:
        """Unfreeze the last ``n`` backbone blocks (Guide Day 5, run 2)."""
        # timm stubs type `blocks` as a union; it is a ModuleList at runtime.
        blocks = list(self.backbone.blocks)  # type: ignore[arg-type]
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
