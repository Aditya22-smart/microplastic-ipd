"""Shape + freeze-helper tests for the MobileNetV3 VisionTower (Task 1).

The module is skipped automatically (``pytest.importorskip``) when torch/timm
are unavailable — e.g. the CI quality job only installs black/ruff/pytest, so
it reports a clean skip there while running for real on machines with torch.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from src.models.vision_tower import VisionTower


@pytest.fixture(scope="module")
def tower() -> VisionTower:
    model = VisionTower(num_classes=5, pretrained=False)
    model.eval()
    return model


def test_forward_output_shape(tower: VisionTower) -> None:
    out = tower(torch.randn(1, 3, 224, 224))
    assert tuple(out.shape) == (1, 5)


def test_forward_batched_shape(tower: VisionTower) -> None:
    out = tower(torch.randn(4, 3, 224, 224))
    assert tuple(out.shape) == (4, 5)


def test_forward_logits_are_finite(tower: VisionTower) -> None:
    out = tower(torch.randn(2, 3, 224, 224))
    assert bool(torch.isfinite(out).all())


def test_freeze_backbone(tower: VisionTower) -> None:
    tower.freeze_backbone(freeze=True)
    assert not any(p.requires_grad for p in tower.backbone.parameters())
    assert all(p.requires_grad for p in tower.head.parameters())


def test_unfreeze_last_blocks(tower: VisionTower) -> None:
    tower.freeze_backbone(freeze=True)
    tower.unfreeze_last_blocks(n=4)
    blocks = list(tower.backbone.blocks)
    assert all(
        any(p.requires_grad for p in block.parameters()) for block in blocks[-4:]
    )
    assert not any(p.requires_grad for p in blocks[0].parameters())


def test_unfreeze_all(tower: VisionTower) -> None:
    tower.freeze_backbone(freeze=True)
    tower.unfreeze_all()
    assert all(p.requires_grad for p in tower.backbone.parameters())
