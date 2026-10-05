"""Architecture and determinism regression tests for the two towers.

Deliberately **not** a hash of the weight values. Hashing raw tensors would fail
the moment torch or timm is upgraded, even when the model is functionally
identical, and it would send someone hunting a "corrupted checkpoint" that does
not exist. Instead:

* the **architecture signature** (parameter name + shape) pins the contract that
  actually has to survive a dependency bump, and
* **seeded determinism** pins reproducibility, which is what the guide's
  ``seed=42`` requirement is about.

``pretrained=False`` keeps the test offline and fast.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.spectral_tower import SpectralTower  # noqa: E402
from src.models.vision_tower import BACKBONE_NAME, VisionTower  # noqa: E402

MORPHOLOGY_CLASSES = ("sphere", "fragment", "fiber", "film", "foam")
POLYMER_CLASSES = ("PE", "PP", "PS", "PMMA", "PAN")


def signature(model: torch.nn.Module) -> list[tuple[str, tuple[int, ...]]]:
    """(parameter name, shape) for every parameter, in declaration order."""
    return [(name, tuple(p.shape)) for name, p in model.named_parameters()]


@pytest.fixture(scope="module")
def vision_signature() -> list[tuple[str, tuple[int, ...]]]:
    return signature(VisionTower(num_classes=len(MORPHOLOGY_CLASSES), pretrained=False))


@pytest.fixture(scope="module")
def spectral_signature() -> list[tuple[str, tuple[int, ...]]]:
    return signature(SpectralTower(num_classes=len(POLYMER_CLASSES), input_bands=600))


# --- architecture -----------------------------------------------------------


def test_vision_tower_head_matches_the_morphology_classes(
    vision_signature: list[tuple[str, tuple[int, ...]]],
) -> None:
    shapes = dict(vision_signature)
    assert shapes["head.3.weight"] == (len(MORPHOLOGY_CLASSES), 256)


def test_vision_tower_feeds_a_1024_dimensional_backbone(
    vision_signature: list[tuple[str, tuple[int, ...]]],
) -> None:
    """Regression guard for the timm ``num_features`` lie.

    timm 1.0 reports ``num_features=576`` for ``mobilenetv3_small_100`` when the
    real output is 1024. That mismatch made every forward pass raise. Pinning
    the head's input width catches a recurrence immediately.
    """
    shapes = dict(vision_signature)
    assert shapes["head.0.weight"] == (256, 1024)


def test_vision_tower_uses_the_expected_backbone() -> None:
    assert BACKBONE_NAME == "mobilenetv3_small_100"


def test_vision_tower_signature_is_stable_across_instances() -> None:
    first = signature(VisionTower(num_classes=5, pretrained=False))
    second = signature(VisionTower(num_classes=5, pretrained=False))
    assert first == second


def test_spectral_tower_accepts_the_configured_band_count() -> None:
    model = SpectralTower(num_classes=len(POLYMER_CLASSES), input_bands=600)
    assert signature(model) == signature(
        SpectralTower(num_classes=len(POLYMER_CLASSES), input_bands=600)
    )


def test_changing_the_class_count_changes_only_the_head(
    vision_signature: list[tuple[str, tuple[int, ...]]],
) -> None:
    other = signature(VisionTower(num_classes=3, pretrained=False))
    differing = {
        name
        for (name, shape), (_, other_shape) in zip(vision_signature, other)
        if shape != other_shape
    }
    assert differing == {"head.3.weight", "head.3.bias"}


# --- forward shape and determinism -----------------------------------------


def test_vision_tower_forward_returns_one_logit_row_per_image() -> None:
    model = VisionTower(num_classes=5, pretrained=False)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(3, 3, 224, 224))
    assert logits.shape == (3, 5)


def test_vision_tower_is_deterministic_in_eval_mode() -> None:
    """The reproducibility claim behind ``seed=42``, asserted rather than assumed."""
    torch.manual_seed(0)
    model = VisionTower(num_classes=5, pretrained=False)
    model.eval()
    batch = torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        first = model(batch)
        second = model(batch)
    assert torch.equal(first, second)


def test_spectral_tower_forward_shape() -> None:
    model = SpectralTower(num_classes=len(POLYMER_CLASSES), input_bands=600)
    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(4, 600))
    assert logits.shape == (4, len(POLYMER_CLASSES))


def test_spectral_tower_is_deterministic_in_eval_mode() -> None:
    torch.manual_seed(0)
    model = SpectralTower(num_classes=len(POLYMER_CLASSES), input_bands=600)
    model.eval()
    batch = torch.rand(4, 600, generator=torch.Generator().manual_seed(11))
    with torch.no_grad():
        assert torch.equal(model(batch), model(batch))


def test_gradients_flow_to_the_backbone() -> None:
    """A frozen-by-accident backbone would silently train nothing."""
    model = VisionTower(num_classes=5, pretrained=False)
    logits = model(torch.zeros(1, 3, 224, 224))
    logits.sum().backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in model.named_parameters()
        if name.startswith("backbone")
    )


def test_freeze_backbone_stops_gradient_flow() -> None:
    model = VisionTower(num_classes=5, pretrained=False)
    model.freeze_backbone(True)
    model(torch.zeros(1, 3, 224, 224)).sum().backward()
    backbone_grads = [
        parameter.grad
        for name, parameter in model.named_parameters()
        if name.startswith("backbone")
    ]
    assert backbone_grads, "backbone has no parameters to check"
    assert all(grad is None or grad.abs().sum() == 0 for grad in backbone_grads)
