import pytest

torch = pytest.importorskip("torch")

from src.models.spectral_tower import SpectralTower


@pytest.fixture(scope="module")
def tower() -> SpectralTower:
    model = SpectralTower(num_classes=5, input_bands=600)
    model.eval()
    return model


def test_forward_output_shape(tower: SpectralTower) -> None:
    out = tower(torch.randn(1, 600))
    assert tuple(out.shape) == (1, 5)


def test_forward_batched_shape(tower: SpectralTower) -> None:
    out = tower(torch.randn(4, 1, 600))
    assert tuple(out.shape) == (4, 5)


def test_forward_logits_are_finite(tower: SpectralTower) -> None:
    out = tower(torch.randn(2, 600))
    assert bool(torch.isfinite(out).all())
