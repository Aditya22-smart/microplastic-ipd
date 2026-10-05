"""Shared pytest fixtures.

Also makes the ``src`` package importable from pytest (run from anywhere).
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DASHBOARD = REPO_ROOT / "src" / "output" / "dashboard.py"


@pytest.fixture(scope="session")
def weights_available() -> bool:
    """Whether the detection and morphology checkpoints are on disk.

    The dashboard tests below need real weights to produce records. Rather than
    silently passing with nothing asserted, they skip loudly when the
    checkpoints are absent — which is the state of a fresh clone or a CI runner.
    """
    from src.detection.inference import WEIGHTS
    from src.models.predict_morphology import DEFAULT_WEIGHTS

    needed = [*WEIGHTS.values(), DEFAULT_WEIGHTS]
    return all(path.is_file() for path in needed)


@pytest.fixture(scope="session")
def speck_image() -> bytes:
    """A synthetic microplastic-like field: dark specks on a pale background.

    Generated rather than committed so the tests carry no image asset, and
    deterministic (fixed seed) so detection counts are reproducible.
    """
    import io

    import numpy as np
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (640, 480), (238, 240, 235))
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(0)
    for _ in range(14):
        cx, cy = int(rng.integers(40, 600)), int(rng.integers(40, 440))
        radius = int(rng.integers(5, 13))
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(
                int(rng.integers(30, 90)),
                int(rng.integers(40, 80)),
                int(rng.integers(70, 120)),
            ),
        )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()
