"""Guards on the detection weight paths and the H1 training configs.

These assertions exist because the project has already been bitten by a silent
mismatch: ``inference.WEIGHTS`` was keyed ``"yolov26"`` while ``DEFAULT_MODEL``
was ``"yolo26"``, so ``detect_and_crop`` quietly fell back to the YOLOv8
checkpoint and reported its results as if they came from YOLO26. Nothing raised.
Nothing warned. A wrong-model result is worse than a crash in this project,
because it looks like a valid result.

Two things are therefore pinned here:

1. every weight path the code references agrees, by filename, with what is
   actually on disk;
2. the two H1 configs differ only in ``model`` and ``name``, so H1 compares
   architectures rather than schedules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.detection.compare_yolo import MODEL_FILES
from src.detection.inference import DEFAULT_MODEL, WEIGHTS

H1_CONFIGS = ("configs/yolov8.yaml", "configs/yolo26.yaml")


def test_default_model_is_a_known_key() -> None:
    """The silent-fallback regression: the default must resolve in WEIGHTS."""
    assert DEFAULT_MODEL in WEIGHTS, (
        f"DEFAULT_MODEL={DEFAULT_MODEL!r} is not a key of WEIGHTS "
        f"({sorted(WEIGHTS)}); detect_and_crop would fall back silently."
    )


def test_inference_and_compare_agree_on_filenames() -> None:
    """The dashboard path and the comparison table must read the same files."""
    inference = {Path(p).name for p in WEIGHTS.values()}
    compare = {Path(p).name for p in MODEL_FILES.values()}
    assert inference == compare, f"filenames disagree: {inference} != {compare}"


def test_yolov11_is_gone() -> None:
    """YOLOv11 was dropped; no code path may still reference it."""
    for source, mapping in (
        ("inference.WEIGHTS", WEIGHTS),
        ("compare_yolo.MODEL_FILES", MODEL_FILES),
    ):
        for key in mapping:
            assert "11" not in key, f"{source} still contains a YOLOv11 entry: {key!r}"


@pytest.mark.parametrize("path", sorted(WEIGHTS.values()), ids=sorted(WEIGHTS))
def test_referenced_weights_exist(path: Path) -> None:
    """A referenced checkpoint that is absent must be a loud failure."""
    if not path.is_file():
        pytest.skip(f"{path.name} not present locally (weights/ is gitignored)")
    assert path.stat().st_size > 0, f"{path.name} is a zero-byte placeholder"


def _load_config(name: str) -> dict:
    import yaml

    return yaml.safe_load(
        (Path(__file__).resolve().parents[1] / name).read_text("utf-8")
    )


@pytest.mark.parametrize("name", H1_CONFIGS)
def test_h1_configs_exist_and_are_nano(name: str) -> None:
    """The trained checkpoints are nano variants; configs must say so."""
    config = _load_config(name)
    model = str(config["model"])
    assert model.endswith("n.pt"), (
        f"{name}: model={model!r} — the checkpoint in weights/ is the nano "
        "variant, so naming it small would misreport the comparison."
    )
    assert "s_microplastic" not in str(
        config["name"]
    ), f"{name}: run name {config['name']!r} claims a small model."


def test_h1_configs_differ_only_by_model_and_name() -> None:
    """H1 is only fair if the two arms share a schedule."""
    v8, v26 = (_load_config(n) for n in H1_CONFIGS)
    differing = {k for k in set(v8) | set(v26) if v8.get(k) != v26.get(k)}
    assert differing == {
        "model",
        "name",
    }, f"H1 arms differ in more than the architecture: {sorted(differing - {'model', 'name'})}"
