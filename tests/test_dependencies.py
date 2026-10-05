"""Guard the dependency declarations against drift.

There are two dependency files and they must not diverge:

* ``pyproject.toml`` is authoritative for local development, resolved into
  ``uv.lock``.
* ``requirements.txt`` is what Colab and the GPU training machine install.

A package added to one and forgotten in the other fails asymmetrically and
confusingly: a teammate's ``uv sync`` works while their Colab run dies with
``ModuleNotFoundError`` halfway through a notebook, or vice versa. These tests
make the drift a local failure instead.

``uv pip check`` is the runtime half of the same concern and is a shell command,
so it is documented in the README rather than asserted here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Distribution name -> import name, where they differ. Only entries whose
#: mismatch would actually cause a wrong comparison are listed.
IMPORT_NAME_OVERRIDES = {
    "pillow": "PIL",
    "scikit-learn": "sklearn",
    "pyyaml": "yaml",
    "opencv-python": "cv2",
    "python-dateutil": "dateutil",
    "scipy": "scipy",
}


def _load_pyproject() -> dict:
    """Parse pyproject.toml with the stdlib TOML reader.

    PyYAML cannot read TOML, despite the superficial resemblance.
    """
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _canonical(name: str) -> str:
    """Normalise a distribution name for comparison.

    PEP 503 says ``Pillow``, ``pillow`` and ``PILLOW`` are the same package, and
    hyphens/underscores/dots are interchangeable. Everything lowercases and
    collapses to hyphens.
    """
    return re.sub(r"[-_.]+", "-", name.strip().lower())


def _requirement_name(requirement: str) -> str:
    """Extract the distribution name from a requirement line.

    Handles ``torch>=2.0.0``, ``pillow>=10.0``, ``numpy`` and
    ``foo[extra]>=1; python_version < '3.12'``.
    """
    line = requirement.split("#", 1)[0].split(";", 1)[0].strip()
    if not line:
        raise ValueError(f"empty requirement: {requirement!r}")
    match = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)", line)
    if not match:
        raise ValueError(f"cannot parse a package name from {requirement!r}")
    return match.group(1)


def _requirement_specifier(requirement: str) -> str:
    """The version constraint of a requirement line, e.g. ``>=2.0.0``."""
    line = requirement.split("#", 1)[0].split(";", 1)[0].strip()
    match = re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[.*\])?(.+)$", line)
    return (match.group(2) or "").strip() if match else ""


@pytest.fixture(scope="module")
def pyproject_requirements() -> dict[str, str]:
    config = _load_pyproject()
    return {
        _canonical(_requirement_name(item)): _requirement_specifier(item)
        for item in config["project"]["dependencies"]
    }


@pytest.fixture(scope="module")
def requirements_txt() -> dict[str, str]:
    entries: dict[str, str] = {}
    for raw in (REPO_ROOT / "requirements.txt").read_text("utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        entries[_canonical(_requirement_name(line))] = _requirement_specifier(line)
    return entries


def test_requirements_txt_matches_pyproject_exactly(
    pyproject_requirements: dict[str, str],
    requirements_txt: dict[str, str],
) -> None:
    missing = sorted(set(pyproject_requirements) - set(requirements_txt))
    extra = sorted(set(requirements_txt) - set(pyproject_requirements))
    assert not missing and not extra, (
        "requirements.txt and pyproject.toml disagree.\n"
        f"  only in pyproject.toml : {missing}\n"
        f"  only in requirements.txt: {extra}"
    )


def test_version_constraints_agree(
    pyproject_requirements: dict[str, str],
    requirements_txt: dict[str, str],
) -> None:
    """The same package must not be pinned to different floors."""
    mismatched = {
        name: (pyproject_requirements[name], requirements_txt[name])
        for name in pyproject_requirements
        if name in requirements_txt
        and pyproject_requirements[name] != requirements_txt[name]
    }
    assert not mismatched, f"version constraints differ: {mismatched}"


def test_requirements_txt_is_not_pinned_exactly(
    requirements_txt: dict[str, str],
) -> None:
    """Colab supplies its own CUDA torch, so ``==`` pins would break the notebook.

    An earlier ``uv export`` produced ``torch==2.14.1+cpu`` with no
    ``--index-url``, which installs a CPU-only build on Colab and silently loses
    the GPU. Loose floors plus a header comment are the deliberate defence.
    """
    exact = {
        name: spec for name, spec in requirements_txt.items() if spec.startswith("==")
    }
    assert not exact, f"exact pins will fight Colab's own CUDA wheels: {exact}"


def test_requirements_txt_documents_the_split() -> None:
    header = (REPO_ROOT / "requirements.txt").read_text("utf-8")
    assert "uv" in header.lower(), "requirements.txt must say when it is used"
    assert "colab" in header.lower()


def test_python_requires_matches_the_tooling_target() -> None:
    """black/ruff/mypy target py310; the floor must not exceed it."""
    floor = _load_pyproject()["project"]["requires-python"]
    assert floor.startswith(
        ">=3.10"
    ), f"requires-python={floor!r} but black/ruff/mypy target py310"


def test_torch_comes_from_the_cpu_index() -> None:
    """Local installs must not pull the ~2.5 GB CUDA build."""
    text = (REPO_ROOT / "pyproject.toml").read_text("utf-8")
    assert "[tool.uv.sources]" in text
    assert "pytorch-cpu" in text, "torch must be sourced from the CPU wheel index"


def test_every_declared_dependency_is_importable(
    pyproject_requirements: dict[str, str],
) -> None:
    """Each declared distribution must actually be installed and importable.

    Catches a stale ``uv.lock`` or a forgotten ``uv sync`` before a teammate
    hits it mid-demo. Distributions whose import name differs are mapped
    explicitly; any unmapped mismatch is reported rather than skipped.
    """
    import importlib

    missing: dict[str, str] = {}
    for name in sorted(pyproject_requirements):
        module = IMPORT_NAME_OVERRIDES.get(name, name.replace("-", "_"))
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001
            # Not just ImportError: a package can install successfully and still
            # fail at import time. grad-cam 1.5.7 does exactly that, raising
            # NameError from its own __init__ while reporting itself installed.
            missing[name] = f"{module}: {type(exc).__name__}: {exc}"
    assert (
        not missing
    ), "declared in pyproject.toml but not importable — run `uv sync`:\n" + "\n".join(
        f"  {name}: {why}" for name, why in missing.items()
    )
