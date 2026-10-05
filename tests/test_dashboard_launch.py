"""Regression guard for how the dashboard is *launched*, not just what it does.

``tests/test_dashboard.py`` drives the app through ``AppTest`` inside pytest,
where ``tests/conftest.py`` has already put the repo root on ``sys.path``. That
masks a real failure: ``streamlit run src/output/dashboard.py`` puts only the
script's own folder (``src/output/``) on ``sys.path`` -- never the working
directory -- so the documented launch command died with
``ModuleNotFoundError: No module named 'src'`` while all 108 tests passed.

These tests therefore run the app in a **subprocess** whose ``sys.path`` is
stripped of the repo root, reproducing what Streamlit actually does. If the
``sys.path`` bootstrap in ``dashboard.py`` is ever removed, this file fails.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = REPO_ROOT / "src" / "output" / "dashboard.py"
THEME = REPO_ROOT / ".streamlit" / "config.toml"

# Executed with ``python -c`` from a neutral working directory. Note that
# ``python -c`` puts ``''`` (the cwd) on sys.path[0], so the cwd must NOT be the
# repo root -- otherwise the repo root is importable and the test proves nothing.
LAUNCH_PROBE = """
import os, sys

ROOT = {root!r}
SCRIPT = {script!r}

# Exactly what streamlit/web/bootstrap.py:_fix_sys_path does: add the script's
# own folder, and nothing else. The working directory is deliberately absent.
sys.path.insert(0, os.path.dirname(SCRIPT))
assert ROOT not in [os.path.realpath(p or os.getcwd()) for p in sys.path], (
    "probe is broken: the repo root is already importable"
)

from streamlit.testing.v1 import AppTest

app = AppTest.from_file(SCRIPT, default_timeout=600).run()
failures = [e.value for e in app.exception]
if failures:
    print("APPFAIL:", failures[0][:300])
    raise SystemExit(2)
print("OK", [tab.label for tab in app.tabs], len(app.get("file_uploader")))
"""


def _run_probe(cwd: Path) -> subprocess.CompletedProcess:
    """Execute the dashboard with Streamlit's real import path."""
    env = os.environ.copy()
    # Both would silently satisfy the thing under test, so neither may be
    # inherited: PYTHONPATH makes `src` importable without the bootstrap, and a
    # leaked MP_SPECTRAL_ENABLED changes which tabs the app builds.
    env.pop("PYTHONPATH", None)
    env.pop("MP_SPECTRAL_ENABLED", None)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            LAUNCH_PROBE.format(root=str(REPO_ROOT), script=str(DASHBOARD)),
        ],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )


@pytest.mark.slow
def test_dashboard_boots_the_way_streamlit_launches_it(tmp_path: Path) -> None:
    """`streamlit run src/output/dashboard.py` must work from the repo root."""
    result = _run_probe(tmp_path)
    combined = result.stdout + result.stderr
    assert result.returncode == 0, (
        "the dashboard failed to boot under streamlit's import path.\n"
        f"exit={result.returncode}\n{combined[-2000:]}"
    )
    assert "No module named 'src'" not in combined
    assert "OK" in result.stdout, combined[-2000:]


@pytest.mark.slow
def test_visual_only_mode_is_the_default_under_a_real_launch(tmp_path: Path) -> None:
    """The subprocess must boot with no spectral tab, flag unset."""
    result = _run_probe(tmp_path)
    assert result.returncode == 0, (result.stdout + result.stderr)[-2000:]
    assert "'Image', 'Summary'" in result.stdout
    assert "1" in result.stdout.rsplit("OK", 1)[-1], "expected exactly one uploader"


def test_theme_file_exists_and_is_valid_toml() -> None:
    assert THEME.is_file(), (
        "the committed theme is what makes the dashboard look designed; without "
        "it Streamlit falls back to the stock theme on every machine"
    )
    config = tomllib.loads(THEME.read_text("utf-8"))
    assert "theme" in config
    # Dark only, declared in one flat [theme] table. A per-mode override left
    # behind would let Streamlit serve a palette the injected stylesheet knows
    # nothing about, which shows up as a light page behind dark chrome.
    assert config["theme"]["base"] == "dark"
    for mode in ("light", "dark"):
        assert mode not in config["theme"], f"theme.{mode} override left behind"
    for key in (
        "primaryColor",
        "backgroundColor",
        "secondaryBackgroundColor",
        "textColor",
        "borderColor",
    ):
        value = config["theme"][key]
        assert value.startswith("#") and len(value) == 7, f"{key}={value!r}"


def test_theme_accent_does_not_collide_with_the_box_colours() -> None:
    """The UI accent must not be a hue the annotation already uses.

    A red or blue accent would make a teal-framed control look like part of the
    annotation, so this pins the accent to a hue outside the class palette.
    """
    import colorsys

    from src.output.annotate import CLASS_COLORS

    accent = tomllib.loads(THEME.read_text("utf-8"))["theme"]["primaryColor"]
    accent_rgb = tuple(int(accent[i : i + 2], 16) / 255 for i in (1, 3, 5))
    accent_hue = colorsys.rgb_to_hsv(*accent_rgb)[0]

    for name, color in CLASS_COLORS.items():
        hue = colorsys.rgb_to_hsv(*(channel / 255 for channel in color))[0]
        distance = min(abs(accent_hue - hue), 1 - abs(accent_hue - hue))
        assert distance > 0.08, (
            f"theme accent {accent} is within {distance:.3f} of the "
            f"{name!r} box colour; they will read as the same thing"
        )
