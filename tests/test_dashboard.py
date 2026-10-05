"""Dashboard tests, driven headlessly through ``streamlit.testing.v1.AppTest``.

The dashboard is the integration point for the whole project: it is where the
two independent branches meet, and where the ``MP_SPECTRAL_ENABLED`` flag decides
what the user can even do. A silent Streamlit exception leaves a blank page and
no traceback, so these tests assert on the rendered tree, not just on "it ran".

Every test that uploads an image needs real checkpoints and is skipped loudly
when they are absent (a fresh clone, or a CI runner with no weights) rather than
passing with nothing asserted.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import SPECTRAL_ENABLED_ENV
from src.models.full_pipeline import RECORD_KEYS

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = REPO_ROOT / "src" / "output" / "dashboard.py"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(autouse=True)
def _restore_flag_env():
    """Undo the ``os.environ`` writes that :func:`_run` performs.

    ``_run`` mutates the real process environment, so without this a test that
    enables the spectral branch leaves it enabled for every later test -- and for
    any subprocess they spawn. That produced a green dashboard suite next to a
    "default mode is visual only" test that failed depending on file order.
    """
    before = os.environ.get(SPECTRAL_ENABLED_ENV)
    yield
    if before is None:
        os.environ.pop(SPECTRAL_ENABLED_ENV, None)
    else:
        os.environ[SPECTRAL_ENABLED_ENV] = before


def _run(env: str | None = None) -> AppTest:
    """Boot the dashboard, optionally seeding the feature flag."""
    import os

    if env is None:
        os.environ.pop(SPECTRAL_ENABLED_ENV, None)
    else:
        os.environ[SPECTRAL_ENABLED_ENV] = env
    return AppTest.from_file(str(DASHBOARD), default_timeout=300).run()


def _upload_speck(at: AppTest, speck_image: bytes) -> AppTest:
    return (
        at.get("file_uploader")[0].upload("speck.png", speck_image, "image/png").run()
    )


# --- boot / feature flag -----------------------------------------------------


def test_visual_only_mode_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SPECTRAL_ENABLED_ENV, raising=False)
    at = _run()
    assert [tab.label for tab in at.tabs] == ["Image", "Summary"]
    assert len(at.get("file_uploader")) == 1, "spectral uploader must not exist"
    assert not at.exception


def test_spectral_mode_adds_the_spectrum_tab() -> None:
    at = _run(env="1")
    assert [tab.label for tab in at.tabs] == ["Image", "Spectrum", "Summary"]
    assert len(at.get("file_uploader")) == 2
    assert not at.exception


def test_sidebar_toggle_round_trips_both_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The widget must not be pinned by the env var it also writes.

    Regression guard: seeding the toggle from the environment and only writing
    on change left the flag stuck ON, so a user could not switch it back off.
    """
    import os

    monkeypatch.setenv(SPECTRAL_ENABLED_ENV, "0")
    at = _run()

    at.get("toggle")[0].set_value(True).run()
    assert len(at.tabs) == 3
    assert os.environ[SPECTRAL_ENABLED_ENV] == "1"

    at.get("toggle")[0].set_value(False).run()
    assert len(at.tabs) == 2
    assert os.environ[SPECTRAL_ENABLED_ENV] == "0"

    at.get("toggle")[0].set_value(True).run()
    assert len(at.tabs) == 3
    assert not at.exception


def test_mode_banner_reflects_the_active_branch() -> None:
    """The banner must state which branches are live.

    Checked across ``info`` and ``success`` together: the visual-only banner is
    blue (``st.info``) while visual+spectral is green (``st.success``), so
    asserting on one element type alone would silently pass on a missing banner.
    """
    off = _run(env="0")
    on = _run(env="1")

    def banners(app: AppTest) -> str:
        return " ".join(
            str(element.value) for element in (*app.info, *app.success, *app.warning)
        )

    assert "visual only" in banners(off)
    assert "visual + spectral" in banners(on)


# --- image analysis ----------------------------------------------------------


def test_upload_produces_records_with_the_frozen_schema(
    weights_available: bool, speck_image: bytes
) -> None:
    if not weights_available:
        pytest.skip("detection/morphology checkpoints not present")
    at = _upload_speck(_run(), speck_image)
    assert not at.exception
    assert not at.error

    frames = at.get("dataframe")
    assert frames, "no results table rendered"
    records = at.session_state["records"]
    assert records, "upload produced no particles"
    for record in records:
        assert tuple(record) == RECORD_KEYS
        assert record["polymer"] is None, "image-only records must not claim a polymer"


def test_no_confidence_slider_is_exposed() -> None:
    """The confidence floor is a fixed constant, so there must be no slider.

    Regression guard for a deliberate simplification: the widget was removed so
    the dashboard, the PDF and the JSON export cannot disagree about the
    operating point. A slider here would silently reintroduce that split.
    """
    at = _run()
    assert not at.get("slider"), "confidence slider must not exist"


def test_the_fixed_threshold_is_stated_in_the_ui(
    weights_available: bool, speck_image: bytes
) -> None:
    """The floor the UI claims must be the constant, not a stale literal.

    Asserted after an upload, since the controls panel only renders once an
    image is loaded.
    """
    if not weights_available:
        pytest.skip("detection/morphology checkpoints not present")
    from src.detection.inference import CONF_THRESHOLD

    at = _upload_speck(_run(), speck_image)
    captions = " ".join(str(c.value) for c in at.caption)
    assert f"{CONF_THRESHOLD:.2f}" in captions


def test_particles_reported_respect_the_fixed_threshold(
    weights_available: bool, speck_image: bytes
) -> None:
    """Every reported particle must clear the floor.

    With no slider there is no way for the user to lower the threshold, so a
    detection slipping through below it means the filter is not being applied.
    """
    if not weights_available:
        pytest.skip("detection/morphology checkpoints not present")
    from src.detection.inference import CONF_THRESHOLD

    at = _upload_speck(_run(), speck_image)
    records = at.session_state["records"]
    assert records, "the fixture image should yield detections at the fixed floor"
    below = [
        record["detection_confidence"]
        for record in records
        if float(record["detection_confidence"]) < CONF_THRESHOLD
    ]
    assert not below, f"{len(below)} particle(s) below the floor: {below}"


def test_zero_detection_is_explained_not_silent(weights_available: bool) -> None:
    """No particles must produce an explanation naming the threshold.

    The fixed floor is now the only lever, so a zero-detection result is the most
    likely thing a user will hit. It has to say why.
    """
    from src.detection.inference import CONF_THRESHOLD

    at = _upload_speck(_run(), _blank_png())
    assert not at.exception
    warnings_text = " ".join(str(w.value) for w in at.warning)
    assert warnings_text, "a blank image should warn, not render silently"
    assert f"{CONF_THRESHOLD:.2f}" in warnings_text


def test_detector_selector_is_reported_back_to_the_user(
    weights_available: bool, speck_image: bytes
) -> None:
    if not weights_available:
        pytest.skip("detection/morphology checkpoints not present")
    at = _upload_speck(_run(), speck_image)
    for name, label in (("yolo26", "YOLO26n"), ("yolov8", "YOLOv8n")):
        at = at.get("selectbox")[0].set_value(name).run()
        assert not at.exception
        captions = " ".join(str(c.value) for c in at.caption)
        assert (
            f"Detector: {label}" in captions
        ), f"summary does not name the active detector; captions were: {captions}"


def test_colour_legend_is_rendered_next_to_the_annotation(
    weights_available: bool, speck_image: bytes
) -> None:
    """Red/green/blue boxes are meaningless without a key.

    The legend is raw HTML, so it lands in ``markdown`` as a source string
    containing entities (``&#9632;``) rather than the rendered glyph. Assert on
    the markup, not on the character.
    """
    if not weights_available:
        pytest.skip("detection/morphology checkpoints not present")
    at = _upload_speck(_run(), speck_image)
    records = at.session_state["records"]

    legends = [str(m.value) for m in at.markdown if "span" in str(m.value)]
    assert legends, "no colour legend rendered"

    from src.output.annotate import class_legend, to_hex

    expected = class_legend(records)
    assert expected, "records carry no class to build a legend from"
    for name, colour in expected:
        assert to_hex(colour) in legends[0], f"{name} colour missing from the legend"
        assert name in legends[0], f"{name} missing from the legend"

    # The per-class counts printed in the legend must sum to the particles
    # actually reported, or the key is quietly lying about the result.
    counts = [
        int(chunk.split("(")[1].rstrip(")"))
        for chunk in legends[0].replace("&nbsp;", " ").split()
        if "(" in chunk and chunk.endswith(")")
    ]
    assert len(counts) == len(expected)
    assert sum(counts) == len(records)


def test_summary_tiles_and_downloads_appear_after_an_upload(
    weights_available: bool, speck_image: bytes
) -> None:
    if not weights_available:
        pytest.skip("detection/morphology checkpoints not present")
    at = _upload_speck(_run(), speck_image)
    labels = {metric.label for metric in at.metric}
    assert {"Particles", "Mean detection conf", "Dominant morphology"} <= labels
    buttons = {button.label for button in at.get("download_button")}
    assert {"Download JSON", "Download PDF"} <= buttons


# --- failure paths -----------------------------------------------------------


def test_missing_detection_weights_shows_an_error_not_a_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing checkpoint must degrade to a readable message.

    Regression guard for the silent model fallback: when no weights resolve,
    ``_load_model`` raises instead of quietly substituting another model.
    """
    from pathlib import Path

    import src.detection.inference as inference

    monkeypatch.setattr(
        inference,
        "WEIGHTS",
        {name: Path("/nonexistent") / name for name in inference.WEIGHTS},
    )
    monkeypatch.setattr(inference, "_model_cache", {})

    at = _run(env="0")
    at = at.get("file_uploader")[0].upload("speck.png", _blank_png(), "image/png").run()
    assert not at.exception
    assert at.error, "a missing checkpoint must surface as st.error"


def test_spectral_upload_without_checkpoint_reports_cleanly() -> None:
    """Enabling the spectral branch without its checkpoint must not crash."""
    import io

    import numpy as np

    at = _run(env="1")
    payload = io.BytesIO()
    np.savetxt(payload, np.zeros((2, 600)), delimiter=",")
    at = (
        at.get("file_uploader")[1]
        .upload("spectra.csv", payload.getvalue(), "text/csv")
        .run()
    )
    assert not at.exception
    assert at.error, "expected an error about the missing spectral checkpoint"


def _blank_png() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (240, 240, 240)).save(buffer, format="PNG")
    return buffer.getvalue()
