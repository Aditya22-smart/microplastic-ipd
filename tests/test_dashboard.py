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
from src.output import theme

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

    legend_els = [m for m in at.markdown if "span" in str(m.value)]
    assert legend_els, "no colour legend rendered"
    legends = [str(m.value) for m in legend_els]

    # The string above is identical whether or not Streamlit is allowed to
    # render it, so asserting on the source cannot catch the real failure --
    # which is the user seeing literal
    # `<span style="color:#286EE6;font-size:1.15em">` on the page because
    # st.markdown escapes HTML by default. Check the flag that decides it.
    assert legend_els[0].proto.allow_html, (
        "legend HTML will be escaped and shown as markup soup; "
        "st.markdown needs unsafe_allow_html=True"
    )

    from src.output.annotate import class_legend, to_hex

    legend = legends[0]
    expected = class_legend(records)
    assert expected, "records carry no class to build a legend from"
    for name, colour in expected:
        assert to_hex(colour) in legend, f"{name} colour missing from the legend"
        assert name in legend, f"{name} missing from the legend"

    # The per-class counts printed in the legend must sum to the particles
    # actually reported, or the key is quietly lying about the result.
    counts = [
        int(chunk.split("(")[1].rstrip(")"))
        for chunk in legend.replace("&nbsp;", " ").split()
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


# --- palette ----------------------------------------------------------------


def test_there_is_no_appearance_toggle() -> None:
    """The dashboard is dark-only; a theme control would be a dead end.

    Streamlit exposes no Python API to set the active theme, so a sidebar switch
    could only repaint part of the page. Asserting the control is *absent* keeps a
    half-removed toggle from creeping back in.
    """
    assert not _run().get(
        "segmented_control"
    ), "an appearance control reappeared; the dashboard is dark-only by design"


def test_a_stylesheet_is_emitted_on_every_run() -> None:
    """``st.html`` with a <style> block arrives as a SpecialBlock.

    Asserted as a count rather than by content: AppTest computes no CSS, and the
    SpecialBlock carries no readable value, so the contents are verified in
    tests/test_dashboard_theme.py against the pure function that produced them.
    """
    at = _run()
    blocks = [e for e in at.main if type(e).__name__ == "SpecialBlock"]
    assert blocks, "no stylesheet emitted; the page would use Streamlit's default"


def test_every_summary_chart_is_themed(
    weights_available: bool, speck_image: bytes
) -> None:
    """No chart may fall back to Streamlit's default colours.

    ``st.plotly_chart`` takes no ``layout=`` argument, so a chart that forgets the
    ``_themed`` helper keeps white text and a white plot background -- invisible
    details of dark mode that only a rendered page would show. Checked on the
    serialized spec, which is what the browser receives.
    """
    if not weights_available:
        pytest.skip("detection/morphology checkpoints not present")
    at = _upload_speck(_run(), speck_image)
    charts = at.get("plotly_chart")
    assert charts, "no charts rendered"

    for chart in charts:
        layout = _chart_layout(chart)
        assert layout, "a chart arrived with no layout"
        assert (
            layout["font"]["color"] == theme.palette()["text"]
        ), f"chart font is {layout['font']['color']}, not the palette text colour"
        # Transparent, so the figure inherits the dark page rather than painting
        # a white block over it.
        assert layout["paper_bgcolor"] == "rgba(0,0,0,0)"
        assert layout["plot_bgcolor"] == "rgba(0,0,0,0)"


def _chart_layout(chart) -> dict:
    """The ``layout`` sub-dict of a chart's serialized spec."""
    import json

    spec = getattr(chart, "spec", None)
    if not spec:
        return {}
    try:
        parsed = json.loads(spec)
    except (TypeError, ValueError):
        return {}
    return parsed.get("layout", {})


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


def test_spectral_upload_without_checkpoint_reports_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enabling the spectral branch without its checkpoint must not crash."""
    import io
    from pathlib import Path

    import numpy as np

    import src.models.predict_polymer as predict_polymer

    monkeypatch.setattr(
        predict_polymer, "DEFAULT_WEIGHTS", Path("/nonexistent/spectral_weights.pth")
    )
    monkeypatch.setattr(predict_polymer, "_model_cache", {})

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


def test_spectral_upload_with_checkpoint_produces_predictions() -> None:
    """Uploading spectra when the checkpoint exists renders predictions and charts."""
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
    assert not at.error
    assert len(at.session_state["spectrum_results"]) == 2
    frames = at.get("dataframe")
    assert frames, "no spectral results table rendered"


@pytest.mark.parametrize(
    ("colour", "expected"),
    [
        ((220, 60, 60), "#DC3C3C"),
        ((0, 0, 0), "#000000"),
        ((255, 255, 255), "#FFFFFF"),
    ],
)
def test_to_hex_emits_uppercase_six_digit_hex(
    colour: tuple[int, int, int], expected: str
) -> None:
    """The legend colour is interpolated into a raw-HTML ``style`` attribute.

    So the exact shape of the string matters twice over: a malformed value would
    break the styling, and anything but hex could carry markup into the page.
    """
    from src.output.annotate import to_hex

    assert to_hex(colour) == expected


@pytest.mark.parametrize("bad", [(-1, 0, 0), (0, 300, 0), (0, 0, 999)])
def test_to_hex_clamps_out_of_range_channels(bad: tuple[int, int, int]) -> None:
    """Out-of-range channels clamp instead of widening the string.

    ``"#{:02X}"`` happily renders 300 as ``12C``, silently shifting the digits
    and corrupting the colour -- and a longer value would let non-hex text slip
    through into the style attribute.
    """
    from src.output.annotate import to_hex

    value = to_hex(bad)
    assert len(value) == 7, f"expected #RRGGBB, got {value!r}"
    assert value.startswith("#")
    int(value[1:], 16)  # must parse as hex, or it is not a colour


def _blank_png() -> bytes:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), (240, 240, 240)).save(buffer, format="PNG")
    return buffer.getvalue()
