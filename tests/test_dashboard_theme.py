"""Palette and summary-statistics tests.

Two groups:

* :mod:`src.output.theme` is a pure function of one palette dict, so it is tested
  directly. The stylesheet targets Streamlit ``data-testid`` attributes, which
  are an internal implementation detail rather than a documented API -- so each
  selector is asserted individually. A Streamlit upgrade that renames one should
  fail a test here instead of silently leaving part of the page unpainted.
* :mod:`src.output.viz` holds the arithmetic behind every headline number, so it
  is checked against hand-computed values rather than "whatever it returned".

The stylesheet's *rendering* cannot be asserted through AppTest: ``st.html`` with
a ``<style>`` block arrives in the element tree as a ``SpecialBlock`` with no
readable value, and AppTest computes no CSS. So the tests assert the stylesheet
was emitted and that its contents are well formed and derived from the palette,
not that a browser would paint it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

from src.output import theme, viz

CONFIG = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"


# --- theme ------------------------------------------------------------------
def test_palette_defines_every_colour_the_stylesheet_interpolates() -> None:
    """A missing key raises KeyError at render time, not at import time.

    That failure mode is bad: it would appear only once someone loaded the page.
    """
    required = {
        "name",
        "background",
        "surface",
        "surface_alt",
        "text",
        "text_muted",
        "border",
        "grid",
        "accent",
    }
    assert required <= set(theme.palette()), required - set(theme.palette())
    assert theme.palette()["name"] == "dark"


def test_streamlit_is_also_served_the_dark_palette() -> None:
    """``theme.DARK`` and ``config.toml`` must agree.

    Streamlit paints the widgets from its own config and the injected stylesheet
    paints the containers. If the two files drift, the page ends up dark chrome on
    a light background -- and only the rendered page shows it, which is why this is
    compared rather than eyeballed.
    """
    table = tomllib.loads(CONFIG.read_text("utf-8"))["theme"]
    assert table["base"] == "dark"
    assert table["backgroundColor"] == theme.DARK["background"]
    assert table["secondaryBackgroundColor"] == theme.DARK["surface"]
    assert table["textColor"] == theme.DARK["text"]
    assert table["primaryColor"] == theme.DARK["accent"]


def test_light_mode_is_gone() -> None:
    """The dashboard is dark-only, deliberately.

    Guards against a half-removed toggle: if someone re-adds a light palette or
    an appearance argument, this fails rather than the sidebar quietly gaining a
    control that half works.
    """
    for name in ("LIGHT", "ACCENT", "_SERIES_LIGHT", "PALETTES"):
        assert not hasattr(theme, name), f"{name} reintroduced a light mode"
    config = tomllib.loads(CONFIG.read_text("utf-8"))["theme"]
    assert "light" not in config, "config.toml still defines a light palette"


def test_text_contrasts_with_its_own_background() -> None:
    """Body text must not be near-invisible against its page surface.

    Checked as a coarse relative-luminance gap rather than a WCAG ratio: the
    exact foreground the browser paints is not knowable here. This catches the
    real regression -- a dark text colour left at its light value.
    """

    def luminance(hex_colour: str) -> float:
        r, g, b = (int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    p = theme.palette()
    assert abs(luminance(p["text"]) - luminance(p["background"])) > 0.4
    # Muted text is used for captions, which sit on the surface rather than the
    # page, so it needs its own gap rather than inheriting the one above.
    assert abs(luminance(p["text_muted"]) - luminance(p["surface"])) > 0.3


def test_theme_css_is_well_formed() -> None:
    css = theme.theme_css()
    assert css.startswith("<style>") and css.endswith("</style>")
    assert css.count("{") == css.count("}"), "unbalanced braces"
    # A stray f-string placeholder surviving into the stylesheet would silently
    # emit a property named "background" with no value.
    assert not re.search(r"\{[a-zA-Z_]+\}", css.replace("{ ", "").replace(" }", ""))


def test_theme_css_targets_every_declared_testid() -> None:
    """Each selector in the target table must appear in the emitted stylesheet.

    Guards the drift where a row is added to ``THEME_TARGETS`` but the
    corresponding rule is forgotten.
    """
    css = theme.theme_css()
    for selector, _prop in theme.THEME_TARGETS:
        assert f'[data-testid="{selector}"]' in css, f"{selector} has no rule"


def test_theme_css_is_built_from_the_palette() -> None:
    """No hardcoded colours: the stylesheet must come from ``palette()``.

    A literal left in the CSS would render one element light on a dark page, and
    the only way to see it is to load the page.
    """
    css = theme.theme_css()
    p = theme.palette()
    for key in ("background", "surface", "text", "text_muted", "border", "accent"):
        assert p[key] in css, f"{key}={p[key]} is not in the stylesheet"
    # Strip the palette's own hex codes; what is left must not be another colour.
    for hex_colour in set(re.findall(r"#[0-9A-Fa-f]{6}", css)) - set(p.values()):
        raise AssertionError(f"{hex_colour} is hardcoded in the stylesheet")


def test_plotly_layout_is_transparent_and_matches_the_palette() -> None:
    layout = theme.plotly_layout()
    # Transparent so the figure inherits the page surface instead of painting
    # its own rectangle over it.
    assert layout["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert layout["plot_bgcolor"] == "rgba(0,0,0,0)"
    p = theme.palette()
    assert layout["font"]["color"] == p["text"]
    assert layout["xaxis"]["gridcolor"] == p["grid"]
    assert layout["yaxis"]["tickfont"]["color"] == p["text_muted"]


def test_series_colours_are_valid_hex() -> None:
    colours = theme.series_colours()
    assert len(colours) >= 4
    assert len(set(colours)) == len(colours), "duplicate series colour"
    for colour in colours:
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", colour), colour


def test_series_colours_avoid_the_detection_box_colors() -> None:
    """Chart series must not reuse the annotation palette.

    A bar in #DC3C3C invites the reader to match it to a red detection box, which
    is a different thing entirely.
    """
    from src.output.annotate import CLASS_COLORS, to_hex

    box_colours = {to_hex(c) for c in CLASS_COLORS.values()}
    assert not (set(theme.series_colours()) & box_colours)


# --- viz --------------------------------------------------------------------
def _record(pid, bbox, det_conf, cls="bead", morph="sphere", morph_conf=0.9):
    return {
        "particle_id": pid,
        "bbox": bbox,
        "detection_confidence": det_conf,
        "class": cls,
        "morphology": morph,
        "morphology_confidence": morph_conf,
        "polymer": None,
        "polymer_confidence": None,
    }


def test_field_stats_on_an_empty_list_is_not_an_error() -> None:
    """The Summary tab renders this path before any upload, so it must not raise."""
    stats = viz.field_stats([], (640, 480), 0.5)
    assert stats["count"] == 0
    assert viz.interpret(stats) == [
        "No particles were reported, so there is nothing to interpret."
    ]


def test_field_stats_arithmetic_matches_hand_computed_values() -> None:
    records = [
        _record(1, [0, 0, 10, 10], 0.90),  # area 100, centre (5, 5)
        _record(2, [100, 100, 20, 20], 0.60),  # area 400, centre (110, 110)
        _record(3, [300, 200, 30, 10], 0.80, morph="fiber"),  # area 300, (315, 205)
    ]
    stats = viz.field_stats(records, (640, 480), 0.5)

    assert stats["count"] == 3
    # mean 0.7666.., median 0.80 -- the median is what the tile reports, and it
    # must not be dragged toward the low outlier the way a mean would be.
    assert stats["mean_detection"] == pytest.approx((0.9 + 0.6 + 0.8) / 3)
    assert stats["median_detection"] == pytest.approx(0.80)
    assert stats["min_detection"] == pytest.approx(0.60)
    assert stats["mean_morphology"] == pytest.approx(0.9)
    assert stats["dominant_morphology"] == "sphere"
    assert stats["dominant_share"] == pytest.approx(2 / 3)
    assert stats["area_min"] == pytest.approx(100)
    assert stats["area_max"] == pytest.approx(400)
    assert stats["area_median"] == pytest.approx(300)
    assert stats["coverage"] == pytest.approx((100 + 400 + 300) / (640 * 480))
    assert stats["centroids"] == [(5.0, 5.0), (110.0, 110.0), (315.0, 205.0)]


def test_coverage_is_clamped_to_a_fraction() -> None:
    """Overlapping boxes can sum past the frame; the figure must not read 342%."""
    records = [_record(i, [0, 0, 300, 300], 0.9) for i in range(1, 5)]
    stats = viz.field_stats(records, (320, 240), 0.5)
    assert stats["coverage"] == 1.0


def test_coverage_guards_a_zero_sized_image() -> None:
    """A missing image must not raise ZeroDivisionError on the Summary tab."""
    records = [_record(1, [0, 0, 10, 10], 0.9)]
    assert viz.field_stats(records, (0, 0), 0.5)["coverage"] == 0.0


def test_nearest_neighbour_flags_particles_that_overlap() -> None:
    """Two boxes on the same object read as two particles, which overcounts."""
    touching = viz.field_stats(
        [_record(1, [0, 0, 40, 40], 0.9), _record(2, [2, 2, 40, 40], 0.9)],
        (640, 480),
        0.5,
    )
    assert touching["touching"] == 1.0

    separated = viz.field_stats(
        [_record(1, [0, 0, 10, 10], 0.9), _record(2, [600, 440, 10, 10], 0.9)],
        (640, 480),
        0.5,
    )
    assert separated["touching"] == 0.0


def test_nearest_neighbour_handles_a_single_particle() -> None:
    stats = viz.field_stats([_record(1, [0, 0, 10, 10], 0.9)], (640, 480), 0.5)
    assert stats["touching"] == 0.0


def test_confidence_histogram_always_reports_every_band() -> None:
    """Bins are fixed, so a run with no high-confidence detection still shows the
    axis. A histogram that silently drops empty bands changes width between runs
    and makes two results incomparable."""
    records = [_record(1, [0, 0, 5, 5], 0.55), _record(2, [0, 0, 5, 5], 0.95)]
    histogram = viz.confidence_histogram(records)
    assert set(histogram) == {label for _, _, label in viz.CONFIDENCE_BINS}
    assert sum(histogram.values()) == 2
    assert histogram["0.50-0.60"] == 1
    assert histogram["0.90-1.00"] == 1
    assert histogram["0.60-0.70"] == 0


def test_confidence_histogram_ignores_values_below_the_first_bin() -> None:
    """The floor is enforced upstream, so anything under 0.50 is impossible.

    If it ever appears, the count must not silently vanish -- the total is what a
    reader checks against the particle count.
    """
    histogram = viz.confidence_histogram([_record(1, [0, 0, 5, 5], 0.10)])
    assert sum(histogram.values()) == 0


def test_marginal_fraction_is_the_share_below_the_ceiling() -> None:
    records = [
        _record(i, [0, 0, 5, 5], conf) for i, conf in enumerate((0.55, 0.65, 0.95), 1)
    ]
    assert viz.marginal_fraction(records) == pytest.approx(2 / 3)
    assert viz.marginal_fraction([]) == 0.0


def test_median_handles_both_parities_and_empty() -> None:
    assert viz.median([]) == 0.0
    assert viz.median([3.0, 1.0, 2.0]) == 2.0
    assert viz.median([4.0, 1.0, 3.0, 2.0]) == 2.5


def test_interpret_reports_every_number_it_claims() -> None:
    """The prose is only trustworthy if each clause matches the statistics."""
    records = [
        _record(1, [0, 0, 10, 10], 0.55),
        _record(2, [200, 0, 20, 20], 0.60),
    ]
    stats = viz.field_stats(records, (640, 480), 0.5)
    text = " ".join(viz.interpret(stats))

    assert "2 particle(s) detected" in text
    # Median of 0.55 and 0.60 is 0.575. Formatted to two places it prints 0.57,
    # because 0.575 is not exactly representable in binary and lands just under.
    assert "0.57" in text or "0.58" in text
    assert "sphere" in text
    assert "100%" in text  # both detections are below 0.70
    # The size caveat must travel with the size number, or the reader takes px²
    # for microns.
    assert "no calibration metadata" in text


def test_interpret_flags_an_overcount_risk() -> None:
    records = [
        _record(1, [0, 0, 40, 40], 0.95),
        _record(2, [1, 1, 40, 40], 0.95),
    ]
    text = " ".join(viz.interpret(viz.field_stats(records, (640, 480), 0.5)))
    assert "upper bound" in text, "touching boxes must be called out as a count risk"


def test_interpret_omits_the_marginal_line_when_nothing_is_marginal() -> None:
    records = [_record(i, [i * 100, 0, 10, 10], 0.95) for i in range(1, 4)]
    text = " ".join(viz.interpret(viz.field_stats(records, (640, 480), 0.5)))
    assert "cleared 0.70 confidence" in text
