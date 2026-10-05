"""Dashboard palette and the CSS that applies it.

Dark only. A light/dark toggle was offered and removed: the annotated micrograph
is what the viewer is here to read, and a bright page around it washes out the
dark-field images the detector was tuned on. One palette also removes a whole
class of bug -- there is no longer a way for the page, the charts and the PDF to
disagree about what the dashboard looks like.

Two constraints shaped the design:

* **Charts must stay readable.** Streamlit's chart components take their text
  colour from the theme config, which CSS cannot reach. So the charts here are
  Plotly figures built from :func:`plotly_layout`, which keeps them consistent
  with the page rather than depending on a stylesheet reaching inside a canvas.
* **The detection annotation must not be recoloured.** The boxes are red, green
  and blue, so the accent is teal. A same-family accent would compete with the
  thing the viewer is meant to read first.

Selectors are Streamlit's ``data-testid`` attributes, verified present in the
1.65 frontend bundle. They are an implementation detail of Streamlit rather than
a documented API, so they are kept in one place here and
``tests/test_dashboard_theme.py`` asserts each one is a non-empty, brace-balanced
block -- an upgrade that renames a testid fails a test instead of silently
leaving part of the page unpainted.
"""

from __future__ import annotations

from typing import Any

#: Single palette. Must stay in sync with the ``[theme]`` table in
#: ``.streamlit/config.toml``, which is what Streamlit itself uses; this dict is
#: what the injected stylesheet and the Plotly figures use.
DARK: dict[str, Any] = {
    "name": "dark",
    "background": "#0B1220",
    "surface": "#111C2E",
    "surface_alt": "#1E293B",
    "text": "#E2E8F0",
    "text_muted": "#94A3B8",
    "border": "#334155",
    "grid": "#1E293B",
    "accent": "#2DD4BF",
}

#: Categorical series colours. Ordered so adjacent entries stay distinguishable,
#: and deliberately NOT the detection-box colours: a bar in #DC3C3C invites the
#: reader to match it to a red detection box, which is a different thing.
SERIES: list[str] = ["#2DD4BF", "#A78BFA", "#F59E0B", "#38BDF8", "#FB7185"]

#: Streamlit ``data-testid`` values the stylesheet targets. Kept as data so the
#: test can assert on them individually.
THEME_TARGETS: tuple[tuple[str, str], ...] = (
    ("stAppViewContainer", "background"),
    ("stHeader", "background"),
    ("stSidebar", "background"),
    ("stSidebarContent", "background"),
    ("stMain", "background"),
    ("stVerticalBlock", "background"),
    ("stMetric", "background"),
    ("stTabs", "background"),
    ("stAlert", "background"),
    ("stCaptionContainer", "background"),
    ("stExpander", "background"),
    ("stDownloadButton", "background"),
    ("stFileUploader", "background"),
    ("stSelectbox", "background"),
    ("stCheckbox", "background"),
    ("stDataFrame", "background"),
    ("stPlotlyChart", "background"),
    ("stMarkdown", "color"),
)


def palette() -> dict[str, Any]:
    """The one and only palette."""
    return DARK


def series_colours() -> list[str]:
    """Categorical series colours."""
    return SERIES


def _rule(selector: str, prop: str, value: str) -> str:
    return f'[data-testid="{selector}"] {{ {prop}: {value}; }}'


def theme_css() -> str:
    """Return the stylesheet that paints the dashboard.

    Emitted through ``st.html()`` on every run rather than cached, because a
    cached block can outlive the run that created it and leave a stale sheet
    behind after a code change.
    """
    p = palette()
    rules = [
        f'html, body, [class*="css"] {{ background-color: {p["background"]} !important; }}',
        "body, [data-testid='stMarkdown'] p, [data-testid='stMarkdown'] li, "
        "[data-testid='stMarkdown'] h1, [data-testid='stMarkdown'] h2, "
        "[data-testid='stMarkdown'] h3, [data-testid='stMarkdown'] h4, "
        f"[data-testid='stMarkdown'] span {{ color: {p['text']} !important; }}",
        f"[data-testid='stCaptionContainer'] {{ color: {p['text_muted']} !important; }}",
        f"[data-testid='stMarkdown'] code {{ color: {p['text']} !important; "
        f"background-color: {p['surface_alt']} !important; }}",
        f"a, a:visited {{ color: {p['accent']} !important; }}",
        f"hr, [data-testid='stDivider'] {{ background-color: {p['border']} !important; }}",
    ]
    for selector, prop in THEME_TARGETS:
        value = p["text"] if prop == "color" else p["surface"]
        rules.append(_rule(selector, prop, f"{value} !important"))
    rules.append(
        f'[data-testid="stSidebar"] {{ border-right: 1px solid {p["border"]} !important; }}'
    )
    rules.append(
        f'[data-testid="stMetric"] {{ border: 1px solid {p["border"]} !important; '
        f"border-radius: 0.5rem !important; padding: 0.6rem 0.8rem !important; }}"
    )
    rules.append(
        "input, textarea, select, .stTextInput, [data-baseweb='select'] > div "
        f"{{ color: {p['text']} !important; "
        f"background-color: {p['surface_alt']} !important; "
        f"border-color: {p['border']} !important; }}"
    )
    return "<style>" + "\n".join(rules) + "</style>"


def plotly_layout() -> dict[str, Any]:
    """Plotly layout overrides, matching the page.

    Paper and plot backgrounds are transparent so each figure inherits the page
    surface instead of painting its own rectangle over it -- that inheritance is
    what makes the charts blend in rather than sitting on visible blocks.
    """
    p = palette()
    return {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": p["text"], "family": "sans-serif", "size": 13},
        "title": {"font": {"color": p["text"]}},
        "xaxis": {
            "gridcolor": p["grid"],
            "linecolor": p["border"],
            "zerolinecolor": p["border"],
            "title_font": {"color": p["text_muted"]},
            "tickfont": {"color": p["text_muted"]},
        },
        "yaxis": {
            "gridcolor": p["grid"],
            "linecolor": p["border"],
            "zerolinecolor": p["border"],
            "title_font": {"color": p["text_muted"]},
            "tickfont": {"color": p["text_muted"]},
        },
        "legend": {"font": {"color": p["text_muted"]}},
        "margin": {"l": 8, "r": 8, "t": 34, "b": 8},
        "hoverlabel": {"font": {"color": p["text"]}},
    }
