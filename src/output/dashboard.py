"""Streamlit dashboard for the microplastic pipeline.

Two independent branches meet here:

* **visual** — Stage 1A detection + Stage 1B morphology  (image in)
* **spectral** — Stage 2A preprocessing + Stage 2B polymer  (spectrum in)

The spectral branch is behind the ``MP_SPECTRAL_ENABLED`` feature flag and is
**off by default**. When off, the Spectrum tab is never created, so there is no
spectral uploader and no polymer prediction anywhere in the UI. Use the sidebar
toggle to switch it on for a session that needs it.

Grad-CAM re-crops each particle from the *already uploaded* image using the
``bbox`` recorded by the pipeline, so explainability costs no second detection
pass.
"""

from __future__ import annotations

import io
import sys
from collections import Counter
from html import escape
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # `streamlit run` adds only the *script's own folder* (src/output/) to
    # sys.path -- never the working directory -- so `streamlit run
    # src/output/dashboard.py` from the repo root fails with
    # "No module named 'src'" unless the repo root is put on the path here.
    # Every other entry point is launched as `python -m src...`, which gets the
    # working directory for free, so this is the one script that needs it.
    sys.path.insert(0, str(REPO_ROOT))

from src.config import set_spectral, spectral_enabled  # noqa: E402
from src.detection.inference import CONF_THRESHOLD, WEIGHTS  # noqa: E402
from src.models.full_pipeline import analyze_image, analyze_spectrum  # noqa: E402
from src.output.annotate import class_legend, draw_detections, to_hex  # noqa: E402
from src.output.report_generator import build_report_pdf  # noqa: E402
from src.output.theme import (  # noqa: E402
    palette,
    plotly_layout,
    series_colours,
    theme_css,
)
from src.output.viz import CONFIDENCE_BINS, field_stats, interpret  # noqa: E402

st.set_page_config(page_title="Microplastic IPD", layout="wide")

st.title("Automated Microplastic Detection & Classification")
st.caption(
    "Stage 1A detection  ·  Stage 1B morphology  ·  "
    "Stage 2A/2B spectral polymer classification"
)

MODEL_LABELS = {
    name: {"yolo26": "YOLO26n (primary)", "yolov8": "YOLOv8n (baseline)"}.get(
        name, name
    )
    for name in WEIGHTS
}
UPLOAD_DIR = Path("results") / "dashboard"


def _themed(figure):
    """Apply the palette to a Plotly figure.

    ``st.plotly_chart`` has no ``layout=`` argument, so the overrides from
    :func:`plotly_layout` are applied to the figure itself. Mutating here rather
    than at each call site keeps every chart on the same palette, which is the
    whole point -- a summary tab where one chart ignores the palette looks broken
    even when each chart is individually correct.
    """
    figure.update_layout(**plotly_layout())
    return figure


def _legend_strip(records: list[dict]) -> None:
    """Render the colour key for the annotated image.

    Without this the viewer sees red, green and blue boxes with nothing telling
    them what the colours mean.
    """
    items = class_legend(records)
    if not items:
        return
    counts = Counter(record["class"] for record in records)
    # `unsafe_allow_html=True` is required: Streamlit defaults it to False and
    # escapes the markup, so the legend rendered as visible
    # `<span style="color:#286EE6...">&#9632;</span>` soup instead of coloured
    # swatches. Safe here because every interpolated value is ours -- `to_hex`
    # is clamped to `#RRGGBB` and the class name is HTML-escaped below. Nothing
    # user-supplied reaches this string.
    swatches = "&nbsp;&nbsp;&nbsp;".join(
        f'<span style="color:{to_hex(color)};font-size:1.15em">&#9632;</span> '
        f"&nbsp;{escape(name)} ({counts.get(name, 0)})"
        for name, color in items
    )
    st.markdown(swatches, unsafe_allow_html=True)


# --- palette ----------------------------------------------------------------
# Single dark palette, applied on every run. `.streamlit/config.toml` already
# tells Streamlit to serve dark; this stylesheet is what makes the elements
# Streamlit does not theme from config -- metric cards, tabs, dataframe and
# chart surfaces -- match. No toggle: there is no Python API to set Streamlit's
# active theme (`st.context.theme` is read-only), so an in-app switch would have
# to repaint the page, and the page is the thing a dark-mode toggle must not get
# half-right.
st.html(theme_css(), unsafe_allow_javascript=False)

# --- sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    if "spectral_on" not in st.session_state:
        # Seed from the env var exactly once per session. Afterwards the widget
        # owns the value, so the flag cannot be pinned ON by a stale environment.
        st.session_state["spectral_on"] = spectral_enabled()
    spectral_on = st.toggle(
        "Enable spectral branch (Stage 2A/2B)",
        key="spectral_on",
        help=(
            "Off: image pipeline only. On: adds the polymer classifier, which "
            "needs a trained 1D-CNN checkpoint."
        ),
    )
    st.caption(
        "`MP_SPECTRAL_ENABLED=1` enables the spectral branch; "
        "it is **off by default**."
    )

# Publish the sidebar choice to the process env so the pipeline modules and the
# report generator observe the same value.
set_spectral(spectral_on)
spectral_active = spectral_on

if spectral_active:
    st.success(
        "**Mode: visual + spectral** — image analysis (Stage 1A/1B) and polymer "
        "classification (Stage 2A/2B) are both active."
    )
else:
    st.info(
        "**Mode: visual only** — the spectral branch is disabled, so only image "
        "analysis (Stage 1A/1B) runs. Enable it in the sidebar to add polymer "
        "classification."
    )

# --- tabs -------------------------------------------------------------------
tab_labels = (
    ["Image", "Summary"] if not spectral_active else ["Image", "Spectrum", "Summary"]
)
if spectral_active:
    tab_image, tab_spectrum, tab_summary = st.tabs(tab_labels)
else:
    tab_image, tab_summary = st.tabs(tab_labels)


# --- Image tab --------------------------------------------------------------
with tab_image:
    uploaded = st.file_uploader(
        "Upload a microscope image",
        type=["jpg", "jpeg", "png"],
        help="Bright-field or dark-field micrograph. Detection runs on CPU.",
    )

    if uploaded is None:
        with st.container(border=True):
            st.markdown("#### No image loaded")
            st.caption(
                "Upload a micrograph to run Stage 1A detection and Stage 1B "
                "morphology classification. Input and annotated output are shown "
                "side by side below."
            )
    else:
        from PIL import Image

        image = Image.open(uploaded).convert("RGB")

        with st.container(border=True):
            detector_col, label_col = st.columns([1, 1])
            with detector_col:
                model_name = st.selectbox(
                    "Detector",
                    options=list(WEIGHTS),
                    index=list(WEIGHTS).index("yolo26") if "yolo26" in WEIGHTS else 0,
                    format_func=lambda name: MODEL_LABELS.get(name, name),
                    help="Both checkpoints are nano variants trained on microplastic data.",
                )
            with label_col:
                show_labels = st.checkbox("Annotate image", value=True)
            st.caption(
                f"Detection confidence floor is fixed at "
                f"{CONF_THRESHOLD:.2f} (`CONF_THRESHOLD` in "
                "`src/detection/inference.py`)."
            )

        # No confidence widget: the floor is a fixed project-wide constant, so
        # every reported number -- dashboard, PDF, JSON -- comes from one value.
        conf_threshold = CONF_THRESHOLD

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        image_path = UPLOAD_DIR / "dashboard_upload.png"
        image.save(image_path)

        try:
            records = analyze_image(
                image_path,
                conf_threshold=conf_threshold,
                model_name=model_name,
            )
        except FileNotFoundError as exc:
            st.error(str(exc))
            st.caption(
                "The checkpoints are committed under `weights/`. If this persists, "
                "run `uv sync` and confirm the files are present."
            )
            records = []
        except Exception as exc:  # noqa: BLE001 - surface, never crash the tab
            st.error(f"Detection failed: {type(exc).__name__}: {exc}")
            records = []
        else:
            st.session_state["records"] = records
            st.session_state["source_image"] = image
            st.session_state["model_name"] = model_name

        annotated = draw_detections(image, records, show_labels=show_labels)

        # Input and output side by side. Both images are rendered at roughly
        # half the dashboard width, which for a typical 640-480 micrograph is
        # its native size -- stacking them full width upscaled each one ~2x,
        # which read as blur and forced a very long scroll.
        input_col, result_col = st.columns(2)
        with input_col:
            st.markdown("**Input**")
            st.image(image, width="stretch")
        with result_col:
            st.markdown("**Detected**")
            st.image(annotated, width="stretch")
            st.caption(
                f"{len(records)} particle(s) · "
                f"{MODEL_LABELS.get(model_name, model_name)} · "
                f"conf ≥ {conf_threshold:.2f}",
                help=(
                    "The caption reports the detector actually used and the "
                    "confidence floor actually applied, so a result can be "
                    "reproduced from the screen alone."
                ),
            )

        if records:
            detection_confidences = [
                float(record["detection_confidence"]) for record in records
            ]
            morphology_confidences = [
                float(record["morphology_confidence"]) for record in records
            ]
            morphology_counts = Counter(record["morphology"] for record in records)

            chip = st.columns(4)
            chip[0].metric("Particles", len(records))
            chip[1].metric(
                "Mean detection conf",
                f"{sum(detection_confidences) / len(records):.3f}",
            )
            chip[2].metric(
                "Mean morphology conf",
                f"{sum(morphology_confidences) / len(records):.3f}",
            )
            chip[3].metric(
                "Dominant morphology", morphology_counts.most_common(1)[0][0]
            )

            _legend_strip(records)

            # Morphology mix as a bar rather than a fourth number: the dominant
            # chip above says which class wins, this shows how much of the field
            # it actually is. A 9-of-10 result and a 5-of-10 result both report
            # the same dominant label, and they are not the same finding.
            st.bar_chart(
                {
                    "particles": {
                        name: count for name, count in morphology_counts.most_common()
                    }
                },
                horizontal=True,
                height=28 + 22 * len(morphology_counts),
                color="#0D9488",
            )

            st.subheader("Per-particle results")
            st.dataframe(
                [
                    {
                        "id": record["particle_id"],
                        "class": record["class"],
                        "detection_confidence": record["detection_confidence"],
                        "morphology": record["morphology"],
                        "morphology_confidence": round(
                            float(record["morphology_confidence"]), 4
                        ),
                        "bbox (xywh)": record["bbox"],
                    }
                    for record in records
                ],
                width="stretch",
                hide_index=True,
            )

            with st.expander("Grad-CAM — why this morphology?"):
                particle_id = st.selectbox(
                    "Particle",
                    options=[record["particle_id"] for record in records],
                    format_func=lambda pid: next(
                        f"#{pid} - {r['class']} / {r['morphology']}"
                        for r in records
                        if r["particle_id"] == pid
                    ),
                )
                record = next(r for r in records if r["particle_id"] == particle_id)
                try:
                    from src.evaluation.grad_cam_viz import (
                        grad_cam_for_crop,
                        overlay_cam_on_image,
                    )

                    x, y, width, height = record["bbox"]
                    crop = image.crop((x, y, x + width, y + height))
                    cam = grad_cam_for_crop(crop)
                    cam_col, crop_col = st.columns(2)
                    with cam_col:
                        st.image(
                            overlay_cam_on_image(crop, cam),
                            caption=f"Grad-CAM for #{particle_id}",
                            width="stretch",
                        )
                    with crop_col:
                        st.image(
                            crop,
                            caption=f"particle crop ({crop.width}×{crop.height} px)",
                            width="stretch",
                        )
                    st.caption(
                        f"Predicted **{record['morphology']}** at "
                        f"{float(record['morphology_confidence']):.3f}. Warm-coloured "
                        "regions (red) contributed most to that decision. The crop is "
                        f"only {crop.width}×{crop.height} px, so it is displayed "
                        "magnified."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"Grad-CAM unavailable: {type(exc).__name__}: {exc}")
        else:
            st.warning(
                f"No particles above {conf_threshold:.2f}. Lower the confidence "
                "floor, or this image may not contain microplastics."
            )


# --- Spectrum tab (spectral branch only) ------------------------------------
if spectral_active:
    with tab_spectrum:
        spectra = st.file_uploader(
            "Upload preprocessed spectra (CSV/txt)",
            type=["csv", "txt"],
            accept_multiple_files=True,
            help="Stage 2A preprocessing produces these; Stage 2B classifies them.",
        )
        if spectra:
            import numpy as np

            results: list[dict] = []
            for uploaded_spectrum in spectra:
                try:
                    spectrum = np.loadtxt(
                        io.StringIO(uploaded_spectrum.getvalue().decode("utf-8")),
                        delimiter=",",
                    )
                    if spectrum.ndim == 1:
                        spectrum = spectrum[None, :]
                    for row in spectrum:
                        results.append(analyze_spectrum(row))
                except FileNotFoundError as exc:
                    st.error(str(exc))
                    break
                except Exception as exc:  # noqa: BLE001
                    st.error(
                        f"{uploaded_spectrum.name}: " f"{type(exc).__name__}: {exc}"
                    )
            if results:
                st.session_state["spectrum_results"] = results
                st.dataframe(
                    [
                        {
                            "id": result["particle_id"],
                            "polymer": result["polymer"],
                            "confidence": round(float(result["polymer_confidence"]), 4),
                        }
                        for result in results
                    ],
                    width="stretch",
                    hide_index=True,
                )
        elif not st.session_state.get("spectrum_results"):
            with st.container(border=True):
                st.markdown("#### No spectra loaded")
                st.caption(
                    "Upload one or more preprocessed spectra to run Stage 2B polymer "
                    "classification. The spectral model is not committed yet, so this "
                    "branch will report a missing-checkpoint error until it is trained."
                )


# --- Summary tab ------------------------------------------------------------
with tab_summary:
    records = st.session_state.get("records", [])
    spectrum_results = (
        st.session_state.get("spectrum_results", []) if spectral_active else []
    )

    if records:
        import plotly.express as px
        import plotly.graph_objects as go

        source_image = st.session_state.get("source_image")
        image_size = source_image.size if source_image else (0, 0)
        model_label = MODEL_LABELS.get(
            st.session_state.get("model_name"),
            st.session_state.get("model_name", "n/a"),
        )
        series = series_colours()

        stats = field_stats(records, image_size, CONF_THRESHOLD)

        st.subheader("What this field shows")
        for line in interpret(stats):
            st.markdown(f"- {line}")

        st.caption(
            f"Detector: {model_label} · confidence floor {CONF_THRESHOLD:.2f} · "
            f"{image_size[0]}×{image_size[1]} px"
        )

        tile = st.columns(5)
        tile[0].metric(
            "Particles", stats["count"], help="Boxes reported above the fixed floor."
        )
        tile[1].metric(
            "Median detection conf",
            f"{stats['median_detection']:.3f}",
            help="Median, not mean: one confident outlier should not lift the headline.",
            delta_color="off",
        )
        tile[2].metric(
            "Below 0.70 conf",
            f"{stats['marginal_share']:.0%}",
            help="Detections the model was least sure about.",
            delta_color="inverse",
        )
        tile[3].metric(
            "Frame covered",
            f"{stats['coverage']:.1%}",
            help="Summed box area over frame area.",
            delta_color="off",
        )
        tile[4].metric(
            "Overlapping pairs",
            f"{stats['touching']:.0%}",
            help="Particles within touching distance of a neighbour.",
            delta_color="inverse",
        )

        _legend_strip(records)

        # --- distributions --------------------------------------------------
        left, right = st.columns(2)

        with left:
            st.markdown("**Morphology mix (Stage 1B)**")
            st.plotly_chart(
                _themed(
                    px.bar(
                        x=list(stats["morphology_counts"].values()),
                        y=list(stats["morphology_counts"].keys()),
                        orientation="h",
                        labels={"x": "particles", "y": ""},
                        color_discrete_sequence=series,
                        text=list(stats["morphology_counts"].values()),
                    ),
                ),
                width="stretch",
                config={"displayModeBar": False},
            )

        with right:
            st.markdown("**Detection confidence spread**")
            st.plotly_chart(
                _themed(
                    go.Figure(
                        go.Bar(
                            x=[label for _, _, label in CONFIDENCE_BINS],
                            y=[
                                stats["histogram"].get(label, 0)
                                for _, _, label in CONFIDENCE_BINS
                            ],
                            marker_color=series[0],
                            hovertemplate="%{x}<br>%{y} particle(s)<extra></extra>",
                        )
                    ).update_layout(barmode="stack"),
                ),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.caption(
                "A tall first bar means the count rests on marginal detections, "
                "and would drop if the floor were raised."
            )

        left, right = st.columns(2)

        with left:
            st.markdown("**Particle size distribution**")
            areas = sorted(float(r["bbox"][2]) * float(r["bbox"][3]) for r in records)
            st.plotly_chart(
                _themed(
                    go.Figure(
                        go.Histogram(
                            x=areas,
                            marker_color=series[1],
                            nbinsx=max(4, min(14, len(areas))),
                            hovertemplate="%{x:.0f} px²<br>%{y} particle(s)<extra></extra>",
                        )
                    ),
                ),
                width="stretch",
                config={"displayModeBar": False},
            )
            st.caption(
                f"Median {stats['area_median']:.0f} px² "
                f"({stats['area_min']:.0f}-{stats['area_max']:.0f} px²). "
                "Pixels only — the images carry no calibration metadata, so a "
                "micron figure is not available."
            )

        with right:
            st.markdown("**Where the particles sit**")
            if image_size[0] > 0:
                xs = [c[0] for c in stats["centroids"]]
                ys = [c[1] for c in stats["centroids"]]
                st.plotly_chart(
                    _themed(
                        go.Figure(
                            go.Scatter(
                                x=xs,
                                y=ys,
                                mode="markers",
                                marker={
                                    "size": 11,
                                    "color": [
                                        float(r["detection_confidence"])
                                        for r in records
                                    ],
                                    "colorscale": "Teal",
                                    "line": {
                                        "width": 1,
                                        "color": palette()["border"],
                                    },
                                    "colorbar": {
                                        "title": "conf",
                                        "thickness": 10,
                                        "outlinewidth": 0,
                                    },
                                },
                                hovertemplate="(%{x:.0f}, %{y:.0f}) px<extra></extra>",
                            )
                        ).update_yaxes(
                            scaleanchor="x", scaleratio=1, autorange="reversed"
                        ),
                    ),
                    width="stretch",
                    config={"displayModeBar": False},
                )
                st.caption(
                    "Particle centres in frame coordinates, marker colour = "
                    "detection confidence. Empty regions are areas the detector "
                    "found nothing."
                )
            else:
                st.info("Image size unavailable, so positions cannot be plotted.")

        # --- cross-tab ------------------------------------------------------
        st.markdown("**Detection class × morphology**")
        cross = Counter((r["class"], r["morphology"]) for r in records)
        classes = sorted({key[0] for key in cross})
        morphologies = sorted({key[1] for key in cross})
        st.plotly_chart(
            _themed(
                go.Figure(
                    go.Heatmap(
                        z=[
                            [cross.get((c, m), 0) for m in morphologies]
                            for c in classes
                        ],
                        x=morphologies,
                        y=classes,
                        colorscale="Teal",
                        text=[
                            [str(cross.get((c, m), 0)) for m in morphologies]
                            for c in classes
                        ],
                        texttemplate="%{text}",
                        hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
                    )
                ),
            ),
            width="stretch",
            config={"displayModeBar": False},
        )

        if spectral_active and spectrum_results:
            polymer_counts = Counter(r["polymer"] for r in spectrum_results)
            st.subheader("Polymer (Stage 2B)")
            st.plotly_chart(
                _themed(
                    px.bar(
                        x=list(polymer_counts.keys()),
                        y=list(polymer_counts.values()),
                        labels={"x": "", "y": "spectra"},
                        color_discrete_sequence=series,
                    ),
                ),
                width="stretch",
                config={"displayModeBar": False},
            )

        with st.expander("Per-particle table"):
            st.dataframe(
                [
                    {
                        "id": record["particle_id"],
                        "class": record["class"],
                        "detection_confidence": record["detection_confidence"],
                        "morphology": record["morphology"],
                        "morphology_confidence": round(
                            float(record["morphology_confidence"]), 4
                        ),
                        "area_px2": round(
                            float(record["bbox"][2]) * float(record["bbox"][3])
                        ),
                        "bbox (xywh)": record["bbox"],
                    }
                    for record in records
                ],
                width="stretch",
                hide_index=True,
            )
    else:
        with st.container(border=True):
            st.markdown("#### Nothing to summarise")
            st.caption("Run an image analysis on the Image tab to populate this page.")
        if spectral_active and spectrum_results:
            # No image records, so `series` was never bound above -- fall back to
            # the palette's own series colours for this branch.
            import plotly.express as px

            polymer_counts = Counter(r["polymer"] for r in spectrum_results)
            st.subheader("Polymer (Stage 2B)")
            st.plotly_chart(
                _themed(
                    px.bar(
                        x=list(polymer_counts.keys()),
                        y=list(polymer_counts.values()),
                        labels={"x": "", "y": "spectra"},
                        color_discrete_sequence=series_colours(),
                    ),
                ),
                width="stretch",
            )

    with st.container(border=True):
        st.subheader("Reports")
        if records:
            import json

            payload: dict = {"particles": records}
            if spectral_active and spectrum_results:
                payload["spectra"] = spectrum_results
            left, right = st.columns(2)
            with left:
                st.download_button(
                    "Download JSON",
                    json.dumps(payload, indent=2),
                    file_name="report.json",
                    mime="application/json",
                    width="stretch",
                )
            with right:
                st.download_button(
                    "Download PDF",
                    build_report_pdf(records),
                    file_name="report.pdf",
                    mime="application/pdf",
                    width="stretch",
                )
        else:
            st.caption("Reports appear once an image has been analysed.")
