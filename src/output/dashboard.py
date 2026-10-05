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


def _legend_strip(records: list[dict]) -> None:
    """Render the colour key for the annotated image.

    Without this the viewer sees red, green and blue boxes with nothing telling
    them what the colours mean.
    """
    items = class_legend(records)
    if not items:
        return
    counts = Counter(record["class"] for record in records)
    swatches = "&nbsp;&nbsp;&nbsp;".join(
        f'<span style="color:{to_hex(color)};font-size:1.15em">&#9632;</span> '
        f"&nbsp;{name} ({counts.get(name, 0)})"
        for name, color in items
    )
    st.markdown(swatches)


# --- feature flag: sidebar toggle, seeded from the environment ---------------
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
            st.image(
                annotated,
                caption=(
                    f"{len(records)} particle(s) via "
                    f"{MODEL_LABELS.get(model_name, model_name)} "
                    f"at conf ≥ {conf_threshold:.2f}"
                ),
                width="stretch",
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
        detection_counts = Counter(record["class"] for record in records)
        morphology_counts = Counter(record["morphology"] for record in records)
        mean_detection = sum(
            float(record["detection_confidence"]) for record in records
        ) / len(records)
        mean_morphology = sum(
            float(record["morphology_confidence"]) for record in records
        ) / len(records)
        dominant = morphology_counts.most_common(1)[0]

        tile = st.columns(4)
        tile[0].metric("Particles", len(records))
        tile[1].metric("Mean detection conf", f"{mean_detection:.3f}")
        tile[2].metric("Mean morphology conf", f"{mean_morphology:.3f}")
        tile[3].metric("Dominant morphology", dominant[0])

        st.caption(
            f"Detector: {MODEL_LABELS.get(st.session_state.get('model_name'), 'n/a')}"
        )
        _legend_strip(records)

        detection_col, morphology_col = st.columns(2)
        with detection_col:
            st.subheader("Detection class (Stage 1A)")
            st.bar_chart(dict(sorted(detection_counts.items())), width="stretch")
        with morphology_col:
            st.subheader("Morphology (Stage 1B)")
            st.bar_chart(dict(sorted(morphology_counts.items())), width="stretch")

        st.subheader("Morphology share")
        import plotly.express as px

        st.plotly_chart(
            px.pie(
                names=list(morphology_counts.keys()),
                values=list(morphology_counts.values()),
                title="",
            ),
            width="stretch",
        )

        if spectral_active and spectrum_results:
            polymer_counts = Counter(r["polymer"] for r in spectrum_results)
            st.subheader("Polymer (Stage 2B)")
            st.bar_chart(dict(sorted(polymer_counts.items())), width="stretch")
    else:
        with st.container(border=True):
            st.markdown("#### Nothing to summarise")
            st.caption("Run an image analysis on the Image tab to populate this page.")
        if spectral_active and spectrum_results:
            st.subheader("Polymer (Stage 2B)")
            polymer_counts = Counter(r["polymer"] for r in spectrum_results)
            st.bar_chart(dict(sorted(polymer_counts.items())), width="stretch")

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
