"""Streamlit dashboard for the microplastic pipeline."""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st

from src.models.full_pipeline import analyze_image, analyze_spectrum
from src.output.report_generator import build_report_pdf

st.set_page_config(page_title="Microplastic IPD", layout="wide")
st.title("Automated Microplastic Detection & Classification")

tab_image, tab_spectrum, tab_summary = st.tabs(["Image", "Spectrum", "Summary"])

with tab_image:
    uploaded = st.file_uploader("Upload a microscope image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        from PIL import Image

        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="input", use_container_width=True)
        tmp = Path("results") / "dashboard_upload.jpg"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        image.save(tmp)
        try:
            records = analyze_image(tmp)
            st.subheader(f"{len(records)} particle(s) detected")
            for record in records:
                st.write(
                    f"- id {record['particle_id']}: {record['class']} "
                    f"({record['detection_confidence']:.2f}), "
                    f"morphology {record['morphology']} "
                    f"({record['morphology_confidence']:.2f})"
                )
            st.session_state["records"] = records
        except FileNotFoundError as exc:
            st.error(str(exc))

with tab_spectrum:
    uploaded_spectrum = st.file_uploader("Upload a spectrum (CSV/txt)", type=["csv", "txt"])
    if uploaded_spectrum is not None:
        try:
            import numpy as np

            spectrum = np.loadtxt(io.StringIO(uploaded_spectrum.getvalue().decode("utf-8")), delimiter=",")
            result = analyze_spectrum(spectrum)
            st.metric("polymer", result["polymer"])
            st.metric("confidence", f"{result['polymer_confidence']:.3f}")
            st.session_state["spectrum_result"] = result
        except Exception as exc:
            st.error(f"could not classify spectrum: {exc}")

with tab_summary:
    records = st.session_state.get("records", [])
    if records:
        from collections import Counter

        import plotly.express as px

        morphology_counts = Counter(record["morphology"] for record in records)
        st.write(f"total particles: {len(records)}")
        st.plotly_chart(
            px.pie(names=list(morphology_counts.keys()), values=list(morphology_counts.values()), title="Morphology"),
            use_container_width=True,
        )
    else:
        st.info("Run an image analysis to see the summary.")

    if st.button("Download JSON report"):
        import json

        st.download_button(
            "report.json",
            json.dumps({"particles": records}, indent=2),
            mime="application/json",
        )

    if st.button("Download PDF report"):
        buffer = build_report_pdf(records)
        st.download_button("report.pdf", buffer, mime="application/pdf")
