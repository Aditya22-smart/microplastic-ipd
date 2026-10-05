"""Tests for the PDF report generator and the feature-flag gating around it.

The gate matters: a visual-only run must not print a polymer column, because a
report that shows an empty polymer field implies a classification that was never
computed. Assertions read the PDF's actual text rather than trusting the code
path, since a stale or re-ordered section would pass a naive flag check.
"""

from __future__ import annotations

import base64
import io
import re
import zlib

import pytest

from src.config import SPECTRAL_ENABLED_ENV, set_spectral
from src.output.report_generator import build_report_pdf

RECORDS = [
    {
        "particle_id": 1,
        "bbox": [10, 20, 30, 40],
        "detection_confidence": 0.91,
        "class": "bead",
        "morphology": "sphere",
        "morphology_confidence": 0.80,
        "polymer": "PE",
        "polymer_confidence": 0.70,
    },
    {
        "particle_id": 2,
        "bbox": [50, 60, 70, 80],
        "detection_confidence": 0.72,
        "class": "foam",
        "morphology": "film",
        "morphology_confidence": 0.65,
        "polymer": "PP",
        "polymer_confidence": 0.55,
    },
]


def pdf_text(pdf: bytes) -> str:
    """Extract the text from a ReportLab PDF.

    ReportLab's default page filter is ``[/ASCII85Decode /FlateDecode]``, so a
    stream has to be ASCII85-decoded *before* it can be inflated. Skipping that
    step yields printable garbage that looks like a plausible failure, which is
    how this helper first reported that "Morphology distribution" was missing
    from a report that plainly contained it. Each stage is therefore optional,
    in the order ReportLab applies them.
    """
    chunks: list[str] = []
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", pdf, re.S):
        raw = match.group(1).strip()
        for decode in (
            lambda b: zlib.decompress(base64.a85decode(b, adobe=True)),
            lambda b: zlib.decompress(b),
            lambda b: base64.a85decode(b, adobe=True),
            lambda b: b,
        ):
            try:
                chunks.append(decode(raw).decode("latin-1"))
                break
            except Exception:  # noqa: BLE001 - try the next filter combination
                continue
    return "\n".join(chunks)


@pytest.fixture(autouse=True)
def _deterministic_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SPECTRAL_ENABLED_ENV, raising=False)


def test_produces_a_valid_pdf_header() -> None:
    assert build_report_pdf(RECORDS).startswith(b"%PDF-")


def test_empty_record_list_does_not_crash() -> None:
    """A zero-detection run is a legitimate outcome, not an error."""
    pdf = build_report_pdf([])
    assert pdf.startswith(b"%PDF-")
    assert b"Total particles: 0" in pdf or "Total particles: 0" in pdf_text(pdf)


def test_visual_only_report_omits_polymer_entirely() -> None:
    """The headline assertion: no polymer text anywhere when the branch is off."""
    set_spectral(False)
    text = pdf_text(build_report_pdf(RECORDS))
    assert "polymer" not in text.lower()
    assert "Polymer distribution" not in text
    assert "visual only" in text


def test_spectral_report_includes_the_polymer_section() -> None:
    set_spectral(True)
    text = pdf_text(build_report_pdf(RECORDS))
    assert "Polymer distribution" in text
    assert "polymer" in text.lower()
    assert "visual + spectral" in text


def test_morphology_section_is_always_present() -> None:
    for enabled in (False, True):
        set_spectral(enabled)
        text = pdf_text(build_report_pdf(RECORDS))
        assert "Morphology distribution" in text, f"missing with spectral={enabled}"


def test_detail_column_count_follows_the_flag() -> None:
    """Off: 5 columns. On: 6 columns including polymer."""
    set_spectral(False)
    off = pdf_text(build_report_pdf(RECORDS))
    set_spectral(True)
    on = pdf_text(build_report_pdf(RECORDS))
    for header in ("id", "class", "det_conf", "morphology", "morph_conf"):
        assert header in off
    assert "polymer" not in off
    assert "polymer" in on


def test_title_is_honoured() -> None:
    text = pdf_text(build_report_pdf(RECORDS, title="Field Sample 42"))
    assert "Field Sample 42" in text


def test_missing_confidence_fields_do_not_crash() -> None:
    """Records from an older format must still render."""
    sparse = [{"particle_id": 1, "class": "bead", "morphology": "sphere"}]
    assert build_report_pdf(sparse).startswith(b"%PDF-")


def test_report_is_reproducible_in_structure() -> None:
    set_spectral(False)
    first = pdf_text(build_report_pdf(RECORDS))
    second = pdf_text(build_report_pdf(RECORDS))
    # The embedded timestamp differs between calls; compare everything else.
    strip = lambda text: re.sub(r"\d{4}-\d{2}-\d{2}T[\d:]+", "", text)  # noqa: E731
    assert strip(first) == strip(second)


def test_bytes_are_returned_not_a_file_handle() -> None:
    pdf = build_report_pdf(RECORDS)
    assert isinstance(pdf, bytes)
    io.BytesIO(pdf).read(1)  # consumable as a stream
