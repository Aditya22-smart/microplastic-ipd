"""PDF report generation for pipeline results."""

from __future__ import annotations

import io
from collections import Counter
from datetime import datetime
from pathlib import Path

from src.config import spectral_enabled

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_report_pdf(
    records: list[dict], title: str = "Microplastic Analysis Report"
) -> bytes:
    """Render ``records`` to PDF bytes.

    The polymer column and distribution are omitted when the spectral branch is
    switched off, so an image-only run produces a report that does not imply
    polymer results that were never computed.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
    )

    include_polymer = spectral_enabled()
    is_pure_spectral = (
        bool(records)
        and all(r.get("morphology") is None for r in records)
        and any(r.get("polymer") is not None for r in records)
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    count_label = "Total spectra" if is_pure_spectral else "Total particles"
    mode_text = (
        "Pipeline mode: spectral only (chemical polymer identification)"
        if is_pure_spectral
        else (
            "Pipeline mode: visual only (spectral branch disabled)"
            if not include_polymer
            else "Pipeline mode: visual + spectral"
        )
    )

    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 12),
        Paragraph(
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            styles["Normal"],
        ),
        Paragraph(f"{count_label}: {len(records)}", styles["Normal"]),
        Paragraph(mode_text, styles["Normal"]),
        Spacer(1, 12),
    ]

    if records:
        if is_pure_spectral:
            polymer_counts = Counter(
                record.get("polymer") for record in records if record.get("polymer")
            )
            story.append(Paragraph("Polymer distribution", styles["Heading2"]))
            table_data = [["polymer", "count"]] + [
                [k, str(v)] for k, v in sorted(polymer_counts.items())
            ]
            story.append(_make_table(table_data))
            story.append(Spacer(1, 12))

            story.append(Paragraph("Per-spectrum detail", styles["Heading2"]))
            detail = [["id", "polymer", "confidence"]]
            for record in records:
                conf = record.get("polymer_confidence")
                conf_str = f"{float(conf):.4f}" if conf is not None else "n/a"
                detail.append([
                    str(record.get("particle_id", "")),
                    str(record.get("polymer", "")),
                    conf_str,
                ])
            story.append(_make_table(detail))
        else:
            morphology_counts = Counter(
                record.get("morphology") for record in records if record.get("morphology")
            )
            story.append(Paragraph("Morphology distribution", styles["Heading2"]))
            table_data = [["morphology", "count"]] + [
                [k, str(v)] for k, v in sorted(morphology_counts.items())
            ]
            story.append(_make_table(table_data))
            story.append(Spacer(1, 12))

            polymer_counts = Counter(
                record.get("polymer") for record in records if record.get("polymer")
            )
            if include_polymer and polymer_counts:
                story.append(Paragraph("Polymer distribution", styles["Heading2"]))
                table_data = [["polymer", "count"]] + [
                    [k, str(v)] for k, v in sorted(polymer_counts.items())
                ]
                story.append(_make_table(table_data))
                story.append(Spacer(1, 12))

            story.append(Paragraph("Per-particle detail", styles["Heading2"]))
            header = ["id", "class", "det_conf", "morphology", "morph_conf"]
            if include_polymer:
                header.append("polymer")
            detail = [header]
            for record in records:
                row = [
                    str(record.get("particle_id", "")),
                    str(record.get("class", "")),
                    f"{record.get('detection_confidence') or 0:.2f}",
                    str(record.get("morphology", "")),
                    f"{record.get('morphology_confidence') or 0:.2f}",
                ]
                if include_polymer:
                    row.append(str(record.get("polymer", "")))
                detail.append(row)
            story.append(_make_table(detail))

    doc.build(story)
    return buffer.getvalue()


def _make_table(table_data):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    table = Table(table_data)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]
        )
    )
    return table
