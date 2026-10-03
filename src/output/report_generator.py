"""PDF report generation for pipeline results."""

from __future__ import annotations

import io
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def build_report_pdf(records: list[dict], title: str = "Microplastic Analysis Report") -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph(title, styles["Title"]),
        Spacer(1, 12),
        Paragraph(f"Generated: {datetime.now().isoformat(timespec='seconds')}", styles["Normal"]),
        Paragraph(f"Total particles: {len(records)}", styles["Normal"]),
        Spacer(1, 12),
    ]

    if records:
        morphology_counts = Counter(record.get("morphology") for record in records if record.get("morphology"))
        story.append(Paragraph("Morphology distribution", styles["Heading2"]))
        table_data = [["morphology", "count"]] + [[k, str(v)] for k, v in sorted(morphology_counts.items())]
        story.append(_make_table(table_data))
        story.append(Spacer(1, 12))

        polymer_counts = Counter(record.get("polymer") for record in records if record.get("polymer"))
        if polymer_counts:
            story.append(Paragraph("Polymer distribution", styles["Heading2"]))
            table_data = [["polymer", "count"]] + [[k, str(v)] for k, v in sorted(polymer_counts.items())]
            story.append(_make_table(table_data))
            story.append(Spacer(1, 12))

        story.append(Paragraph("Per-particle detail", styles["Heading2"]))
        detail = [["id", "class", "det_conf", "morphology", "morph_conf", "polymer"]]
        for record in records:
            detail.append(
                [
                    str(record.get("particle_id", "")),
                    str(record.get("class", "")),
                    f"{record.get('detection_confidence') or 0:.2f}",
                    str(record.get("morphology", "")),
                    f"{record.get('morphology_confidence') or 0:.2f}",
                    str(record.get("polymer", "")),
                ]
            )
        story.append(_make_table(detail))

    doc.build(story)
    return buffer.getvalue()


def _make_table(table_data):
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors

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
