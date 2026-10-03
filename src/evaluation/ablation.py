"""Model comparison tables for the report."""

from __future__ import annotations

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "classification" / "ablation_table.csv"


def write_ablation_table(rows: list[dict], out: Path = DEFAULT_OUT) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "morphology_f1", "polymer_f1", "avg_f1"]
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["avg_f1"] = round((row["morphology_f1"] + row["polymer_f1"]) / 2, 4)
            writer.writerow(row)
    return out
