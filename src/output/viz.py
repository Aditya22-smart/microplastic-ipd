"""Derived statistics and plain-language findings for the dashboard.

Kept out of ``dashboard.py`` on purpose: everything here is a pure function of the
record list, so it can be unit-tested without booting Streamlit, and the
arithmetic that produces a headline number can be checked directly.

Nothing in this module invents data. Every value it reports is computed from the
frozen 8-key records the pipeline produced -- see
:data:`src.models.full_pipeline.RECORD_KEYS`. Where a number would be misleading
without information the pipeline does not have (particle size in microns, for
example), the caveat is returned alongside the number rather than dropped.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

#: Bins for the detection-confidence histogram. Fixed rather than derived from
#: the data so the same run always produces the same chart, and so the width of a
#: bin stays comparable between runs.
CONFIDENCE_BINS: tuple[tuple[float, float, str], ...] = (
    (0.50, 0.60, "0.50-0.60"),
    (0.60, 0.70, "0.60-0.70"),
    (0.70, 0.80, "0.70-0.80"),
    (0.80, 0.90, "0.80-0.90"),
    (0.90, 1.01, "0.90-1.00"),
)

#: Two detections whose centres are within this fraction of the image diagonal
#: are counted as touching, which is the signal that a "particle count" may
#: actually be one particle fragmented into several boxes.
_TOUCH_FRACTION = 0.02


def particle_areas(records: list[dict[str, Any]]) -> list[float]:
    """Bounding-box area per record, in pixels squared.

    Pixels, deliberately. The images carry no calibration metadata, so a
    micron figure would be invented rather than measured.
    """
    return [float(record["bbox"][2]) * float(record["bbox"][3]) for record in records]


def particle_centroids(records: list[dict[str, Any]]) -> list[tuple[float, float]]:
    """Bounding-box centres as ``(x, y)`` in pixels."""
    centres: list[tuple[float, float]] = []
    for record in records:
        x, y, width, height = (float(v) for v in record["bbox"])
        centres.append((x + width / 2, y + height / 2))
    return centres


def confidence_histogram(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count particles per fixed confidence band.

    A flat block in the lowest band is the thing worth noticing: it means the
    count is riding on detections the model was barely sure about, and the
    number would move a lot if the floor moved with it.
    """
    counts = Counter({label: 0 for _, _, label in CONFIDENCE_BINS})
    for record in records:
        value = float(record["detection_confidence"])
        for low, high, label in CONFIDENCE_BINS:
            if low <= value < high:
                counts[label] += 1
                break
    return dict(counts)


def marginal_fraction(records: list[dict[str, Any]], ceiling: float = 0.70) -> float:
    """Share of particles detected below ``ceiling`` confidence.

    Reported because "12 particles" and "12 particles, 10 of them marginal" are
    different claims, and only the first one reads well in a headline.
    """
    if not records:
        return 0.0
    marginal = sum(1 for r in records if float(r["detection_confidence"]) < ceiling)
    return marginal / len(records)


def median(values: list[float]) -> float:
    """Median, or ``0.0`` for an empty list. Avoids a numpy import for one call."""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def nearest_neighbour_ratio(records: list[dict[str, Any]], image_size) -> float:
    """Fraction of particles whose closest neighbour sits within touching range.

    A high value means the boxes overlap, so the particle count likely
    double-counts a single object. Uses the real image diagonal rather than a
    pixel constant, so it means the same thing for a 640x480 and a 2048x2048
    micrograph.
    """
    if len(records) < 2:
        return 0.0
    width, height = image_size
    diagonal = math.hypot(float(width), float(height)) or 1.0
    limit = _TOUCH_FRACTION * diagonal
    centres = particle_centroids(records)
    touching = 0
    for index, (x, y) in enumerate(centres):
        nearest = min(
            math.hypot(x - ox, y - oy)
            for other, (ox, oy) in enumerate(centres)
            if other != index
        )
        if nearest < limit:
            touching += 1
    return touching / len(centres)


def coverage_fraction(records: list[dict[str, Any]], image_size) -> float:
    """Share of the frame covered by bounding boxes, capped at 1.0.

    Boxes overlap, so the summed areas can exceed the frame; clamping keeps this
    a fraction rather than letting it print "342% of the field", which would be
    a bug in the dashboard rather than a finding.
    """
    if image_size[0] <= 0 or image_size[1] <= 0:
        return 0.0
    total = float(image_size[0]) * float(image_size[1])
    return min(1.0, sum(particle_areas(records)) / total)


def field_stats(
    records: list[dict[str, Any]], image_size: tuple[int, int], conf_threshold: float
) -> dict[str, Any]:
    """Everything the Summary tab reports, in one dictionary.

    ``image_size`` is ``(width, height)`` of the uploaded micrograph, used for the
    frame-relative figures only.
    """
    if not records:
        return {"count": 0}

    detection = [float(r["detection_confidence"]) for r in records]
    morphology = [float(r["morphology_confidence"]) for r in records]
    areas = particle_areas(records)
    morphology_counts = Counter(r["morphology"] for r in records)
    class_counts = Counter(r["class"] for r in records)
    dominant, dominant_n = morphology_counts.most_common(1)[0]

    return {
        "count": len(records),
        "mean_detection": sum(detection) / len(records),
        "median_detection": median(detection),
        "min_detection": min(detection),
        "mean_morphology": sum(morphology) / len(records),
        "median_morphology": median(morphology),
        "morphology_counts": dict(morphology_counts),
        "class_counts": dict(class_counts),
        "dominant_morphology": dominant,
        "dominant_share": dominant_n / len(records),
        "area_median": median(areas),
        "area_min": min(areas),
        "area_max": max(areas),
        "coverage": coverage_fraction(records, image_size),
        "touching": nearest_neighbour_ratio(records, image_size),
        "marginal_share": marginal_fraction(records),
        "histogram": confidence_histogram(records),
        "centroids": particle_centroids(records),
        "conf_threshold": conf_threshold,
    }


def _verdict(stats: dict[str, Any]) -> str:
    """One word for the overall call, so the reader is not left to infer it."""
    if stats["median_detection"] >= 0.80 and stats["marginal_share"] <= 0.25:
        return "high confidence"
    if stats["median_detection"] >= 0.65:
        return "moderate confidence"
    return "low confidence"


def interpret(stats: dict[str, Any]) -> list[str]:
    """Plain-language findings for the Summary tab.

    Returns markdown bullets. Every clause traces to a number in ``stats`` --
    there is deliberately no "looks good" / "needs work" judgement here, because
    a single micrograph with no ground-truth labels cannot support one. What it
    can support is a description of how much the model committed, and that is
    what this says.
    """
    if not stats.get("count"):
        return ["No particles were reported, so there is nothing to interpret."]

    lines = [
        f"**{stats['count']} particle(s) detected** — median detection confidence "
        f"{stats['median_detection']:.2f}, so this field is a "
        f"**{_verdict(stats)}** result."
    ]

    share = stats["dominant_share"]
    mix = (
        "a single morphology across the field"
        if share >= 0.95
        else "a mixed population" if share <= 0.60 else "one morphology in the majority"
    )
    lines.append(
        f"**{stats['dominant_morphology']}** accounts for {share:.0%} of detections "
        f"— {mix}."
    )

    if stats["marginal_share"] > 0:
        lines.append(
            f"{stats['marginal_share']:.0%} of detections fall below 0.70 confidence, "
            "so the count would shrink if the floor were raised."
        )
    else:
        lines.append("Every detection cleared 0.70 confidence.")

    lines.append(
        f"Detected boxes cover {stats['coverage']:.1%} of the frame "
        f"(median {stats['area_median']:.0f} px², "
        f"{stats['area_min']:.0f}-{stats['area_max']:.0f} px²). Sizes are in pixels: "
        "the images carry no calibration metadata, so a micron figure is not "
        "available from this pipeline."
    )

    if stats["touching"] >= 0.5:
        lines.append(
            f"{stats['touching']:.0%} of particles sit within touching distance of a "
            "neighbour. The boxes may be fragmenting one object, so this count is "
            "an upper bound."
        )

    lines.append(
        "Morphology comes from a MobileNetV3 classifier at "
        f"{stats['mean_morphology']:.2f} mean confidence; it is a shape label, not a "
        "polymer identification."
    )
    return lines
