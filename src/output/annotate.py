"""Draw detection boxes and labels onto a microscope image.

The dashboard needs to *show* what the detector found, not just list it. This
module keeps that drawing logic out of ``dashboard.py`` so it can be unit-tested
without Streamlit.

Records are the 8-key dicts from :func:`src.models.full_pipeline.analyze_image`;
only ``bbox``, ``class``, ``detection_confidence``, ``morphology`` and
``morphology_confidence`` are read here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# One colour per canonical detection class (see
# ``src.detection.inference.CANONICAL_CLASSES``). Deliberately saturated and
# distinct from the Streamlit theme colour (teal) so the chrome never competes
# with the annotation the viewer is meant to read first.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "bead": (220, 60, 60),
    "foam": (40, 170, 90),
    "microplastic": (40, 110, 230),
}
_FALLBACK_COLOR = (240, 140, 20)

# Label palettes, picked per image by background luminance. See _label_palette.
_LIGHT_LABEL: tuple[tuple[int, int, int], tuple[int, int, int]] = (
    (255, 255, 255),
    (20, 20, 20),
)
_DARK_LABEL: tuple[tuple[int, int, int], tuple[int, int, int]] = (
    (28, 28, 32),
    (238, 240, 245),
)

# Probed in order; the first readable path wins. If none exist we fall back to
# PIL's built-in bitmap font rather than failing, so the annotation still renders
# on a bare container.
_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
)
_font_cache: dict[int, Any] = {}


def color_for(class_name: str) -> tuple[int, int, int]:
    """Return the drawing colour for a detection class."""
    return CLASS_COLORS.get(str(class_name).strip().lower(), _FALLBACK_COLOR)


def to_hex(color: tuple[int, int, int]) -> str:
    """Convert an RGB triple to a ``#RRGGBB`` string for HTML legends.

    Clamped rather than formatted blind: this value is interpolated into a
    ``<span style="color:...">`` that the dashboard renders as raw HTML, so a
    malformed triple must never be able to inject markup through it.
    """
    r, g, b = (max(0, min(255, int(channel))) for channel in color)
    return f"#{r:02X}{g:02X}{b:02X}"


def class_legend(
    records: list[dict[str, Any]],
) -> list[tuple[str, tuple[int, int, int]]]:
    """Return ``(class name, colour)`` for each class actually present.

    Only classes that appear in ``records`` are listed, so the legend never
    advertises a colour the viewer cannot see on the image.
    """
    present = {
        str(record.get("class", "")).strip().lower()
        for record in records
        if record.get("class")
    }
    return [
        (name, color) for name, color in CLASS_COLORS.items() if name in present
    ] or ([("other", _FALLBACK_COLOR)] if present else [])


def _load_font(size: int):
    """Return a bold TrueType font at ``size``, or PIL's bitmap default.

    The bitmap default is an 11px fixed face; stretched to dashboard width it
    reads as pixelated and unfinished, so a real font is strongly preferred.
    """
    if size in _font_cache:
        return _font_cache[size]
    from PIL import ImageFont

    font = None
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            try:
                font = ImageFont.truetype(candidate, size)
                break
            except OSError:  # unreadable font file; try the next candidate
                continue
    if font is None:
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def _font_size(image_width: int) -> int:
    """Scale the label font to the image, capped so huge frames stay readable.

    Proportional sizing keeps labels legible on a 4K microscope frame without
    rendering them comically large; the bitmap default is sized for a 640px
    image.
    """
    return max(11, min(32, round(image_width / 48)))


def _mean_luminance(image) -> float:
    """Mean 0-255 luminance of an image, from its histogram."""
    histogram = image.convert("L").histogram()
    total = sum(histogram)
    if not total:
        return 255.0
    return sum(level * count for level, count in enumerate(histogram)) / total


def _label_palette(image) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Choose (background, text) for label chips.

    Microplastic fields are shot dark-field (bright particles on a near-black
    background) as often as bright-field. A hardcoded white chip glares on a dark
    frame and swallows the particle it is labelling, so the palette follows the
    image instead.
    """
    return _LIGHT_LABEL if _mean_luminance(image) > 128 else _DARK_LABEL


def _label_text(record: dict[str, Any], show_morphology: bool) -> str:
    """Compose the caption drawn next to a box."""
    confidence = record.get("detection_confidence")
    text = str(record.get("class", ""))
    if confidence is not None:
        text = f"{text} {float(confidence):.2f}"
    if show_morphology:
        morphology = record.get("morphology")
        if morphology:
            morphology_confidence = record.get("morphology_confidence")
            if morphology_confidence is not None:
                text = f"{text} | {morphology} {float(morphology_confidence):.2f}"
            else:
                text = f"{text} | {morphology}"
    return text


def draw_detections(
    image,
    records: list[dict[str, Any]],
    *,
    min_confidence: float = 0.0,
    show_labels: bool = True,
    show_morphology: bool = True,
    line_width: int | None = None,
):
    """Return a copy of ``image`` with each record's box and label drawn on it.

    Args:
        image: a PIL Image (it is copied, never modified in place).
        records: 8-key records from ``analyze_image``.
        min_confidence: records below this detection confidence are skipped, so
            the sidebar slider can filter without a second inference pass.
        show_labels: draw the class/confidence caption.
        show_morphology: append the morphology class to the caption.
        line_width: box thickness; defaults to a fraction of image width so it
            scales with resolution instead of vanishing on small crops.

    Returns:
        A new PIL RGB Image. Returns the input unchanged when there is nothing
        to draw, so a zero-detection result still displays the original image.
    """
    from PIL import ImageDraw

    visible = [
        record
        for record in records
        if float(record.get("detection_confidence") or 0.0) >= min_confidence
    ]
    if not visible:
        return image

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    if line_width is None:
        line_width = max(2, round(canvas.width / 400))

    font = _load_font(_font_size(canvas.width))
    label_background, label_foreground = _label_palette(canvas)
    pad = max(2, line_width)

    for record in visible:
        bbox = record.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, width, height = (int(round(float(v))) for v in bbox)
        color = color_for(record.get("class", ""))
        draw.rectangle([x, y, x + width, y + height], outline=color, width=line_width)

        if not show_labels:
            continue
        text = _label_text(record, show_morphology)
        if not text:
            continue

        text_box = draw.textbbox((0, 0), text, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        chip_width = text_width + 2 * pad
        chip_height = text_height + 2 * pad

        # Anchor the chip inside the frame when there is no room above it, so
        # labels on particles near the top edge stay readable.
        chip_y = y - chip_height if y - chip_height >= 0 else y
        # Clamp horizontally too: a long caption ("microplastic 0.91 | fragment
        # 0.88") on a particle near the right edge would otherwise be drawn
        # outside the canvas and silently clipped mid-word.
        chip_x = min(x, max(0, canvas.width - chip_width))
        chip_y = min(chip_y, max(0, canvas.height - chip_height))

        draw.rectangle(
            [chip_x, chip_y, chip_x + chip_width, chip_y + chip_height],
            fill=label_background,
        )
        draw.text(
            (chip_x + pad, chip_y + pad),
            text,
            fill=label_foreground,
            font=font,
        )

    return canvas
