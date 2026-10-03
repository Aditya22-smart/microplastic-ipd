"""Download and organise morphology image datasets for the vision tower (Pipeline 1B).

Sources
-------
* **Moore Institute "Microplastic Image Explorer"** (OpenAnalysis ``One4All``):
  the image index lives in ``image_metadata.csv`` inside the ``One4All`` GitHub
  repo; the actual images are publicly served from a CloudFront bucket at
  ``https://d2jrxerjcsjhs7.cloudfront.net/<file_names>``.
* **PEESEgroup "Microplastic-Project"** (github.com/PEESEgroup/Microplastic-Project):
  PS/PE microbead images (PNG) used to supplement the ``sphere`` morphology
  class, as described in the IPD Project Guide (Section 3.1). NOTE: as of
  writing, the repo only ships 84 example images (6 SDS folders x 14 files);
  the README's full 846-image set lives behind a Google Drive link that is
  not scriptable. The 84 available images are downloaded automatically.

Only images that map to the five target morphology classes are kept:
``sphere``, ``fragment``, ``fiber``, ``film``, ``foam``.

Usage
-----
    python src/preprocessing/download_data.py moore --workers 12
    python src/preprocessing/download_data.py peese --workers 12
    python src/preprocessing/download_data.py organize
    python src/preprocessing/download_data.py all --workers 12

The ``moore`` and ``peese`` steps write into ``data/raw/``; ``organize``
builds the train/val/test split under ``data/processed/morphology/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parents[2]

MOORE_METADATA_URL = (
    "https://raw.githubusercontent.com/Moore-Institute-4-Plastic-Pollution-Res/"
    "One4All/main/inst/apps/microplastic_image_explorer/image_metadata.csv"
)
MOORE_CDN_BASE = "https://d2jrxerjcsjhs7.cloudfront.net/"

PEESE_REPO = "PEESEgroup/Microplastic-Project"
PEESE_BRANCH = "main"
PEESE_API_TREE_URL = (
    f"https://api.github.com/repos/{PEESE_REPO}/git/trees/{PEESE_BRANCH}?recursive=1"
)
PEESE_RAW_BASE = f"https://raw.githubusercontent.com/{PEESE_REPO}/{PEESE_BRANCH}/"

MORPHOLOGY_CLASSES: tuple[str, ...] = ("sphere", "fragment", "fiber", "film", "foam")

# Normalised morphology label -> target class. Anything not listed here is skipped.
MOORE_CLASS_MAP: dict[str, str] = {
    "sphere": "sphere",
    "fragment": "fragment",
    "fiber": "fiber",
    "fiber bundle": "fiber",
    "film": "film",
    "foam": "foam",
}

PEESE_IMAGE_SUBDIR = "Microplastic Image Dataset"
IMAGE_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

MOORE_RAW_DIR = REPO_ROOT / "data" / "raw" / "moore_institute"
PEESE_RAW_DIR = REPO_ROOT / "data" / "raw" / "peese_microbeads"
PROCESSED_MORPHOLOGY_DIR = REPO_ROOT / "data" / "processed" / "morphology"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEFAULT_SEED = 42
DEFAULT_WORKERS = 8
DEFAULT_RETRIES = 3
DOWNLOAD_TIMEOUT_SEC = 60


# --------------------------------------------------------------------------- #
# Small HTTP helpers
# --------------------------------------------------------------------------- #


def _http_bytes(url: str, retries: int = DEFAULT_RETRIES) -> bytes:
    """Fetch ``url`` and return its raw bytes, retrying transient failures."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "microplastic-ipd/1.0"},
            )
            with urllib.request.urlopen(  # nosec B310 - callers pass hardcoded dataset URLs
                request, timeout=DOWNLOAD_TIMEOUT_SEC
            ) as resp:
                return resp.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {url}: {last_error}")


def download_file(url: str, dest: Path, retries: int = DEFAULT_RETRIES) -> bool:
    """Download ``url`` to ``dest``. Returns True when the file is on disk.

    Skips already-downloaded files (non-empty) so reruns are cheap.
    Only ``https``/``http`` URLs are accepted; anything else (file://, ftp://,
    custom schemes) is rejected so a bad constant cannot read the local disk.
    """
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise ValueError(f"refusing to download non-http(s) URL: {url!r}")
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "microplastic-ipd/1.0"},
            )
            with (
                urllib.request.urlopen(  # nosec B310 - scheme validated above
                    request, timeout=DOWNLOAD_TIMEOUT_SEC
                ) as resp,
                dest.open("wb") as handle,  # nosec B310
            ):
                shutil.copyfileobj(resp, handle)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    print(f"  [warn] failed after {retries} tries: {url} ({last_error})")
    return False


# --------------------------------------------------------------------------- #
# Moore Institute (Pipeline 1B morphology images)
# --------------------------------------------------------------------------- #


def fetch_moore_metadata(cache_path: Path) -> list[dict[str, str]]:
    """Return the Moore Institute image index as a list of row dicts."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return list(csv.DictReader(cache_path.open(newline="", encoding="utf-8")))

    print(f"  fetch metadata: {MOORE_METADATA_URL}")
    raw = _http_bytes(MOORE_METADATA_URL).decode("utf-8")
    cache_path.write_text(raw, encoding="utf-8")
    return list(csv.DictReader(raw.splitlines()))


def select_moore_images(rows: Sequence[dict[str, str]]) -> list[tuple[str, str]]:
    """Map metadata rows to ``(class, file_name)`` pairs for target classes."""
    selected: list[tuple[str, str]] = []
    for row in rows:
        morphology = (row.get("morphology") or "").strip().lower()
        target_class = MOORE_CLASS_MAP.get(morphology)
        file_name = (row.get("file_names") or "").strip()
        if target_class is None or not file_name:
            continue
        selected.append((target_class, file_name))
    return selected


def download_moore(workers: int, limit: int | None) -> None:
    """Download Moore Institute morphology images into ``data/raw/moore_institute``."""
    metadata_path = MOORE_RAW_DIR / "image_metadata.csv"
    rows = fetch_moore_metadata(metadata_path)
    images = select_moore_images(rows)
    if limit is not None:
        images = images[:limit]

    print(f"Moore: {len(images)} images to download ({len(rows)} rows in metadata)")

    def fetch_one(item: tuple[str, str]) -> tuple[str, bool]:
        target_class, file_name = item
        dest = MOORE_RAW_DIR / target_class / file_name
        url = MOORE_CDN_BASE + urllib.parse.quote(file_name)
        return f"{target_class}/{file_name}", download_file(url, dest)

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_one, item) for item in images]
        for future in as_completed(futures):
            name, success = future.result()
            if success:
                ok += 1
            else:
                failed += 1
                print(f"  [fail] {name}")

    per_class = {"sphere": 0, "fragment": 0, "fiber": 0, "film": 0, "foam": 0}
    for target_class, _ in images:
        per_class[target_class] += 1
    print(f"Moore: downloaded/verified {ok}, failed {failed}")
    print(f"Moore: requested per class: {per_class}")
    print(f"Moore: saved under {MOORE_RAW_DIR}")


# --------------------------------------------------------------------------- #
# PEESEgroup microbeads (sphere supplement)
# --------------------------------------------------------------------------- #


def fetch_peese_image_paths() -> list[str]:
    """Return every image path under PEESEgroup's ``Microplastic Image Dataset``."""
    payload = json.loads(_http_bytes(PEESE_API_TREE_URL).decode("utf-8"))
    paths: list[str] = []
    for entry in payload.get("tree", []):
        path: str = entry.get("path", "")
        if entry.get("type") != "blob":
            continue
        if not path.startswith(PEESE_IMAGE_SUBDIR + "/"):
            continue
        if path.lower().endswith(IMAGE_EXTENSIONS):
            paths.append(path)
    return paths


def download_peese(workers: int, limit: int | None) -> None:
    """Download PEESEgroup microbead images into ``data/raw/peese_microbeads``.

    The images land under ``<raw>/sphere/<polymer>__<relative-path>`` so
    ``organize`` can treat them as the ``sphere`` morphology class.
    """
    paths = fetch_peese_image_paths()
    if limit is not None:
        paths = paths[:limit]

    print(f"PEESE: {len(paths)} microbead images to download")

    def fetch_one(path: str) -> tuple[str, bool]:
        # path looks like "Microplastic Image Dataset/PE/SDS 0.01mM/10 PE 3.1_10_7 13.png".
        # The first segment after the dataset folder is the polymer (PS or PE);
        # the raw filename already starts with the polymer, so no extra prefix is needed.
        relative = path[len(PEESE_IMAGE_SUBDIR) + 1 :]
        safe_name = relative.replace("\\", "__").replace("/", "__").replace(" ", "_")
        dest = PEESE_RAW_DIR / "sphere" / safe_name
        url = PEESE_RAW_BASE + urllib.parse.quote(path)
        return str(dest.name), download_file(url, dest)

    ok = failed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch_one, path) for path in paths]
        for future in as_completed(futures):
            name, success = future.result()
            if success:
                ok += 1
            else:
                failed += 1
                print(f"  [fail] {name}")

    print(f"PEESE: downloaded/verified {ok}, failed {failed}")
    print(f"PEESE: saved under {PEESE_RAW_DIR}/sphere/")


# --------------------------------------------------------------------------- #
# Organise into train/val/test split
# --------------------------------------------------------------------------- #


def collect_class_images() -> dict[str, list[tuple[Path, str]]]:
    """Collect a flat list of ``(source_path, display_name)`` per morphology class.

    Moore images map to their metadata class; PEESE beads map to ``sphere``.
    """
    collected: dict[str, list[tuple[Path, str]]] = {
        cls: [] for cls in MORPHOLOGY_CLASSES
    }

    if MOORE_RAW_DIR.exists():
        for target_class in MORPHOLOGY_CLASSES:
            class_dir = MOORE_RAW_DIR / target_class
            if not class_dir.is_dir():
                continue
            for image in sorted(class_dir.iterdir()):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                    collected[target_class].append((image, f"moore__{image.name}"))

    peese_sphere = PEESE_RAW_DIR / "sphere"
    if peese_sphere.is_dir():
        for image in sorted(peese_sphere.iterdir()):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                collected["sphere"].append((image, image.name))

    return collected


def _split_indices(
    n: int, rng: random.Random
) -> tuple[list[int], list[int], list[int]]:
    """Deterministic 70/15/15 split of ``range(n)``."""
    indices = list(range(n))
    rng.shuffle(indices)
    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)
    train = indices[:n_train]
    val = indices[n_train : n_train + n_val]
    test = indices[n_train + n_val :]
    return train, val, test


def organize_morphology(seed: int) -> None:
    """Build ``data/processed/morphology/{train,val,test}/<class>/`` and reports."""
    rng = random.Random(seed)
    collected = collect_class_images()

    total_per_class = {cls: len(files) for cls, files in collected.items()}
    grand_total = sum(total_per_class.values())
    print("Collected classes:")
    for cls in MORPHOLOGY_CLASSES:
        print(f"  {cls:<10} {total_per_class[cls]:>6}")

    if grand_total == 0:
        raise SystemExit("No images found — run 'moore' and/or 'peese' first.")

    manifest: dict[str, dict[str, list[str]]] = {}
    for cls, items in collected.items():
        train_idx, val_idx, test_idx = _split_indices(len(items), rng)
        manifest[cls] = {"train": [], "val": [], "test": []}

        for split_name, indices in (
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ):
            split_dir = PROCESSED_MORPHOLOGY_DIR / split_name / cls
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in indices:
                src, display = items[i]
                dest = split_dir / display
                if not dest.exists():
                    shutil.copy2(src, dest)
                manifest[cls][split_name].append(
                    str(dest.relative_to(PROCESSED_MORPHOLOGY_DIR))
                )

    # Class weights: w_i = total / (n_classes * count_i)
    class_weights = {
        cls: grand_total / (len(MORPHOLOGY_CLASSES) * total_per_class[cls])
        for cls in MORPHOLOGY_CLASSES
        if total_per_class[cls] > 0
    }
    (PROCESSED_MORPHOLOGY_DIR / "class_weights.json").write_text(
        json.dumps(class_weights, indent=2), encoding="utf-8"
    )
    (PROCESSED_MORPHOLOGY_DIR / "splits.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(
        f"Split ratios: train {TRAIN_RATIO} / val {VAL_RATIO} / test {TEST_RATIO} (seed {seed})"
    )
    print(
        f"Written class_weights.json and splits.json under {PROCESSED_MORPHOLOGY_DIR}"
    )
    print("Class weights (inverse frequency):")
    for cls, weight in class_weights.items():
        print(f"  {cls:<10} {weight:.3f}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="download_data",
        description="Download and organise morphology datasets for Pipeline 1B.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_moore = sub.add_parser(
        "moore", help="Download Moore Institute morphology images."
    )
    p_moore.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p_moore.add_argument(
        "--limit", type=int, default=None, help="Cap downloads (testing)."
    )

    p_peese = sub.add_parser(
        "peese", help="Download PEESEgroup microbead images (sphere)."
    )
    p_peese.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p_peese.add_argument(
        "--limit", type=int, default=None, help="Cap downloads (testing)."
    )

    p_organize = sub.add_parser(
        "organize", help="Build 70/15/15 split + class weights."
    )
    p_organize.add_argument("--seed", type=int, default=DEFAULT_SEED)

    p_all = sub.add_parser("all", help="Run moore, peese and organize in sequence.")
    p_all.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p_all.add_argument(
        "--limit", type=int, default=None, help="Cap downloads (testing)."
    )
    p_all.add_argument("--seed", type=int, default=DEFAULT_SEED)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in ("moore", "all"):
        download_moore(args.workers, args.limit)
    if args.command in ("peese", "all"):
        download_peese(args.workers, args.limit)
    if args.command in ("organize", "all"):
        organize_morphology(args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
