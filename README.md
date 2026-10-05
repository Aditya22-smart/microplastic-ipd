# Automated Microplastic Detection & Classification Using Deep Learning

## 🔬 Project Overview

An automated system that detects, localizes, and classifies microplastic particles. The system is built from **two independent deep-learning pipelines** that share one dashboard — a design that works because no open dataset pairs an image and a spectrum for the same particle:

1. **Pipeline 1 — Visual (image in):**
   - **YOLO detection (YOLO26n primary, YOLOv8n baseline)** localizes microplastic particles and crops each one.
   - **MobileNetV3-Small** classifies each crop's morphology → *sphere / fragment / fiber / film / foam*.
2. **Pipeline 2 — Spectral (spectrum in):**
   - FTIR/Raman preprocessing (Savitzky-Golay smoothing → water-vapour band zeroing → SNV normalization).
   - **1D-CNN** classifies the polymer type → *PE / PP / PS / PMMA / PAN*.
3. **Dashboard — Streamlit:** upload an image or spectrum, see detection boxes, morphology and polymer labels, particle counts, JSON export, and a downloadable PDF report.

### Three Research Hypotheses
| # | Hypothesis | Test |
|---|---|---|
| H1 | YOLO26 mAP@0.5 > YOLOv8 baseline on microplastic detection | paired t-test, 5 seeds |
| H2 | MobileNetV3 fine-tuned > from-scratch on morphology | McNemar's test |
| H3 | 1D-CNN generalizes to weathered spectra (FLOPP-e) better than SVM | Wilcoxon signed-rank |

## 🛠️ Tools & Technologies
- **Deep Learning:** PyTorch, Ultralytics (YOLOv8, YOLO26)
- **Vision:** Albumentations, timm (MobileNetV3); Grad-CAM implemented in-repo
- **Spectroscopy:** scipy, spectral, scikit-learn (SVM baseline)
- **Stats:** scipy.stats, statsmodels (McNemar)
- **Tracking:** Weights & Biases (wandb)
- **Output:** Streamlit, Plotly, ReportLab
- **Quality gates:** Black, Ruff, Mypy, Bandit (CI)
- **Environment:** uv (locked `pyproject.toml` + `uv.lock`)

## 📂 Folder Structure
```text
microplastic-ipd/
├── configs/                  # YOLO + shared training hyperparameters
├── pyproject.toml            # Dependencies + tool config (authoritative)
├── uv.lock                   # Exact resolved versions (committed)
├── requirements.txt          # Colab-only mirror of pyproject (loose ranges)
├── data/
│   ├── raw/                  # (Gitignored) Original datasets (Moore, PEESE, Zenodo, ...)
│   └── processed/            # (Gitignored) Morphology splits, spectra .npy, scaler.pkl
├── notebooks/                # EDA + Colab training notebooks (01–05)
├── results/                  # Tables, confusion matrices, Grad-CAMs, hypothesis JSONs
├── src/
│   ├── detection/            # YOLO training, compare_yolo, inference.detect_and_crop
│   ├── preprocessing/        # download_data, merge_datasets, image/spectral preprocess
│   ├── models/               # vision_tower, spectral_tower, predict_morphology, predict_polymer, full_pipeline
│   ├── training/             # train_morphology, train_spectral, metrics
│   ├── evaluation/           # hypothesis_test (H1/H2/H3), grad_cam_viz, ablation
│   └── output/               # Streamlit dashboard + ReportLab PDF
├── tests/                    # Pytest unit tests
└── weights/                  # (Gitignored) Trained checkpoints (.pt / .pth)
```

## 🚀 Getting Started

Local development uses **[uv](https://docs.astral.sh/uv/)**. It resolves from
`pyproject.toml` + `uv.lock`, so everyone gets byte-identical versions and one
command sets up the environment.

```bash
# 1. Install uv (skip if you already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create the venv and install everything from the lock file
uv sync

# 3. Quality gates (dev group is installed by `uv sync` by default)
uv run black --check src tests
uv run ruff check src tests
uv run pytest
```

Run any script inside that environment — no need to activate anything:

```bash
uv run python -m src.training.train_morphology --seed 42
uv run streamlit run src/output/dashboard.py
```

> **No GPU on your machine?** That is fine and expected. `uv sync` installs the
> **CPU** build of PyTorch (~200 MB instead of ~2.5 GB). Training runs on Colab;
> your laptop only runs inference and the dashboard. See
> [Colab](#-google-colab--vs-code-integration-train-on-colab-code-locally).

> **On Colab?** Keep using `pip install -r requirements.txt`. Do **not** use
> `uv sync` there — Colab supplies its own CUDA build of PyTorch, and the local
> lock file pins the CPU wheel.

### Verify your environment

Run this after cloning, and again if anything behaves strangely. It catches the
three failure modes that actually bite: a stale lock file, a missing package, and
a `requirements.txt` that has drifted from `pyproject.toml`.

```bash
uv sync --frozen     # installs exactly what uv.lock pins; fails if it is stale
uv lock --check      # confirms pyproject.toml and uv.lock agree
uv pip check         # confirms no installed package conflicts with another
uv run pytest tests/test_dependencies.py -v   # every declared package importable
```

`tests/test_dependencies.py` is the repeatable version of this: it asserts the
two dependency files list the same packages at the same version floors, that
nothing is pinned with `==` (which would fight Colab's CUDA wheels), and that
every declared distribution actually imports.

### Checkpoints

The trained detector and morphology checkpoints are committed under `weights/`
(~19 MB total), so a fresh clone can run inference and the dashboard immediately:

| File | Model |
|---|---|
| `weights/yolo26_microplastic.pt` | YOLO26n — primary detector |
| `weights/yolov8_microplastic.pt` | YOLOv8n — H1 baseline |
| `weights/mobilenetv3_morphology_best.pth` | MobileNetV3 — morphology |

The filenames carry no size suffix on purpose: these are the **nano** variants,
and labelling them `_s` would misreport the H1 comparison. `spectral_1dcnn_best.pth`
is not committed — until the spectral model is trained, that branch stays behind
the `MP_SPECTRAL_ENABLED` flag.

### 1. Download the morphology dataset (Moore Institute + PEESEgroup)
```bash
uv run python -m src.preprocessing.download_data all --workers 12
```
This fetches `image_metadata.csv` from the Moore Institute One4All repo, downloads the
morphology-labelled images (Sphere/Fragment/Fiber/Film/Foam) from their public CDN, adds the
84 PS/PE bead images actually shipped in `PEESEgroup/Microplastic-Project` (the README describes
846, but the rest live behind a Google Drive link) to the sphere class, then builds
`data/processed/morphology/{train,val,test}/<class>/` with a **70/15/15 split by image**
(seed 42) plus `class_weights.json` and `splits.json`. Resumable — safe to re-run.

### 2. Train the morphology classifier (MobileNetV3)
```bash
uv run python -m src.training.train_morphology --seed 42 --wandb
```

### 3. Train the polymer classifier (1D-CNN)
```bash
uv run python -m src.training.train_spectral --seed 42 --svm-baseline --wandb
```

### 4. Run the dashboard
```bash
uv run streamlit run src/output/dashboard.py
```

`dashboard.py` puts the repo root on `sys.path` itself, so this works from the
repo root with no environment variables. (Streamlit adds only the *script's own
folder* to the import path, never the working directory, so without that
bootstrap the command fails with `No module named 'src'`.
`tests/test_dashboard_launch.py` guards it by booting the app in a subprocess
with the repo root stripped from `sys.path`.)

By default the dashboard runs in **visual-only mode**: the Image and Summary
tabs, no spectral upload. The spectral branch (Stage 2A/2B) is behind the
`MP_SPECTRAL_ENABLED` feature flag because it needs a trained 1D-CNN
checkpoint. Enable it either with the sidebar toggle or:

```bash
MP_SPECTRAL_ENABLED=1 uv run streamlit run src/output/dashboard.py
```

With it off, `analyze_spectrum()` raises `FeatureDisabledError`, and the PDF
report omits the polymer column, so an image-only run never implies polymer
results that were not computed.

**What you should see.** The Image tab shows the input and the annotated output
side by side — at half width each, which is roughly the native resolution of a
typical micrograph, so the images stay sharp instead of being upscaled. Above
them sit four metric chips (particle count, mean confidences, dominant
morphology); below are a colour key for the detection boxes, the per-particle
table, and a Grad-CAM explainability panel that re-crops the stored bounding
box, so it costs no extra inference.

Appearance is fixed by `.streamlit/config.toml` (committed), so the dashboard
looks the same on every machine. The accent colour is teal by design: the
detection boxes are red, green and blue, and a UI accent in the same hue family
would compete with the annotation you are meant to read first.

## 🤝 Contributor Guidelines

We use a strict branching and pull-request workflow. **Never commit directly to `main`.**

1. **Clone & branch** — one branch per module, e.g. `feat/detection`, `feat/spectral`, `feat/vision`, `feat/integration`.
2. **Code & test** — run the local quality gates before pushing:
   ```bash
   uv run black --check src tests
   uv run ruff check src tests
   uv run pytest
   ```
3. **Commit** — `[module] brief description` (e.g. `[download] moore + peese split script`).
4. **PR** — open against `main`, request review from a teammate, include the wandb link.

## ☁️ Google Colab & VS Code Integration (Train on Colab, Code Locally)

1. Open a Colab notebook with a GPU runtime (T4/A100).
2. Install and run colab-ssh in the first cell:
   ```python
   !pip install colab_ssh --upgrade -q
   from colab_ssh import launch_ssh_cloudflared
   launch_ssh_cloudflared(password="microplastic")
   ```
3. Copy the VS Code SSH command output by the cell; in VS Code run **Remote-SSH: Connect to Host** and paste it.
4. Clone the repo inside the Colab server and install dependencies:
   ```bash
   git clone https://github.com/YOUR-ORG/microplastic-ipd.git
   cd microplastic-ipd
   pip install -r requirements.txt
   ```
5. **Important:** commit and push weights (.pt/.pth) or copy them to Google Drive when training finishes — Colab sessions are ephemeral.

## 📖 Resources
- Live project plan: `IPD_Project_Guide.md` (authoritative — member 10-day plans, datasets, deliverables)
- Agent instructions: `AGENT.md` (architecture spec, interface contracts, naming conventions)