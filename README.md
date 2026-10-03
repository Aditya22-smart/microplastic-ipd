# Automated Microplastic Detection & Classification Using Deep Learning

IPD Semester VI project — Dwarkadas J. Sanghvi College of Engineering (CSE Data Science)
**Guide:** Dr. Namita Pulgam · **Team:** Naman Jain, Kunsh Kaul, Aditya Lakhotia, Rohan Nevrekar

## 🔬 Project Overview

An automated system that detects, localizes, and classifies microplastic particles. The system is built from **two independent deep-learning pipelines** that share one dashboard — a design that works because no open dataset pairs an image and a spectrum for the same particle:

1. **Pipeline 1 — Visual (image in):**
   - **YOLOv11 detection** localizes microplastic particles and crops each one.
   - **MobileNetV3-Small** classifies each crop's morphology → *sphere / fragment / fiber / film / foam*.
2. **Pipeline 2 — Spectral (spectrum in):**
   - FTIR/Raman preprocessing (Savitzky-Golay smoothing → water-vapour band zeroing → SNV normalization).
   - **1D-CNN** classifies the polymer type → *PE / PP / PS / PMMA / PAN*.
3. **Dashboard — Streamlit:** upload an image or spectrum, see detection boxes, morphology and polymer labels, particle counts, JSON export, and a downloadable PDF report.

### Three Research Hypotheses
| # | Hypothesis | Test |
|---|---|---|
| H1 | YOLOv11 mAP@0.5 > YOLOv8 on microplastic detection | paired t-test, 5 seeds |
| H2 | MobileNetV3 fine-tuned > from-scratch on morphology | McNemar's test |
| H3 | 1D-CNN generalizes to weathered spectra (FLOPP-e) better than SVM | Wilcoxon signed-rank |

## 🛠️ Tools & Technologies
- **Deep Learning:** PyTorch, Ultralytics (YOLOv8/v11/v26)
- **Vision:** Albumentations, timm (MobileNetV3), pytorch-grad-cam
- **Spectroscopy:** scipy, spectral, scikit-learn (SVM baseline)
- **Stats:** scipy.stats, statsmodels (McNemar)
- **Tracking:** Weights & Biases (wandb)
- **Output:** Streamlit, Plotly, ReportLab
- **Quality gates:** Black, Ruff, Mypy, Bandit (CI)

## 📂 Folder Structure
```text
microplastic-ipd/
├── configs/                  # YOLO + shared training hyperparameters
├── data/
│   ├── raw/                  # (Gitignored) Original datasets (Moore, PEESE, Zenodo, ...)
│   └── processed/            # (Gitignored) Morphology splits, spectra .npy, scaler.pkl
├── notebooks/                # EDA + Colab training notebooks (01–05)
├── results/                  # Tables, confusion matrices, Grad-CAMs, hypothesis JSONs
├── src/
│   ├── detection/            # Naman: YOLO training, compare_yolo, inference.detect_and_crop
│   ├── preprocessing/        # download_data, merge_datasets, image/spectral preprocess
│   ├── models/               # vision_tower, spectral_tower, predict_morphology, predict_polymer, full_pipeline
│   ├── training/             # train_morphology, train_spectral, metrics
│   ├── evaluation/           # hypothesis_test (H1/H2/H3), grad_cam_viz, ablation
│   └── output/               # Streamlit dashboard + ReportLab PDF
├── tests/                    # Pytest unit tests
└── weights/                  # (Gitignored) Trained checkpoints (.pt / .pth)
```

## 🚀 Getting Started

```bash
python -m venv venv
# Windows: venv\Scripts\activate   ·  Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
pip install ruff black mypy pytest bandit
```

### 1. Download the morphology dataset (Moore Institute + PEESEgroup)
```bash
python src/preprocessing/download_data.py all --workers 12
```
This fetches `image_metadata.csv` from the Moore Institute One4All repo, downloads the
morphology-labelled images (Sphere/Fragment/Fiber/Film/Foam) from their public CDN, adds the
84 PS/PE bead images actually shipped in `PEESEgroup/Microplastic-Project` (the README describes
846, but the rest live behind a Google Drive link) to the sphere class, then builds
`data/processed/morphology/{train,val,test}/<class>/` with a **70/15/15 split by image**
(seed 42) plus `class_weights.json` and `splits.json`. Resumable — safe to re-run.

### 2. Train the morphology classifier (MobileNetV3)
```bash
python src/training/train_morphology.py --seed 42 --wandb
```

### 3. Train the polymer classifier (1D-CNN)
```bash
python src/training/train_spectral.py --seed 42 --svm-baseline --wandb
```

### 4. Run the dashboard
```bash
streamlit run src/output/dashboard.py
```

## 🤝 Contributor Guidelines

We use a strict branching and pull-request workflow. **Never commit directly to `main`.**

1. **Clone & branch** — teams use `naman/detection`, `kunsh/spectral`, `rohan/vision`, `aditya/integration`.
2. **Code & test** — run the local quality gates before pushing:
   ```bash
   black .
   ruff check .
   mypy src/ --ignore-missing-imports
   pytest tests/
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