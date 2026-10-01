# AGENT.md — Automated Microplastic Detection & Classification Using Deep Learning
> **Project:** IPD Semester VI — Dwarkadas J. Sanghvi College of Engineering  
> **Department:** Computer Science & Engineering (Data Science)  
> **Guide:** Dr. Namita Pulgam  
> **Team:** Naman Jain (60009240169) · Kunsh Kaul (60009240033) · Aditya Lakhotia (60009240183) · Rohan Nevrekar (60009240272)  
> **Authority:** This file + `IPD_Project_Guide.md` (both must agree; the Guide is the live project plan, this file is the agent-facing spec).
> **This file:** Instructions for any AI agent, coding assistant, or automated tool working on this codebase.

---

## 1. PROJECT OVERVIEW

This project builds an automated system for detecting and classifying microplastic particles. Microplastics are plastic particles smaller than 5 mm found in water bodies, soil, air, and food chains. Manual detection is slow and error-prone. The system replaces manual microscopy/spectroscopy with **two independent deep-learning pipelines that share one dashboard**:

```
             ┌────────────────────────────────────────────────────────────┐
             │                    STREAMLIT DASHBOARD                      │
             │  upload image → boxes + morphology · upload spectrum →      │
             │  polymer · summaries · JSON export · PDF report             │
             └───────────────┬───────────────────────────────┬────────────┘
                             │                               │
   PIPELINE 1 (visual)       │                   PIPELINE 2 (spectral) │
   ┌─────────────────────────▼──────────┐       ┌─────────────▼──────────────┐
   │ Stage 1A YOLOv11 detection         │       │ Stage 2A spectral preproc   │
   │   detect + crop particle ROIs      │       │   SG smooth → H2O zero → SNV│
   │ Stage 1B MobileNetV3 morphology    │       │ Stage 2B 1D-CNN polymer     │
   │   sphere/fragment/fiber/film/foam  │       │   PE/PP/PS/PMMA/PAN         │
   └────────────────────────────────────┘       └────────────────────────────┘
```

**Design principle — no cross-modal fusion.** The previous CMAF fusion / NADAM
domain-adaptation / multi-task / risk-head architecture was **dropped** because
no open dataset pairs an image and a spectrum for the same particle. The two
pipelines are trained, validated and demonstrated **independently**; they
integrate only at the dashboard layer.

### Three research hypotheses (must be statistically tested, 5 seeds each)
| # | Hypothesis | Test | Decision rule |
|---|---|---|---|
| **H1** | YOLOv11 mAP@0.5 > YOLOv8 mAP@0.5 on microplastic detection | paired t-test (`ttest_rel`, 5 seeds) | p < 0.05, one-sided `alternative="greater"` |
| **H2** | MobileNetV3 fine-tuned > MobileNetV3 from scratch on morphology | McNemar's test on paired predictions | p < 0.05 (statsmodels `mcnemar`) |
| **H3** | 1D-CNN generalises to weathered spectra (FLOPP-e) better than SVM | Wilcoxon signed-rank on F1 across 5 runs | p < 0.05, `alternative="greater"` |

Report Cohen's `d` alongside every test (see `src/evaluation/hypothesis_test.py`).

---

## 2. REPOSITORY STRUCTURE

Agents must follow this exact directory layout. Do not create files outside these locations without justification.

```
microplastic-ipd/
│
├── AGENT.md                        ← You are here (agent authority)
├── README.md                       ← Human-readable project overview
├── requirements.txt                ← All pip dependencies
├── .env.example                    ← Environment variable template (no secrets)
├── pr_checks.yml                   ← CI: black, ruff, mypy, bandit
│
├── data/
│   ├── microplastic.yaml           ← YOLO dataset config (merged detection data)
│   ├── raw/                        ← Original downloaded datasets (do not modify)
│   │   ├── roboflow_unified/       ← Roboflow detection dataset (5,678 train)
│   │   ├── roboflow_unified_v26/   ← v26 variant of the same dataset
│   │   ├── moore_institute/        ← Morphology images (sphere/fragment/fiber/...)
│   │   ├── peese_microbeads/       ← PEESEgroup PS/PE beads (846 → sphere class)
│   │   ├── panats_dataset/         ← Panats microplastic detection dataset
│   │   ├── mina_dataset/           ← MiNa SEM images (test-only, 105 images)
│   │   ├── zenodo_ftir/            ← FTIR hyperspectral (.hdr/.raw)
│   │   ├── zenodo_ftir_c4/         ← FTIR-PLASTIC-c4 (supplementary)
│   │   ├── kaggle_raman/           ← Kaggle RaSPI Raman spectra (supplementary)
│   │   ├── flopp_e/                ← FLOPP-e weathered spectra (OOD test-only)
│   │   ├── ocean_raman/            ← Ocean Raman Library (OOD test-only)
│   │   ├── mendeley_algae/         ← Mendeley MP+Algae (false-positive test)
│   │   ├── sewage_microplastic/    ← Sewage microplastic (noise test)
│   │   └── slopp_raman/            ← SLOPP Raman (legacy reference)
│   └── processed/
│       ├── morphology/             ← train/val/test/<class>/ + class_weights.json
│       │                               + splits.json (built by download_data.py)
│       ├── images/                 ← Promise: merged YOLO splits (train/val/test)
│       ├── spectra/                ← train_spectra.npy, train_labels.npy, scaler.pkl
│       └── splits.json             ← Train/val/test image IDs
│
├── src/
│   ├── detection/                  ← NAMAN's lane
│   │   ├── train_yolo11.py         ← YOLOv11 training script
│   │   ├── train_yolo26.py         ← YOLO26 training script
│   │   ├── compare_yolo.py         ← v8 vs v11 vs v26 comparison table
│   │   └── inference.py            ← detect_and_crop() → cropped ROIs + JSON
│   │
│   ├── preprocessing/
│   │   ├── download_data.py        ← Moore Institute + PEESEgroup download/split
│   │   ├── merge_datasets.py       ← Merge YOLO datasets into data/processed/
│   │   ├── image_preprocess.py     ← Albumentations pipeline (Rohan)
│   │   └── spectral_preprocess.py  ← SG-smooth → H2O-zero → SNV (Kunsh)
│   │
│   ├── models/
│   │   ├── vision_tower.py         ← MobileNetV3-Small morphology classifier
│   │   ├── spectral_tower.py       ← 1D-CNN polymer classifier
│   │   ├── predict_morphology.py   ← STANDALONE: image crop → morphology+conf
│   │   ├── predict_polymer.py      ← STANDALONE: spectrum → polymer+conf
│   │   └── full_pipeline.py        ← Assembles detect→morphology + polymer
│   │
│   ├── training/
│   │   ├── train_morphology.py     ← Rohan's progressive-unfreeze loop
│   │   ├── train_spectral.py       ← Kunsh's 1D-CNN loop + SVM baseline
│   │   └── metrics.py              ← macro_f1, cm, roc_auc, cohens_d helpers
│   │
│   ├── evaluation/
│   │   ├── ablation.py             ← Model comparison tables
│   │   ├── hypothesis_test.py      ← H1 t-test / H2 McNemar / H3 Wilcoxon
│   │   └── grad_cam_viz.py         ← Grad-CAM for detection + morphology
│   │
│   └── output/
│       ├── dashboard.py            ← Streamlit app (image + spectrum tabs)
│       └── report_generator.py     ← ReportLab PDF + JSON export
│
├── notebooks/
│   ├── 01_eda_images.ipynb         ← Image dataset EDA
│   ├── 02_eda_spectra.ipynb        ← Spectral EDA and visualization
│   ├── 03_yolo_training.ipynb      ← Detection training on Colab
│   ├── 04_classification.ipynb     ← Classification demo
│   └── 05_results_analysis.ipynb   ← Hypothesis tests + final plots
│
├── configs/
│   ├── yolo11.yaml                 ← YOLOv11 hyperparameters
│   ├── yolo26.yaml                 ← YOLO26 hyperparameters
│   └── training_config.yaml        ← Shared classification hyperparameters
│
├── weights/                        ← Saved model checkpoints (gitignored)
│   ├── yolov8s_microplastic.pt     ← Naman (baseline)
│   ├── yolo11s_microplastic.pt     ← Naman (primary)
│   ├── yolo26s_microplastic.pt     ← Naman (comparison)
│   ├── mobilenetv3_morphology_best.pth  ← Rohan
│   └── spectral_1dcnn_best.pth     ← Kunsh
│
├── results/
│   ├── detection/                  ← mAP curves, inference examples, gradcam
│   ├── classification/             ← Confusion matrices, F1 scores, gradcam
│   ├── eda/                        ← EDA plots
│   └── hypothesis/                 ← h1/h2/h3_test_results.json
│
└── tests/
    ├── test_vision_tower.py        ← Rohan (output shape [1, 5])
    ├── test_spectral_tower.py      ← Kunsh (output shape [1, 5])
    └── test_detection.py           ← Naman (3 sample images)
```

---

## 3. DATASETS

Agents must know which dataset feeds which module. **Never mix datasets across modules without documentation.**

### 3.1 Pipeline 1 image data
| Dataset | Size | Use | Priority |
|---|---|---|---|
| Roboflow Unified (existing) | 5,678 | YOLOv11 detection training | ✅ have |
| Panats | 963 | merge into detection data | 🔴 download |
| Moore Institute Image Explorer | 5,653+ | MobileNetV3 morphology — filter by class | 🔴 download |
| PEESEgroup Microplastic-Project | 846 (PS 355 / PE 491) described; **84 shipped in repo** (6 SDS × 14) | sphere-class supplementation (rest on Google Drive) | 🔴 download |
| MiNa (Rezvani 2024) | 105 SEM | polymer test-only | test only |
| Mendeley MP+Algae | ~500 | false-positive rejection (algae = negative) | OOD test |
| Sewage microplastic | varies | noisy real-world robustness | OOD test |

Morphology classes used by the vision tower — **always lowercase**:
`['sphere', 'fragment', 'fiber', 'film', 'foam']`.

**Detection classes** (Roboflow): `['microplastic', 'foam', 'bead']`.

### 3.2 Pipeline 2 spectral data
| Dataset | Polymers | Use | Priority |
|---|---|---|---|
| Zenodo FTIR HSI (2555732) | PE, PP, PS, PMMA, PAN | primary 1D-CNN training (~600 bands) | ✅ have |
| Zenodo FTIR-PLASTIC-c4 (10736650) | PE, PP, PS, PET, PVC | secondary training | 🔴 download |
| Kaggle RaSPI Raman | mixed | supplementary Raman | supplementary |
| FLOPP-e | 15 types | 195 weathered spectra — **H3 OOD test** | test only |
| Ocean Raman Library | 40+22 | marine weathered — OOD test | test only |

Polymer classes used by the spectral tower — **always uppercase**:
`['PE', 'PP', 'PS', 'PMMA', 'PAN']`.

### 3.3 Download & split rules
- **Morphology + PEESEgroup:** `python src/preprocessing/download_data.py all`
  - `moore` → `data/raw/moore_institute/<class>/`, `peese` → `data/raw/peese_microbeads/sphere/`
  - `organize` builds `data/processed/morphology/{train,val,test}/<class>/` — **70/15/15 split by IMAGE** (seed 42), writes `class_weights.json` (inverse frequency) and `splits.json`.
- **Spectral:** split **by SAMPLE/IMAGE, never by pixel.** Pixels from the same HSI image are near-identical → pixel splits leak and inflate accuracy to ~95%+ (fake).
- **Detection:** merge via `merge_datasets.py` → `data/processed/images/{train,val,test}/` + `data/microplastic.yaml`. Verify class-name consistency across datasets and remap if needed.

---

## 4. ARCHITECTURE — DETAILED SPECIFICATION

### 4.1 Stage 1A: Detection — YOLOv8/v11/v26

- **Primary:** YOLOv11s → YOLOv11m for final reported results.
- **Baseline:** YOLOv8s (required for H1).
- **Comparison:** YOLO26s (NMS-free, STAL label assignment).
- **Input:** raw microscope image → resize 640×640; **Output:** bbox `[x, y, w, h]` + confidence + class per particle.
- **Hyperparameters:** `configs/yolo11.yaml` and `configs/yolo26.yaml` — `epochs=100, batch=16, imgsz=640, optimizer=AdamW, lr0=0.01, seed=42`. Same hyperparameters for v8 (baseline) so H1 is fair.

#### Metrics to report
mAP@0.5 (primary), mAP@0.5:0.95, Precision, Recall, F1 per class, inference time (ms/image).

#### Critical deliverable — `src/detection/inference.py`
```python
def detect_and_crop(image_path) -> list[dict]:
    # returns [{bbox: [x, y, w, h], confidence: float, class: str, crop_img}, ...]
    # crop_img format must be agreed with Rohan (numpy RGB preferred).
```

### 4.2 Stage 1B: Morphology — MobileNetV3-Small (`src/models/vision_tower.py`)

```
Input: RGB crop [B, 3, 224, 224]
  → timm mobilenetv3_small_100 (pretrained, num_classes=0, global_pool='avg')
  → features [B, 1024]  # ARCHITECTURE CHANGE (runtime fact): timm keeps the 1x1
                        # conv_head (576→1024) even at num_classes=0; the head is
                        # sized from backbone.num_features at runtime, never hardcoded.
  → Linear(1024→256) → LayerNorm → ReLU → Linear(256→5)
Output: [B, 5] morphology logits  (sphere/fragment/fiber/film/foam)
```

**Progressive unfreezing:**
| Epochs | Scope | LR |
|---|---|---|
| 1–10 | backbone frozen, head only | 1e-3 |
| 11–30 | unfreeze last 4 blocks | blocks 1e-4 / head 1e-3 (differential NAdam) |
| 31–50 | full fine-tune | 1e-5 |

Optimizer `NAdam`, scheduler `CosineAnnealingLR`, loss `CrossEntropyLoss(weight=class_weights)`,
early stop patience 10 on val macro-F1. Checkpoint: `weights/mobilenetv3_morphology_best.pth`.
A **ResNet18 baseline is trained with identical settings** for H2.

**Augmentation** (`src/preprocessing/image_preprocess.py`, albumentations):
train = HFlip/VFlip/Rotate90/ColorJitter + Normalize(ImageNet); val/test = Resize + Normalize only.

### 4.3 Stage 2A: Spectral preprocessing (`src/preprocessing/spectral_preprocess.py`)

Mandatory order — do not reorder:
1. **Savitzky-Golay smoothing** — `savgol_filter(x, window_length=11, polyorder=2, axis=1)`
2. **Water-vapour band removal** — zero bands `280:320` (600-band FTIR)
3. **SNV normalisation** — `(x - mean) / std` per spectrum (mean 0, std 1)

Save `data/processed/spectra/train_spectra.npy`, `train_labels.npy`, `scaler.pkl`.
The fitted scaler **must** be saved and reused for OOD (FLOPP-e) evaluation.

### 4.4 Stage 2B: 1D-CNN polymer classifier (`src/models/spectral_tower.py`)

```
Input: [B, 600] → unsqueeze → [B, 1, 600]
Conv1D(1→64, k=7) → BN → ReLU → MaxPool(2)        → [B, 64, 300]
Conv1D(64→128, k=5) → BN → ReLU → MaxPool(2)       → [B, 128, 150]
Conv1D(128→256, k=3) → BN → ReLU → AdaptiveAvgPool(1) → [B, 256]
Linear(256→256) → LayerNorm → ReLU → Linear(256→5) → [B, 5]   (PE/PP/PS/PMMA/PAN)
```

Training (`src/training/train_spectral.py`): NAdam lr=1e-3, cosine `T_max=50 eta_min=1e-6`,
class-weighted CE, patience 10 on val macro-F1. **SVM baseline** (`sklearn.svm.SVC`) on identical
preprocessed spectra for H3. Checkpoint: `weights/spectral_1dcnn_best.pth`.

### 4.5 Stage 3: Dashboard + reports (`src/output/`)

- **Streamlit** `dashboard.py` — Tab 1 image (detect → morphology boxes+labels), Tab 2 spectrum
  (preprocess → polymer bar), Tab 3 summary (counts, morphology pie, polymer bar), JSON export,
  PDF download (ReportLab `report_generator.py`).
- **JSON schema per particle:**
  `{particle_id, bbox, detection_confidence, morphology, morphology_confidence, polymer, polymer_confidence}`

### 4.6 Integration contracts (Aditya)

- `predict_morphology(image_crop) → {"morphology": str, "confidence": float}` *(PIL Image or numpy HWC RGB)*
- `predict_polymer(spectrum_array) → {"polymer": str, "confidence": float}` *(preprocessed [600] array)*
- `detect_and_crop(image_path) → [ {bbox, confidence, class, crop_img}, ... ]`

`full_pipeline.py` imports these three. Keep signatures stable.

---

## 5. TRAINING CONFIGURATION

`configs/training_config.yaml` is the shared hyperparameter authority:

```yaml
training:
  epochs: 50
  batch_size: 32
  optimizer: nadam
  learning_rate: 1e-3
  lr_scheduler: cosine_annealing
  weight_decay: 1e-4
  early_stopping_patience: 10
  random_seed: 42

model:
  vision_backbone: mobilenetv3_small_100
  spectral_input_bands: 600

data:
  image_size: 224
  train_split: 0.70
  val_split: 0.15
  test_split: 0.15
  split_by: sample            # CRITICAL: never split by pixel
```

### Reproducibility rule (mandatory in every training script)
```python
import torch, numpy as np, random
SEED = 42
def set_seed(seed=SEED):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()
```
H1/H2/H3 use seeds `{42, 123, 456, 789, 1024}`.

---

## 6. THREE RESEARCH HYPOTHESES — IMPLEMENTATION

All live in `src/evaluation/hypothesis_test.py`; results → `results/hypothesis/h{n}_test_results.json`.

```python
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

def h1_paired_ttest(yolo11_maps, yolov8_maps):
    stat, p = ttest_rel(yolo11_maps, yolov8_maps, alternative="greater")
    d = (np.mean(yolo11_maps) - np.mean(yolov8_maps)) / np.std(yolo11_maps - yolov8_maps)
    return {"statistic": stat, "p_value": p, "cohens_d": float(d)}

def h2_mcnemar(y_true, pred_finetuned, pred_scratch):
    b = np.sum(pred_finetuned == y_true) & np.sum(pred_scratch != y_true)
    c = ...
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=True)
    return {"statistic": result.statistic, "p_value": result.pvalue}

def h3_wilcoxon(cnn_f1s, svm_f1s):
    stat, p = wilcoxon(cnn_f1s, svm_f1s, alternative="greater")
    return {"statistic": stat, "p_value": p}
```

Common mistakes: two-sided vs one-sided (`alternative="greater"`), computing McNemar on
non-paired samples, running tests on a single seed. Both are forbidden.

---

## 7. METRICS TO REPORT

Agents must compute and log ALL of these — missing any will be flagged.

- **Detection:** mAP@0.5, mAP@0.5:0.95, precision/recall/F1 per class, inference ms/image.
- **Classification (morphology & polymer):** accuracy, **macro-F1 (primary)**,
  per-class F1, confusion matrix PNG, ROC-AUC one-vs-rest.
- **Hypotheses:** t-statistic / McNemar stat / Wilcoxon stat, p-value, Cohen's d, decision.
- **Mandatory comparison tables** (results/*.csv):
  - detection: `YOLOv8s vs YOLOv11s vs YOLO26s — mAP@0.5, mAP@0.5:0.95, Params, Inference(ms)`
  - classification: ablation `Model | Morphology_F1 | Polymer_F1 | Avg_F1`

---

## 8. EXPERIMENT TRACKING (wandb)

All training must be logged to wandb project **`microplastic-ipd`** (no exceptions):

```python
import wandb
wandb.init(project="microplastic-ipd",
           name="morphology-unfreeze-run1", config={...})
wandb.log({"train/loss": ..., "val/macro_f1": ..., "val/accuracy": ..., "epoch": e})
wandb.log({"confusion": wandb.plot.confusion_matrix(y_true=..., preds=..., class_names=...)})
```

---

## 9. VISUALIZATION REQUIREMENTS

1. **Grad-CAM on morphology** — target `vision_tower.backbone.blocks[-1]`, 15 images (3 per class) → `results/classification/gradcam_morphology/`.
2. **Grad-CAM on detection** — target `model.model.model[-2]`, 10 images → `results/detection/gradcam_examples/`.
3. **Spectral saliency map** — vanilla gradient per polymer class; verify peaks align to known bands (e.g., PE C-H 2800–3000 cm⁻¹) → `results/classification/spectral_saliency_maps/`.
4. **Mean spectra per class overlay** → `results/eda/mean_spectra_per_class.png`.
5. **Training curves** (loss + macro-F1) → `results/classification/training_curves.png`.
6. **YOLO inference examples** (≥5 images with boxes+conf) → `results/detection/inference_examples/`.

---

## 10. COMMON MISTAKES — AGENTS MUST AVOID

### Data leakage (critical)
- ❌ Splitting FTIR hyperspectral data by pixel → **always by sample/image ID**
- ❌ Splitting the morphology set by crop/patch → **splits.json is by image**
- ✅ Reuse the exact split from `data/processed/morphology/splits.json`

### Training
- ❌ Training the morphology classifier on full images — always YOLO ROI crops
- ❌ Forgetting the seed block → results unreproducible
- ❌ Metrics computed on logits instead of softmax (except CE loss, which wants logits)
- ❌ Running H1/H3 on a single seed

### Integration
- ❌ Changing `predict_morphology` / `predict_polymer` / `detect_and_crop` signatures without team agreement
- ❌ dtype mismatches (numpy vs PIL vs torch tensor) between `detect_and_crop` output and `predict_morphology` input — pass RGB numpy arrays
- ❌ Saving `.pth` but training from a different device (GPU/CPU) → save `model.state_dict` and load with `map_location`

### Reporting
- ❌ Accuracy alone — report macro-F1 (class imbalance: Film 35 / Foam 8 vs Sphere ~2200)
- ❌ Hypothesis tests without Cohen's d / effect size

---

## 11. ENVIRONMENT & DEPENDENCIES

`requirements.txt` is the single source of truth: torch, torchvision, ultralytics, timm,
albumentations, roboflow, spectral, scipy, scikit-learn, numpy, pandas, matplotlib, seaborn,
grad-cam, wandb, streamlit, plotly, reportlab, statsmodels.

### Colab setup cell
```python
!pip install ultralytics timm albumentations roboflow spectral grad-cam wandb streamlit plotly reportlab statsmodels -q
from google.colab import drive
drive.mount('/content/drive')
```

### Local quality gates (mirrors CI)
```bash
black --check src/ tests/
ruff check src/ tests/
mypy src/ --ignore-missing-imports
bandit -r src/ -ll -ii
```

---

## 12. 10-DAY PARALLEL PLAN AND DEPENDENCIES

| Day | Naman (detection) | Kunsh (spectral) | Rohan (vision) | Aditya (integration) |
|---|---|---|---|---|
| 1 | Repo setup + env | Download spectral data | Download Moore + PEESEgroup | EDA template + wandb |
| 2 | Dataset download + merge | Spectral EDA | Morphology EDA | Ablation framework |
| 3 | Image EDA + class analysis | Preprocessing pipeline | MobileNetV3 + augmentation | metrics.py + full_pipeline stub |
| 4 | YOLOv8 baseline ⏳ | 1D-CNN build | Training run 1 (frozen) | Integrate stubs |
| 5 | YOLOv11 training ⏳ | Training run 1 + SVM | Training run 2 (unfreeze) | Test framework |
| 6 | YOLO26 training + table | Results + confusion | Final fine-tune + Grad-CAM | Integration prep |
| 7 | ⭐ inference.py DONE | OOD testing + H3 | ⭐ predict_morphology.py DONE | Integrate both |
| 8 | H1 hypothesis test | Spectral saliency | Streamlit dashboard | H1 + H2 + H3 |
| 9 | Grad-CAM detection | ⭐ predict_polymer.py DONE | PDF report | Integrate spectral |
| 10 | Docs + PR | Docs + PR | Demo + deploy | ⭐ Final merge |

⭐ = critical dependency deliverable. ⏳ = long GPU job — start before leaving.
Partial-by-role guidance: if you are asked for code belonging to another member's
day, scaffold it to the contract and flag it; do not silently reimplement.

---

## 13. FILE NAMING CONVENTIONS

```
weights/
  yolov8s_microplastic.pt            yolov8s baseline
  yolo11s_microplastic.pt            yolo11s primary
  yolo26s_microplastic.pt            yolo26s comparison
  mobilenetv3_morphology_best.pth    morphology classifier
  spectral_1dcnn_best.pth            polymer classifier

results/
  detection/yolo_v{version}_{split}_results.csv
  detection/yolo_comparison_table.csv
  classification/confusion_{task}_{model}.png      e.g. confusion_morphology_mobilenetv3.png
  hypothesis/h{n}_test_results.json                e.g. h1_test_results.json
  sample_report.pdf
```

---

## 14. GIT WORKFLOW

- Branch names: `naman/detection`, `kunsh/spectral`, `rohan/vision`, `aditya/integration`.
- Commit message: `[module] brief description` — e.g. `[download] moore institute + peese split script`.
- Never push to main directly — always open a Pull Request.
- Merge order on Day 10: naman → kunsh → rohan → aditya (Aditya does final merge).
- PR checklist: code runs, CI green (black/ruff/mypy/bandit), tests pass, wandb link included.

---

## 15. CONTACT & ESCALATION

If an agent encounters an ambiguous requirement or a conflict between this AGENT.md
and the codebase, follow this priority order:

1. **`IPD_Project_Guide.md` + this AGENT.md** — live plan and agent spec; if they conflict, the Guide wins and AGENT.md must be updated.
2. **`configs/*.yaml`** — hyperparameter authority.
3. **Existing code in `src/`** — defer to docs on architecture; fix code if it matches neither.
4. **Ask the team** — Naman, Kunsh, Aditya, or Rohan via the college group channel.

Any architecture change (adding/removing modules, changing signatures, changing splits)
must be documented with a comment `# ARCHITECTURE CHANGE: <reason>` and noted in the project log.

---

*Last updated: September 2026 | Guide: Dr. Namita Pulgam | DJSCE Department of CSE (Data Science)*