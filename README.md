# Industrial Texture Anomaly Detection via 4D Simplex Diffusion (DTU-Net)

An unsupervised anomaly detection system designed for stochastic industrial textures (textiles and leather). This project implements and extends the **DTU-Net** diffusion architecture (*Diffusion Models for Industrial Anomaly Detection*), leveraging 4D Perlin/Simplex noise corruptions and partial reverse diffusion to locate structural, surface, and chemical defects without requiring defective samples during training.

---

## Executive Summary

In industrial manufacturing, anomalous samples are rare, unpredictable, and expensive to collect. Traditional discriminative classifiers fail when presented with novel defect types. 

This project explores **generative denoising diffusion probabilistic models (DDPMs)** trained strictly on defect-free reference textures. When an anomalous image is partially perturbed with continuous Simplex noise and iteratively denoised, the model projects the surface back to the manifold of defect-free textures. By computing the multi-scale photometric residual between the original capture and its reconstructed counterpart, anomalies are localized and scored.

### Key Highlights
* **Unsupervised Training**: Trained exclusively on normal, pristine texture patches.
* **Continuous Multi-Scale Noise**: Utilizes 4D Simplex/Perlin noise perturbations rather than Gaussian noise to preserve structural frequency coherence.
* **Tiled High-Resolution Inference**: Evaluates high-resolution surfaces ($1024 \times 1024$ and larger) using overlapping sliding-window tiles ($512 \times 512$, stride $384$) with edge-falloff compensation.
* **Dual-Pathway Scoring**: Combines high-quantile pixel residuals with spatial top-k anomaly density to robustly detect both diffuse stains and sharp structural defects.

---

## Methodology & Architecture

```
                       [ Input Texture Tile (512x512) ]
                                      │
                                      ▼
             [ 4D Simplex Noise Perturbation (t = 50 steps) ]
                                      │
                                      ▼
               [ DTU-Net Generative U-Net Denoising Engine ]
                                      │
                                      ▼
                        [ Reconstructed Clean Tile ]
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
  [ Pixel L2 Residual Map ]                           [ Structural / Edge Error ]
            │                                                   │
            └─────────────────────────┬─────────────────────────┘
                                      ▼
                        [ Dual-Pathway Anomaly Map ]
                                      │
                         [ Tile-Aggregation & Top-K ]
                                      │
                                      ▼
                        [ Defect Decision / Score ]
```

### 1. Partial Diffusion & Reconstruction
Rather than generating images from pure noise ($T = 1000$), the pipeline applies partial forward diffusion ($T_{simplex} = 50$). This corrupts subtle localized surface anomalies (such as stains, scratches, or punctures) while retaining global fabric weave and macro-texture patterns. The reverse diffusion step then "heals" the corrupted region according to the learned normal distribution.

### 2. Simplex Noise vs. Gaussian Noise
Standard Gaussian noise acts as independent high-frequency white noise, which tends to be ignored or smoothed out by autoencoders. Simplex noise generates continuous, multi-octave spatial frequency patterns that closely mimic physical texture irregularities, forcing the U-Net to learn semantic texture continuity.

### 3. Tiled Inference for Full-Field Inspection
Industrial textile rolls exceed standard GPU memory limits. A sliding-window tiled inference engine splits macro captures into overlapping patches, runs reverse diffusion on each tile, and merges the reconstructed tiles using a 2D cosine blending window to eliminate edge stitching artifacts.

---

## Datasets & Benchmark Results

### 1. MVTec AD Leather Benchmark
* **Objective**: Standard baseline validation against canonical industrial benchmarks.
* **Resolution**: $512 \times 512$
* **Result**: Achieved **~0.99 Image AUROC**, reliably isolating cuts, fold defects, and surface abrasions.

### 2. Custom Industrial Fabric Stain Dataset
* **Objective**: Real-world evaluation on challenging textile captures containing subtle liquid stains, oil marks, and chemical discolorations.
* **Test Set**: 100 defective stained captures, 10 pristine normal references across varied weave patterns.
* **Evaluation Framework**: Frozen, independent test set evaluated under tiled inference ($t = 50$, stride $384$, quantile $0.999$).

| Metric | Score | 95% Confidence Interval |
| :--- | :---: | :---: |
| **Image AUROC** | **0.773** | [0.588, 0.910] |
| **Average Precision (AP)** | **0.961** | [0.909, 0.985] |
| **Sensitivity (Recall)** | **51.0%** | [41.3%, 60.6%] |
| **Specificity** | **90.0%** | [59.6%, 98.2%] |
| **Balanced Accuracy** | **70.5%** | — |

---

## Directory Structure

```
diffusion_texture_inspection/
├── README.md                                  # Project documentation
├── requirements.txt                           # PyTorch and generative ML dependencies
│
├── src/
│   ├── dual_pathway.py                        # Dual-pathway anomaly aggregation
│   ├── baseline.py                            # Author DTU-Net diffusion wrapper
│   └── author_smoke.py                        # U-Net forward/reverse smoke verification
│
├── notebooks/
│   ├── 03_mvtec_leather_train.ipynb           # MVTec Leather diffusion training
│   ├── 04_mvtec_leather_evaluate.ipynb        # MVTec Leather evaluation & ROC analysis
│   ├── 07_fabric_stain_train.ipynb            # Fabric stain diffusion training
│   ├── 08_fabric_stain_evaluate_baseline.ipynb# Baseline fabric stain evaluation
│   └── 09_fabric_stain_evaluate_v9_dual_pathway.ipynb # Production dual-pathway evaluation
│
├── results/
│   ├── fabric_stain/                          # Evaluation metrics, JSONs, and summary plots
│   └── mvtec_leather/                         # MVTec benchmark results
│
├── artifacts/
│   ├── fabric_stain_tiled_final_evaluation/   # Frozen final benchmark reports & score maps
│   └── mvtec_leather_evaluation/              # Leather test score distributions
│
└── tools/
    ├── build_fabric_stain_training_notebook.py
    ├── build_fabric_stain_evaluation_notebook_v9.py
    └── prepare_fabric_stain_pilot.py
```

---

## Setup & Execution

### 1. Environment Setup
Create a virtual environment and install the required PyTorch and computer vision packages:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Running the Model Check
Verify that the U-Net architecture, weights, and simplex noise generation pass self-consistency tests:
```bash
python src/author_smoke.py
```

### 3. Running Evaluations via Jupyter
Launch Jupyter Lab or Notebook to inspect the evaluated pipelines:
```bash
jupyter lab notebooks/
```
* **For Fabric Stain**: Open [`notebooks/09_fabric_stain_evaluate_v9_dual_pathway.ipynb`](notebooks/09_fabric_stain_evaluate_v9_dual_pathway.ipynb).
* **For MVTec Leather**: Open [`notebooks/04_mvtec_leather_evaluate.ipynb`](notebooks/04_mvtec_leather_evaluate.ipynb).
