# Dataset Guide & Kaggle Integration

This directory contains lightweight sample captures for local smoke testing, as well as instructions for accessing the full industrial benchmark datasets on Kaggle.

---

## 1. Local Sample Data (`sample_fabric/`)
A lightweight 27 MB subset of the industrial fabric stain dataset is included directly in this repository for local testing, algorithm verification, and quick experimentation without needing to download gigabytes of raw data:
* `train_normal/`: 10 pristine normal reference fabric captures.
* `val_normal/`: 5 normal validation captures.
* `test/`: 15 evaluation samples containing liquid stains, discoloration, and clean controls.
* `manifest.csv`: Image metadata and lighting conditions.

---

## 2. Full Benchmark Datasets (Kaggle & Public Sources)

For full 2,000-step training and complete evaluation runs, the full high-resolution datasets are hosted on Kaggle:

### A. Industrial Fabric Stain Dataset
* **Description**: 468 high-resolution textile captures across varied weave patterns, with 100 defective stained captures and normal reference sets.
* **Kaggle Link**: [Industrial Fabric Stain Anomaly Dataset](https://www.kaggle.com/)
* **Format**: Standard MVTec-style directory structure (`train/good`, `test/good`, `test/stain`).

### B. MVTec Anomaly Detection (MVTec AD) Leather
* **Description**: The standard academic benchmark for industrial texture anomaly segmentation.
* **Kaggle Link**: [MVTec AD Dataset on Kaggle](https://www.kaggle.com/datasets/ipythonx/mvtec-ad)
* **Official Website**: [MVTec Software GmbH](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

## 3. Automated Kaggle Download (Optional)

If you have the Kaggle CLI configured (`kaggle.json`), you can fetch the full datasets directly:

```bash
# Download MVTec AD
kaggle datasets download -d ipythonx/mvtec-ad -p data/mvtec --unzip
```
