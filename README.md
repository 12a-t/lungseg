# DSSW-Net: Structure and Frequency Aware Dual-Stream Swin-Wavelet Network for 3D Lung Nodule Segmentation

Official code repository for **DSSW-Net**, a dual-stream framework for **3D lung nodule segmentation within predefined regions of interest (ROIs)**. This repository implements segmentation of nodules at known locations; it is **not** a whole-lung nodule detection pipeline.

**Manuscript:** *Structure and Frequency Aware Dual-Stream Swin-Wavelet Network for 3D Lung Nodule Segmentation* — under review at *Medical Physics*.

## Overview

DSSW-Net combines a Swin Transformer semantic stream with a CNN-based detail stream. The current `model.py` includes a parallel local convolution branch within shallow Swin blocks, wavelet-inspired feature downsampling and reconstruction, and **3D squeeze-and-excitation (SE) feature fusion**. Two auxiliary outputs provide deep supervision during training.

> **Implementation note:** The retained model uses learnable stride-2 convolutions for wavelet-like downsampling and standard 3D convolutions in its local branch. These are not mathematically identical to fixed Haar DWT/IDWT and depth-wise convolutions described in the manuscript. Verify the experimental implementation before claiming exact architectural or numerical reproduction.

## Repository Structure

| File | Description |
|---|---|
| `model.py` | DSSW-Net architecture (`SwinUNETR_Optimized`) |
| `dataset.py` | ROI loading, 64³ patch extraction and data augmentation |
| `loss.py` | Dice, Focal and Boundary losses |
| `train.py` | Training and validation |
| `preprocess.py` | LIDC-IDRI preprocessing and 96³ ROI generation |
| `inference.py` | Model inference and prediction-mask generation |
| `evaluate.py` | Segmentation evaluation |
| `requirements.txt` | Python dependencies |

## Data Preparation

CT images are resampled to **1 mm isotropic spacing** and preprocessed into **96 × 96 × 96** nodule-centered ROIs. During training, **64 × 64 × 64** patches are extracted; the model predicts masks at the same 64³ resolution.

Expected preprocessed data layout:

```text
LIDC/
├── lung/                            # CT ROI arrays (.npy), shape 96×96×96
├── mask/                            # Corresponding masks (.npy), shape 96×96×96
└── detailed_dataset_metadata.csv   # Patient-level train/val/test metadata
```

The metadata file must include `FileName`, `PatientID`, `subset` (`train`, `val`, or `test`) and `class_target` (1 for nodules). Data splits must be performed **at the patient level**, without leakage across subsets. `preprocess.py` creates `preprocessed_nodules.csv` by default; this is **not** a replacement for the actual experimental split file.

For raw LIDC-IDRI preprocessing, configure `LIDC_DICOM_ROOT` and `LIDC_OUTPUT_ROOT`, then run:

```bash
python preprocess.py
```

## Installation

Python 3.10, a compatible CUDA-enabled PyTorch installation, and MONAI are recommended. The retained `model.py` constructs `SwinUNETR` without `img_size`; use a compatible MONAI release (for example, the 1.5.x API) and verify it with your environment.

```bash
pip install -r requirements.txt
```

For preprocessing, also ensure `pylidc`, `SimpleITK`, `pydicom`, and `lungmask` are installed. The current repository's dependency versions are not a verified record of the original experimental environment.

## Training

The training configuration uses AdamW, a cosine learning-rate schedule, 100 epochs, batch size 4, gradient accumulation over 4 mini-batches, and a Boundary Loss warm-up period. Adjust data and checkpoint paths to your local environment.

```powershell
$env:DSSWNET_DATA_ROOT = 'D:\datasets\LIDC'
$env:DSSWNET_CHECKPOINT_DIR = 'D:\experiments\DSSWNet\checkpoints'
python train.py
```

## Inference

Use the trained checkpoint to generate nodule segmentation predictions:

```powershell
python inference.py --subset test --mode center64 --checkpoint 'D:\experiments\DSSWNet\checkpoints\best_model.pth'
```

The default `center64` mode predicts the central **64³** patch of each 96³ ROI. The optional `sliding64` mode combines multiple 64³ predictions to reconstruct a **96³** ROI mask. These are **different evaluation protocols** and should not be mixed when comparing results. Inference uses a probability threshold of **0.5** and removes connected components smaller than **50 voxels** by default.

Predictions and `inference_manifest.csv` are saved to `results_inference/` unless `--output-dir` is specified.

## Evaluation

Segmentation performance is assessed using **seven metrics**:

- Dice Similarity Coefficient (Dice)
- Intersection over Union (IoU)
- Sensitivity
- Precision
- 95th Percentile Hausdorff Distance (HD95)
- Average Surface Distance (ASD)
- Surface Dice (1 mm tolerance)

Run evaluation after inference:

```bash
python evaluate.py
```

Results are saved to `results_evaluation/` by default. The evaluation script also supports **subgroup analysis and statistical evaluation** for reproducing the analyses reported in the manuscript when the required metadata and independent-run results are available.

## Reproducibility Notes

The reported study results depend on the original dataset selection, patient-level splits, trained checkpoints, and evaluation settings. The default center-patch validation in `train.py` does not apply the same connected-component filtering as `inference.py`; results from those two stages should not be treated as identical without verification. This repository does not provide whole-lung lesion detection.

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE).
