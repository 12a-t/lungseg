# DSSW-Net: Structure and Frequency Aware Dual-Stream Swin-Wavelet Network for 3D Lung Nodule Segmentation

PyTorch implementation for **known-location lung nodule ROI segmentation**. The manuscript has been submitted to *Medical Physics*; a DOI and publication status should be added only after confirmation.

> **Reproducibility notice:** The supplied `model.py` implements a trainable stride-2 Conv3d-based wavelet-like downsampling block, channel-to-space rearrangement for upsampling, and standard Conv3d (`groups=1`) in the parallel local MLP branch. These are not identical to the fixed Haar DWT/IDWT and depth-wise convolution operators described in the current manuscript. This README describes the code as implemented; it does not claim that unverified results reproduce the published/manuscript metrics.

## Files

| File | Role |
|---|---|
| `model.py` | `SwinUNETR_Optimized` dual-stream network, deep supervision and SE fusion |
| `dataset.py` | Reads 96³ preprocessed ROIs, trains on 64³ patches with spatial augmentations |
| `loss.py` | Dice, Focal and signed-distance Boundary losses (GitHub version clips distance values to [-5,5]) |
| `train.py` | 100 epochs, AdamW, cosine schedule, batch size 4 and accumulation 4 |
| `preprocess.py` | LIDC-IDRI preprocessing, annotation consensus and 96³ ROI extraction |
| `inference.py` | 64³ center-patch or sliding-window prediction; threshold 0.5, min component 50 voxels |
| `evaluate.py` | Per-nodule metrics, subgroups, 2,000 patient-cluster bootstrap and optional three-run statistics |

**Input geometry:** `96×96×96` is the saved ROI. A `64×64×64` patch is supplied to the model, whose main output is also `64×64×64`. The model's `feature_size=96` specifies feature **channels**, not spatial dimensions.

## Environment: Windows / PyCharm

Use Python 3.10 and a PyTorch build compatible with your CUDA driver (install PyTorch separately before other dependencies). Then:

```powershell
pip install -r requirements.txt
python -c "import torch, monai; print('torch:',torch.__version__, 'monai:',monai.__version__, 'cuda:',torch.cuda.is_available())"
```

The current `model.py` constructs MONAI `SwinUNETR` **without** the `img_size` argument; a compatible 1.5.x MONAI release is therefore recommended. MONAI 1.3.x requires `img_size`, while MONAI 1.5.x removed it. This is a compatibility recommendation, not proof of the original experiment environment. Test your real checkpoint before any reproduction claim.

## Data layout

```text
LIDC/
├── lung/                               # Preprocessed CT ROI .npy, shape [96,96,96]
├── mask/                               # Consensus segmentation .npy, shape [96,96,96]
└── detailed_dataset_metadata.csv      # At minimum: FileName, PatientID, subset, class_target
```

`subset` must be one of `train`, `val`, `test`. Keep a **patient-level split** with no patient shared across splits. The preprocessing script by default creates `preprocessed_nodules.csv`, **not** a historical train/val/test split; to reproduce the manuscript, provide the actual split file from the experiment. Do not regenerate a new random split and label it as the original study.

The internal manuscript cohort has 1,343 nodules from 1,010 eligible patients; the held-out test set contains 282 nodules. These numbers are historical dataset descriptors and are **not** produced or guaranteed by this source code.

`preprocess.py` is specific to LIDC-IDRI/pylidc. The independent clinical cohort requires its own approved external-data import and verified annotations. Do not publish patient-identifying information.

## Commands

Point the project at your own data (replace Windows paths as appropriate):

```powershell
$env:DSSWNET_DATA_ROOT='D:\datasets\LIDC'
$env:DSSWNET_CHECKPOINT_DIR='D:\experiments\DSSWNet\checkpoints'
$env:LIDC_DICOM_ROOT='D:\datasets\raw_LIDC-IDRI'
$env:LIDC_OUTPUT_ROOT='D:\datasets\LIDC'

python train.py
python inference.py --subset test --mode center64 --checkpoint 'D:\experiments\DSSWNet\checkpoints\best_model.pth'
python evaluate.py
```

`center64` predicts the middle 64³ of the known 96³ ROI. The alternative `--mode sliding64` produces a 96³ prediction by blending multiple 64³ windows; the two modes are distinct **evaluation protocols** and should not be pooled. Both modes avoid selecting prediction crops from the ground-truth mask.

`inference.py` removes foreground connected components smaller than **50 voxels** after thresholding at **0.5**. Note that `train.py`'s current validation Dice is calculated **without** this component filtering. Thus checkpoint selection and final postprocessed test Dice are not exactly the same computation. The manuscript does not specify whether validation used component filtering; do not silently reinterpret the historical selection protocol.

## Metrics and statistical analysis

`evaluate.py` reports per-nodule Dice, IoU, pixel sensitivity, precision, HD95, ASD and Surface Dice (1-mm tolerance). Its implementation uses count-weighted surface voxels, which is not necessarily identical to an area-weighted surface Dice library. Empty predictions have Dice = 0 but undefined HD95/ASD; inspect valid-case counts before interpreting boundary-metric averages.

For three *actual independent runs* of DSSW-Net:

```powershell
python evaluate.py --run-csvs run1/per_nodule_metrics.csv run2/per_nodule_metrics.csv run3/per_nodule_metrics.csv --bootstrap 2000 --output-dir results_statistics
```

For Welch run-level comparison with three independently reproduced MedNeXt runs, add:

```text
--reference-run-csvs med1/per_nodule_metrics.csv med2/per_nodule_metrics.csv med3/per_nodule_metrics.csv
```

No scores, p-values or CI bounds are hard-coded. Bootstrap requires real patient IDs and a fixed matched test cohort. Morphology subgroup metrics require real Solid/Sub-solid labels; the code does not infer these labels from imaging intensities.

## Implementation caveats

- The current uploaded `model.py` is structurally different from the fixed-Haar/depth-wise operators described in the manuscript; resolving this requires checking the actual experiment or correcting the paper, not relabeling code.
- The `loss.py` currently present in the GitHub ZIP **clips the signed distance map to [-5,5]**. A separately provided older file did not; this changes the Boundary Loss. Keep the version actually used in your experimental runs.
- Full GPU execution, original weight loading and reproduction of the reported results are **not** established by syntax checks or synthetic CPU unit tests.
- `preprocess.py` depends on `pylidc`, `SimpleITK` and `lungmask` and requires local DICOM data and pylidc indexing/configuration.

## License

MIT (see `LICENSE`).
