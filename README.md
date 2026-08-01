# DSSW-Net: Dual-Stream Swin-Wavelet Network for 3D Lung Nodule Segmentation

Official PyTorch implementation of **DSSW-Net**, a dual-stream architecture for accurate 3D lung nodule segmentation, published in **Medical Physics (2026)**.

## 📄 Paper

DSSW-Net: Dual-Stream Swin-Wavelet Network with Structure-Frequency Awareness for 3D Lung Nodule Segmentation

*Medical Physics, 2026*

[Link to paper - DOI to be added]

## 🏗️ Architecture Overview

DSSW-Net employs a dual-stream encoder design:

- **Semantic Stream**: Swin Transformer backbone with Structure-Aware Parallel Modules for global context modeling
- **Detail Stream**: Wavelet-CNN branch using 3D Discrete Wavelet Transform (DWT) for high-frequency feature preservation
- **Feature Fusion**: Asymmetric dual-stream fusion with bidirectional cross-attention
- **Reconstruction**: Wavelet-Enhanced Reconstruction Module (WERM) with inverse DWT for boundary refinement

## ⚙️ Requirements

- Python 3.8+
- CUDA 11.7 (recommended)
- PyTorch 2.0+
- MONAI 1.3.0+

Install all dependencies:

```bash
pip install -r requirements.txt
