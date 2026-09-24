# MNI-Based Anatomically-Guided 3D Latent Diffusion Model for BraTS 2026 Challenge (Task 4)

[![Challenge](https://img.shields.io/badge/BraTS-2026--Task4-blue.svg)](https://www.synapse.org)
[![Framework](https://img.shields.io/badge/PyTorch-MONAI-orange.svg)](https://monai.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author:** Jheng-Jie Wang (112062117)  
**Project:** BraTS 2026 Challenge - Task 4: 3D Brain MRI Inpainting  

---

## 📖 Overview

This repository contains the official implementation of an MNI-based, anatomically-guided 3D Latent Diffusion Model (LDM) framework for brain MRI inpainting, developed for the **BraTS 2026 Challenge (Task 4)**[cite: 1].

The pipeline solves the brain lesion/artifact inpainting problem by combining:
1. **Spatial Normalization**: Bidirectional spatial mapping using the MNI152 template to eliminate anatomical variability across patient scans[cite: 1].
2. **Early Feature Fusion**: A 17-channel concatenation input strategy (`Z_noisy`, `mask_latent`, `Z_voided`) for strict voxel-level spatial alignment[cite: 1].
3. **Anatomical Guidance (AG-LDM)**: Two-stage training utilizing a frozen MONAI 3D brain segmentation model (**WarpSeg**) as a teacher network for semantic regularization and sampling feedback[cite: 1].
4. **Region-Aware Modeling**: Specialized diffusion models for **Deep**, **Cortical**, and **Center** brain regions to handle diverse topological variations[cite: 1].

---

## 🚀 Key Features & Architecture
### 1. Spatial Standardization (Stage 0)
* Rigid/non-rigid alignment of native MRI volumes into the common MNI152 1mm template space[cite: 1].
* Guarantees structural consistency across different patients before latent encoding[cite: 1].

### 2. Anatomy-Aware Autoencoder Regularization (Stage 1)
* Trained with a custom `VaeGanTrainer` and `PatchGAN` discriminator[cite: 1].
* Employs sparse anatomical segmentation feedback from a frozen teacher model (`WarpSeg`) every $N_{steps}=4$[cite: 1].
* **Low-VRAM Optimization**: Optimized to fit within 16GB GPUs via patch size `[96, 128, 96]`, batch size `1`, and sparse loss evaluation[cite: 1].

### 3. Mask-Conditioned Latent Diffusion (Stage 2)
* **17-Channel Latent Input**:
  $$\text{Input}_{\text{diffusion}} = [Z_{\text{noisy}}\,(8\text{ch}) \,\vert{}\vert{}\, \text{mask}_{\text{latent}}\,(1\text{ch}) \,\vert{}\vert{}\, Z_{\text{voided}}\,(8\text{ch})]$$
[cite: 1]
* **Segmenter-Assisted Sampling Guidance**: Triggers a short-trajectory 10-step DDIM rollout every $f_{seg}=20$ iterations to compute structural boundary gradients for dynamic trajectory tuning[cite: 1].
* **Region-Aware Partitioning**:
  * **Deep Model**: Subcortical and deep gray matter[cite: 1].
  * **Cortical Model**: Superficial lesions and sulcal topology[cite: 1].
  * **Center Model**: Midline axis and ventricular symmetry[cite: 1].

### 4. Seamless Inverse Mapping & Fusion
* Composites generated pseudo-healthy tissues inside the mask boundary with original healthy background outside[cite: 1].
* Warps synthesized MNI volumes back into the original patient's native clinical space ($T_{mri \rightarrow mni}^{-1}$)[cite: 1].

---

## 📊 Experimental Results

Evaluating on the BraTS Validation Set ($N=219$)[cite: 1]:

| Metric | Value ($\text{Mean} \pm \text{Std}$) |
| :--- | :--- |
| **SSIM** $\uparrow$ | $0.7005 \pm 0.1216$[cite: 1] |
| **PSNR (dB)** $\uparrow$ | $16.1258 \pm 2.5945$[cite: 1] |
| **MSE** $\downarrow$ | $0.0293 \pm 0.0212$[cite: 1] |
| **RMSE** $\downarrow$ | $0.0517 \pm 0.0259$[cite: 1] |
| **MAE** $\downarrow$ | $0.0342 \pm 0.0182$[cite: 1] |

---

## 🛠️ Usage

### 1. Data Preprocessing & MNI Cache Building
Transform raw BraTS volumes into standardized MNI space:
```bash
python build_mni_dataset.py --input_dir /path/to/raw_brats --output_dir /path/to/mni_cache
```[cite: 1]

### 2. Stage 1: Autoencoder Training
Train the anatomy-aware 8-channel latent VAE:
```bash
python train_stage1_ae.py --config configs/stage1_ae.yaml

python train_stage2_diffusion.py --config configs/stage2_diffusion.yaml --model_type [deep|cortical|center]
```[cite: 1]

### 4. Inference
Run inpainting on test subjects and export native space `.nii.gz` results:
```bash
python run_inference.py \
    --input_case /path/to/native_case.nii.gz \
    --mask /path/to/mask.nii.gz \
    --output_path ./inpainted_native.nii.gz
```[cite: 1]

---

## 📚 References

1. **Palette**: Saharia, C., et al. "Palette: Image-to-image diffusion models." *ACM SIGGRAPH*, 2022.[cite: 1]
2. **RePaint**: Lugmayr, A., et al. "Repaint: Inpainting using denoising diffusion probabilistic models." *CVPR*, 2022.[cite: 1]
3. **AG-LDM**: Wan, C., et al. "Anatomically Guided Latent Diffusion for Brain MRI Progression Modeling." *arXiv:2601.14584*, 2026.[cite: 1]
