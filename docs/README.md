# MNI-Based Anatomically-Guided 3D Latent Diffusion Model for BraTS 2026 Challenge (Task 4)

[![Challenge](https://img.shields.io/badge/BraTS-2026--Task4-blue.svg)](https://www.synapse.org)
[![License](https://img.shields.io/badge/License-Apache--2.0-green.svg)](../LICENSE)

**Author:** Jheng-Jie Wang (112062117)  
**Project:** BraTS 2026 Challenge - Task 4: 3D Brain MRI Inpainting  

---

## 📖 Overview

This repository contains an MNI-based, anatomically-guided 3D Latent Diffusion Model (LDM) framework for brain MRI inpainting, developed for the **BraTS 2026 Challenge (Task 4)**.

The pipeline solves the brain lesion/artifact inpainting problem by combining:
1. **Spatial Normalization**: Bidirectional spatial mapping using the MNI152 template to reduce anatomical variability across patient scans.
2. **Latent Conditioning**: A 17-channel concatenation input strategy (`Z_noisy`, `mask_latent`, `Z_voided`) for voxel-level spatial alignment.
3. **Anatomical Guidance**: Two-stage training with a frozen MONAI 3D segmentation teacher for semantic regularization and sampling feedback.
4. **Region-Aware Modeling**: Separate diffusion checkpoints for **Deep**, **Cortical**, and **Center** brain regions.

---

## Pretrained Model

The pretrained model used in this project is the base MONAI 3D latent diffusion model before the region-specific mask-conditioned fine-tuning. It provides the initialization and baseline for 3D brain MRI inpainting.

The pretrained weights are loaded from the local `models/` directory:

* `models/model_autoencoder.pt`: pretrained 3D `AutoencoderKL` used to encode MRI volumes into 8-channel latent representations.
* `models/model.pt`: pretrained 3D diffusion U-Net used as the base diffusion model before mask conditioning and region-specific training.

These weights are loaded by the inference and fine-tuning configs; they are not downloaded automatically by the training commands. The region-specific checkpoints (`model_inpaint_mni_deep.pt`, `model_inpaint_mni_cortical.pt`, and `model_inpaint_mni_center.pt`) are derived training outputs and should be distinguished from the original pretrained baseline.

The pretrained baseline was evaluated on 219 validation cases using five image-reconstruction metrics:

| Metric | Value (Mean +/- Std) |
| :--- | :--- |
| **SSIM** | 0.5468175114192994 +/- 0.15803867813841377 |
| **PSNR (dB)** | 9.807287362476877 +/- 3.659877873659053 |
| **MSE** | 0.13279866336268328 +/- 0.07318928674526683 |
| **RMSE** | 0.11181356570780017 |
| **MAE** | 0.08476125223832869 |

`cases_evaluated`: 219

---

## 🚀 Key Features & Architecture
### 1. Spatial Standardization (Stage 0)
* Alignment of native MRI volumes into the common MNI152 1mm template space.
* Produces reusable per-case MNI cache files before latent encoding.

### 2. Anatomy-Aware Autoencoder Regularization (Stage 1)
* Trained with the repository's anatomy-aware VAE/GAN trainer and PatchGAN discriminator.
* Employs sparse anatomical segmentation feedback from a frozen teacher model every `seg_every_n_steps=4`.
* **Low-VRAM Optimization**: The current configuration uses patch size `[96, 128, 96]`, batch size `1`, and sparse loss evaluation.

### 3. Mask-Conditioned Latent Diffusion (Stage 2)
* **17-Channel Latent Input**:
  ```text
  diffusion input = [Z_noisy (8 channels) || mask_latent (1 channel) || Z_voided (8 channels)]
  ```
* **Segmenter-Assisted Sampling Guidance**: Triggers a short 10-step DDIM rollout every `fseg=20` iterations to compute structural guidance.
* **Region-Aware Partitioning**:
  * **Deep Model**: Learns masks and inpainting patterns in interior or subcortical regions. Checkpoint: `models/model_inpaint_mni_deep.pt`.
  * **Cortical Model**: Learns masks near the cortical boundary and superficial tissue. Checkpoint: `models/model_inpaint_mni_cortical.pt`.
  * **Center Model**: Learns masks near the central or midline region. Checkpoint: `models/model_inpaint_mni_center.pt`.

### Custom mask augmentation

`RandomMaskAugmentd` can augment the real lesion mask with synthetic 3D masks. The synthetic mask generator samples box, sphere, and blob shapes, then places them according to the training region:

* `region='deep'` places the mask around the volume interior.
* `region='cortical'` places the mask near the volume boundary.
* `region='center'` places the mask near the volume center.

The region-specific training configs disable random region selection and pass the matching fixed region to `make_inpaint_mni_transforms`. This keeps synthetic mask placement aligned with the checkpoint being trained. The placement is a geometric heuristic in MNI space, not an atlas-derived anatomical segmentation.

### 4. Seamless Inverse Mapping & Fusion
* Composites generated tissue inside the mask boundary with the original voided image outside.
* Warps synthesized MNI volumes back into the original patient's native space when transform metadata is available.

---

## 📊 Experimental Results

Evaluating on the BraTS Validation Set ($N=219$):

| Metric | Value ($\text{Mean} \pm \text{Std}$) |
| :--- | :--- |
| **SSIM** $\uparrow$ | $0.7005 \pm 0.1216$ |
| **PSNR (dB)** $\uparrow$ | $16.1258 \pm 2.5945$ |
| **MSE** $\downarrow$ | $0.0293 \pm 0.0212$ |
| **RMSE** $\downarrow$ | $0.0517 \pm 0.0259$ |
| **MAE** $\downarrow$ | $0.0342 \pm 0.0182$ |

---

## 🛠️ Usage

### 1. Data Preprocessing & MNI Cache Building
Run commands from the repository root. Transform raw BraTS volumes into standardized MNI space:
```bash
python scripts/build_mni_dataset.py \
  --data_dir /path/to/raw_brats \
  --template_path ./MNI152_T1_1mm_brain.nii.gz \
  --output_dir ./data/mni_cache \
  --overwrite
```

If the cache is already aligned and only pseudo-healthy targets are needed:
```bash
python scripts/build_mni_dataset.py \
  --aligned_cache_dir ./data/mni_cache \
  --template_path ./MNI152_T1_1mm_brain.nii.gz \
  --overwrite
```

Each cached case may contain `t1n_mni.nii.gz`, `mask_mni.nii.gz`, `unhealthy_mask_mni.nii.gz`, `healthy_mask_mni.nii.gz`, `t1n_voided_mni.nii.gz`, `t1n_pseudo_healthy_mni.nii.gz`, the MNI transform, and `metadata.json`.

### 2. Stage 1: Autoencoder Training
Train the anatomy-aware 8-channel latent VAE:
```bash
python -m monai.bundle run --config_file configs/train_autoencoder_mni.json
```

### 3. Stage 2: Region-Specific Diffusion Training
Train one checkpoint for each region. Each config filters the MNI cache by `train_region` and uses a matching fixed synthetic-mask region.
```bash
python -m monai.bundle run --config_file configs/train_diffusion_inpaint_mni_deep.json
python -m monai.bundle run --config_file configs/train_diffusion_inpaint_mni_cortical.json
python -m monai.bundle run --config_file configs/train_diffusion_inpaint_mni_center.json
```

Each config saves a final checkpoint named `model_inpaint_mni_<region>.pt` under `models/`. If that file exists at startup, its weights are loaded before training; otherwise training starts from the initialized diffusion network.

### 4. Inference
Run MNI inpainting on validation subjects and export MNI/native-space `.nii.gz` results:
```bash
python -m monai.bundle run --config_file configs/inference_inpaint_mni.json
```

The sampler determines the case region and routes it to `model_inpaint_mni_deep.pt`, `model_inpaint_mni_cortical.pt`, or `model_inpaint_mni_center.pt`. It fails if the required region checkpoint is missing; it does not silently fall back to a unified checkpoint.

Large datasets, cache volumes, model weights, and generated outputs should remain outside the normal Git history. The repository `.gitignore` excludes paths such as `data/`, `models/`, `*.pt`, and `*.nii.gz`; Git LFS is required if these artifacts must be stored remotely.

---

## 📚 References

1. **Palette**: Saharia, C., et al. "Palette: Image-to-image diffusion models." *ACM SIGGRAPH*, 2022.
2. **RePaint**: Lugmayr, A., et al. "RePaint: Inpainting using denoising diffusion probabilistic models." *CVPR*, 2022.
3. **AG-LDM**: Wan, C., et al. "Anatomically Guided Latent Diffusion for Brain MRI Progression Modeling." *arXiv:2601.14584*, 2026.
