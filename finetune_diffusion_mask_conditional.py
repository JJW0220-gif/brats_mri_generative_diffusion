#!/usr/bin/env python
"""Fine-tune diffusion model with mask conditioning (9-channel input).

Pipeline:
1) Load pretrained 8-channel diffusion model
2) Expand first conv to 9 channels (mask conditioning)
3) Load BraTS training data with random hole augmentation
4) Fine-tune on latent representations with mask as additional input
5) Save checkpoint and sample outputs

Transfer learning approach:
- AutoencoderKL: frozen (pretrained)
- DiffusionModelUNet: expand 8→9 channels, fine-tune for 20-50 epochs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader, Dataset
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler


def _load_ckpt_flexible(model: torch.nn.Module, ckpt_path: str) -> None:
    """Load checkpoint flexibly, handling various formats."""
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if hasattr(model, "load_old_state_dict"):
        model.load_old_state_dict(checkpoint)
        return

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint from {ckpt_path}")


def _expand_first_conv_layer(model: torch.nn.Module, old_in_channels: int = 8, new_in_channels: int = 9) -> None:
    """Expand first conv layer from 8 to 9 channels for mask conditioning.
    
    The new channel(s) are initialized to 0, preserving pretrained behavior.
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv3d) and module.in_channels == old_in_channels:
            old_weight = module.weight.data  # (out_c, in_c, k, k, k)
            old_bias = module.bias
            
            out_channels, _, kernel_d, kernel_h, kernel_w = old_weight.shape
            
            # Create new weight with expanded in_channels
            new_weight = torch.zeros(
                out_channels, new_in_channels, kernel_d, kernel_h, kernel_w,
                dtype=old_weight.dtype, device=old_weight.device
            )
            # Copy old weights into first 8 channels
            new_weight[:, :old_in_channels, :, :, :] = old_weight
            
            # Replace the layer
            module.weight.data = new_weight
            if old_bias is not None:
                module.bias.data = old_bias
            
            print(f"✓ Expanded first conv layer '{name}': in_channels {old_in_channels} → {new_in_channels}")
            break


def collect_brats_cases(data_dir: str) -> List[Dict]:
    """Collect BraTS training cases (t1, t1ce, t2, flair).
    
    We'll use t1 modality for now (can extend to multi-modal).
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")

    patient_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    
    def _pick_first_3d(candidates: List[Path]) -> Path | None:
        for p in sorted(candidates):
            try:
                if len(nib.load(str(p)).shape) == 3:
                    return p
            except Exception:
                continue
        return None

    cases: List[Dict] = []
    for patient_dir in patient_dirs:
        # Look for t1 modality
        t1_candidates = list(patient_dir.glob("*_t1.nii.gz")) + list(patient_dir.glob("*-t1.nii.gz"))
        t1_path = _pick_first_3d(t1_candidates)
        
        if t1_path is None:
            continue

        cases.append({
            "image": str(t1_path),
            "name": patient_dir.name,
        })

    if not cases:
        raise ValueError(f"No valid 3D t1 images found in {data_dir}")

    print(f"Found {len(cases)} training cases")
    return cases


def make_transforms(spacing=(1.1, 1.1, 1.1), roi_size=(144, 176, 112)):
    """Standard preprocessing + random hole augmentation."""
    return transforms.Compose([
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(keys=["image"], pixdim=spacing, mode="bilinear"),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=roi_size),
        transforms.ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=99.5, b_min=0.0, b_max=1.0
        ),
    ])


def create_random_mask(image_shape: tuple, mask_prob: float = 0.15) -> torch.Tensor:
    """Create random binary mask (1=hole, 0=known).
    
    Args:
        image_shape: (D, H, W) spatial dimensions
        mask_prob: probability of masking each voxel (~15% = typical hole size)
    
    Returns:
        torch.Tensor of shape (1, D, H, W)
    """
    mask = (torch.rand(image_shape) < mask_prob).float()
    return mask.unsqueeze(0)


def build_models(device: torch.device):
    """Build autoencoder (frozen) and diffusion (9-channel)."""
    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=8,
        num_channels=(64, 128, 256),
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=(False, False, False),
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
    ).to(device)

    diffusion = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=9,  # 8 (latent) + 1 (mask)
        out_channels=8,
        num_channels=(256, 256, 512),
        attention_levels=(False, True, True),
        num_res_blocks=2,
        num_head_channels=(0, 64, 64),
    ).to(device)

    return autoencoder, diffusion


def compute_scale_factor(autoencoder: AutoencoderKL, loader: DataLoader, device: torch.device) -> float:
    """Compute scale factor = std of latent distribution for stable training."""
    print("Computing latent scale factor...")
    latents = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing scale", total=min(100, len(loader))):
            image = batch["image"].to(device)
            z = autoencoder.encode_stage_2_inputs(image)
            latents.append(z.cpu())
            if len(latents) >= 100:
                break
    
    latents = torch.cat(latents, dim=0)
    scale_factor = float(latents.std())
    print(f"Scale factor: {scale_factor:.4f}")
    return scale_factor


def train_epoch(
    diffusion: DiffusionModelUNet,
    autoencoder: AutoencoderKL,
    scheduler: DDPMScheduler,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    scale_factor: float = 0.18215,
) -> float:
    """Single training epoch with mask conditioning."""
    diffusion.train()
    autoencoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(loader, desc="Training", total=len(loader)):
        image = batch["image"].to(device)
        
        # Encode to latent
        with torch.no_grad():
            z = autoencoder.encode_stage_2_inputs(image)
            z = z * scale_factor
        
        # Create random mask (hole region)
        spatial_shape = z.shape[2:]
        mask = create_random_mask(spatial_shape, mask_prob=0.15).to(device)
        
        # Resize mask if needed
        mask_latent = F.interpolate(mask, size=spatial_shape, mode="nearest")
        mask_latent = (mask_latent > 0.5).float()
        
        # Sample timesteps
        batch_size = z.shape[0]
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (batch_size,), device=device)
        
        # Add noise
        noise = torch.randn_like(z)
        z_noisy = scheduler.add_noise(z, noise, timesteps)
        
        # Concatenate latent with mask
        model_input = torch.cat([z_noisy, mask_latent], dim=1)
        
        # Forward pass with AMP
        optimizer.zero_grad()
        with autocast():
            pred_noise = diffusion(model_input, timesteps=timesteps)
            loss = F.mse_loss(pred_noise, noise)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"Epoch loss: {avg_loss:.6f}")
    return avg_loss


@torch.no_grad()
def sample_and_save(
    diffusion: DiffusionModelUNet,
    autoencoder: AutoencoderKL,
    scheduler_sample,
    device: torch.device,
    output_dir: Path,
    num_samples: int = 4,
    scale_factor: float = 0.18215,
) -> None:
    """Sample from model and save reconstructions."""
    diffusion.eval()
    autoencoder.eval()
    
    print("Sampling...")
    
    for sample_idx in range(num_samples):
        # Start with random latent
        z = torch.randn(1, 8, 18, 22, 14, device=device)  # match latent spatial dims
        
        # Create random mask
        mask = create_random_mask(z.shape[2:], mask_prob=0.2).to(device)
        
        # DDIM sampling with mask conditioning
        scheduler_sample.set_timesteps(num_inference_steps=50)
        
        for t in scheduler_sample.timesteps:
            t_tensor = torch.full((z.shape[0],), int(t), device=device, dtype=torch.long)
            model_input = torch.cat([z, mask], dim=1)
            
            pred_noise = diffusion(model_input, timesteps=t_tensor)
            z, _ = scheduler_sample.step(pred_noise, int(t), z)
        
        # Decode
        z = z / scale_factor
        sample = autoencoder.decode_stage_2_outputs(z)
        
        # Save
        sample_np = sample[0, 0].cpu().numpy().astype(np.float32)
        sample_path = output_dir / f"sample_{sample_idx:03d}.npy"
        np.save(str(sample_path), sample_np)
    
    print(f"Samples saved to {output_dir}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    cases = collect_brats_cases(args.train_data_dir)
    ds = Dataset(data=cases, transform=make_transforms())
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Build models
    autoencoder, diffusion = build_models(device)
    
    # Load pretrained weights
    _load_ckpt_flexible(autoencoder, args.autoencoder_ckpt)
    _load_ckpt_flexible(diffusion, args.diffusion_ckpt)
    
    # Expand diffusion from 8→9 channels
    print("Expanding diffusion model to 9-channel (mask conditioning)...")
    _expand_first_conv_layer(diffusion, old_in_channels=8, new_in_channels=9)
    
    # Freeze autoencoder
    for param in autoencoder.parameters():
        param.requires_grad = False
    
    # Setup training
    scheduler_train = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        schedule="scaled_linear_beta",
    )
    
    from generative.networks.schedulers import DDIMScheduler
    scheduler_sample = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        schedule="scaled_linear_beta",
        clip_sample=False,
    )
    
    optimizer = optim.AdamW(diffusion.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = GradScaler()
    
    # Compute scale factor
    scale_factor = compute_scale_factor(autoencoder, loader, device)
    
    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting fine-tuning for {args.num_epochs} epochs...")
    
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} ---")
        
        loss = train_epoch(
            diffusion=diffusion,
            autoencoder=autoencoder,
            scheduler=scheduler_train,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            scale_factor=scale_factor,
        )
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = output_dir / f"diffusion_mask_cond_epoch_{epoch+1:03d}.pt"
            torch.save(diffusion.state_dict(), str(ckpt_path))
            print(f"✓ Checkpoint saved: {ckpt_path}")
        
        # Sample
        if (epoch + 1) % args.sample_interval == 0:
            sample_dir = output_dir / f"samples_epoch_{epoch+1:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            sample_and_save(
                diffusion=diffusion,
                autoencoder=autoencoder,
                scheduler_sample=scheduler_sample,
                device=device,
                output_dir=sample_dir,
                scale_factor=scale_factor,
            )
    
    # Final checkpoint
    final_ckpt = output_dir / "diffusion_mask_cond_final.pt"
    torch.save(diffusion.state_dict(), str(final_ckpt))
    print(f"\n✓ Final checkpoint saved: {final_ckpt}")
    
    print(f"\n✅ Fine-tuning complete. Results in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune diffusion model with mask conditioning")
    parser.add_argument("--train_data_dir", type=str, required=True, help="Path to BraTS training data")
    parser.add_argument("--autoencoder_ckpt", type=str, default="./models/model_autoencoder.pt")
    parser.add_argument("--diffusion_ckpt", type=str, default="./models/model.pt")
    parser.add_argument("--output_dir", type=str, default="./output_finetune_mask_cond")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--sample_interval", type=int, default=5)
    args = parser.parse_args()
    main(args)
