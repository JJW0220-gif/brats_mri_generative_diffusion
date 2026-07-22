#!/usr/bin/env python
"""Run full latent-diffusion inpainting on all validation cases.

Pipeline:
1) load voided + mask volumes
2) preprocess to model input shape
3) encode voided image to latent space (AutoencoderKL)
4) denoise latent with DiffusionModelUNet while preserving known region via mask
5) decode latent and compose final inpainted image with mask
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader, Dataset, MetaTensor
from tqdm import tqdm

from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler
from scripts.uncond_inpaint_sampler import UncondInpaintSampler


def _load_ckpt_flexible(model: torch.nn.Module, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if hasattr(model, "load_old_state_dict"):
        model.load_old_state_dict(checkpoint)
        return

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)


def _expand_first_conv_layer(model: torch.nn.Module, old_in_channels: int = 8, new_in_channels: int = 9) -> None:
    """Expand first conv layer from in_channels=8 to in_channels=9 (for mask conditioning).
    
    The new channel(s) are initialized to 0, preserving the behavior of the pretrained 8-channel model.
    """
    # Find the first conv layer in the model
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
            
            print(f"Expanded first conv layer '{name}': in_channels {old_in_channels} → {new_in_channels}")
            break


def build_models(device: torch.device, use_mask_conditioning: bool = False):
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

    diffusion_in_channels = 9 if use_mask_conditioning else 8
    diffusion = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=diffusion_in_channels,
        out_channels=8,
        num_channels=(256, 256, 512),
        attention_levels=(False, True, True),
        num_res_blocks=2,
        num_head_channels=(0, 64, 64),
    ).to(device)

    return autoencoder, diffusion


def collect_voided_mask_cases(val_data_dir: str) -> List[Dict]:
    root = Path(val_data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Validation directory not found: {root}")

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
        voided_candidates = list(patient_dir.glob("*-t1n-voided.nii.gz")) + list(patient_dir.glob("*_voided.nii.gz"))
        mask_candidates = list(patient_dir.glob("*-mask.nii.gz")) + list(patient_dir.glob("*_mask.nii.gz"))

        voided_path = _pick_first_3d(voided_candidates)
        mask_path = _pick_first_3d(mask_candidates)
        if voided_path is None or mask_path is None:
            continue

        case_name = patient_dir.name
        cases.append(
            {
                "voided": str(voided_path),
                "mask": str(mask_path),
                "name": case_name,
                "voided_path": str(voided_path),
                "mask_path": str(mask_path),
            }
        )

    if not cases:
        raise ValueError("No valid (voided, mask) 3D pairs found in validation directory")

    return cases


def make_transforms(spacing=(1.1, 1.1, 1.1), roi_size=(144, 176, 112)):
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=["voided", "mask"]),
            transforms.EnsureChannelFirstd(keys=["voided", "mask"]),
            transforms.EnsureTyped(keys=["voided", "mask"]),
            transforms.Orientationd(keys=["voided", "mask"], axcodes="RAS"),
            transforms.Spacingd(keys=["voided"], pixdim=spacing, mode="bilinear"),
            transforms.Spacingd(keys=["mask"], pixdim=spacing, mode="nearest"),
            transforms.CenterSpatialCropd(keys=["voided", "mask"], roi_size=roi_size),
            transforms.ScaleIntensityRangePercentilesd(
                keys=["voided"], lower=0, upper=99.5, b_min=0.0, b_max=1.0
            ),
            transforms.ScaleIntensityRanged(keys=["mask"], a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )


def _normalize_mask(mask: torch.Tensor, target_size: tuple[int, ...] | None = None) -> torch.Tensor:
    """Normalize a mask to hole-region semantics (1 = generate, 0 = preserve)."""
    if target_size is not None:
        mask = F.interpolate(mask, size=target_size, mode="nearest")

    mask = mask.float()
    mask_bin = (mask > 0.5).float()
    if mask_bin.mean() > 0.5:
        mask_bin = 1.0 - mask_bin
    return mask_bin


def save_nifti(volume: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4)), str(out_path))


def save_nifti_like(volume: np.ndarray, ref_path: str, out_path: Path) -> None:
    """Save volume with affine/header copied from a reference file."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_img = nib.load(ref_path)
    out_img = nib.Nifti1Image(
        volume.astype(np.float32),
        affine=ref_img.affine,
        header=ref_img.header.copy(),
    )
    nib.save(out_img, str(out_path))


def save_inpainted_with_inverse_transform(
    volume: torch.Tensor,
    ref_path: str,
    out_path: Path,
    inverse_transform,
    source_tensor: torch.Tensor | None = None,
) -> None:
    """Save a volume back into the original image space using the inverse of preprocessing transforms."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_img = nib.load(ref_path)

    meta_dict = dict(getattr(source_tensor, "meta", {}) or {})
    applied_operations = getattr(source_tensor, "applied_operations", None)
    if applied_operations and isinstance(applied_operations, list) and isinstance(applied_operations[0], list):
        applied_operations = applied_operations[0]

    meta_tensor = MetaTensor(
        volume.detach().cpu().contiguous(),
        meta_dict=meta_dict,
        applied_operations=applied_operations,
    )

    inverse_input = {"inpaint": meta_tensor}
    if source_tensor is not None:
        inverse_input["voided"] = source_tensor.detach().cpu()

    inverted = inverse_transform(inverse_input)
    inverted_array = inverted["inpaint"].detach().cpu().numpy()
    if inverted_array.ndim == 5 and inverted_array.shape[0] == 1:
        inverted_array = inverted_array[0]
    if inverted_array.ndim == 4 and inverted_array.shape[0] == 1:
        inverted_array = inverted_array[0]

    out_img = nib.Nifti1Image(
        inverted_array.astype(np.float32),
        affine=ref_img.affine,
        header=ref_img.header.copy(),
    )
    nib.save(out_img, str(out_path))


def to_submission_name(case_name: str) -> str:
    """Return challenge-compliant output filename."""
    # Expected: BraTS-GLI-XXXXX-YYY-t1n-inference.nii.gz
    base = case_name
    if base.endswith(".nii.gz"):
        base = base[:-7]
    if base.endswith(".nii"):
        base = base[:-4]
    return f"{base}-t1n-inference.nii.gz"


@torch.no_grad()
def latent_inpaint_ddim(
    diffusion: DiffusionModelUNet,
    scheduler: DDIMScheduler,
    z_known: torch.Tensor,
    mask_latent: torch.Tensor,
    num_inference_steps: int,
    device: torch.device,
    use_mask_conditioning: bool = False,
):
    """Inpaint in latent space using DDIM and region preservation.

    mask_latent is 1 in hole region (generate), 0 in known region (preserve).
    If use_mask_conditioning=True, concat mask with z_t as input.
    """
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    z_t = torch.randn_like(z_known, device=device)
    fixed_noise = torch.randn_like(z_known, device=device)

    for t in scheduler.timesteps:
        t_tensor = torch.full((z_t.shape[0],), int(t), device=device, dtype=torch.long)
        
        if use_mask_conditioning:
            # Concatenate mask with latent
            model_input = torch.cat([z_t, mask_latent], dim=1)
        else:
            model_input = z_t
        
        model_output = diffusion(model_input, timesteps=t_tensor, context=None)
        z_prev, _ = scheduler.step(model_output, int(t), z_t)

        # Keep non-masked region aligned with known latent at the same noise level.
        z_known_t = scheduler.add_noise(original_samples=z_known, noise=fixed_noise, timesteps=t_tensor)
        z_t = z_prev * mask_latent + z_known_t * (1.0 - mask_latent)

    return z_t


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cases = collect_voided_mask_cases(args.val_data_dir)
    print(f"Found {len(cases)} validation cases with (voided, mask)")

    preprocessing = make_transforms()
    ds = Dataset(data=cases, transform=preprocessing)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    autoencoder, diffusion = build_models(device, use_mask_conditioning=args.use_mask_conditioning)
    _load_ckpt_flexible(autoencoder, args.autoencoder_ckpt)
    _load_ckpt_flexible(diffusion, args.diffusion_ckpt)
    
    # If using mask conditioning, expand first conv layer from 8 to 9 channels
    if args.use_mask_conditioning:
        _expand_first_conv_layer(diffusion, old_in_channels=8, new_in_channels=9)
    
    autoencoder.eval()
    diffusion.eval()

    sampler = UncondInpaintSampler(num_inference_steps=args.num_inference_steps)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0015,
        beta_end=0.0195,
        schedule="scaled_linear_beta",
        clip_sample=False,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inverse_transform = transforms.Invertd(
        keys=["inpaint"],
        transform=preprocessing,
        orig_keys=["voided"],
        nearest_interp=[False],
        to_tensor=True,
    )

    with torch.no_grad():
        for batch in tqdm(loader, total=len(loader), ncols=100):
            voided = batch["voided"].to(device)
            mask = batch["mask"].to(device)

            if args.use_mask_conditioning:
                # Existing path for a 9-channel masked-conditioning model.
                z_known = autoencoder.encode_stage_2_inputs(voided)
                mask_latent = _normalize_mask(mask, target_size=z_known.shape[2:]).to(z_known.dtype)
                z_inpaint = latent_inpaint_ddim(
                    diffusion=diffusion,
                    scheduler=scheduler,
                    z_known=z_known,
                    mask_latent=mask_latent,
                    num_inference_steps=args.num_inference_steps,
                    device=device,
                    use_mask_conditioning=True,
                )
                generated = autoencoder.decode_stage_2_outputs(z_inpaint)
                mask_img = _normalize_mask(mask, target_size=voided.shape[2:]).to(voided.dtype)
                inpainted = voided * (1.0 - mask_img) + generated * mask_img
            else:
                # Unconditional inpainting path: keep the 8-channel UNet untouched.
                inpainted = sampler.run(
                    autoencoder=autoencoder,
                    diffusion=diffusion,
                    scheduler=scheduler,
                    voided=voided,
                    mask=mask,
                    device=device,
                )

            for i in range(voided.shape[0]):
                name = batch["name"][i] if isinstance(batch["name"], list) else f"case_{i:04d}"
                voided_path = batch["voided_path"][i] if isinstance(batch["voided_path"], list) else batch["voided_path"]
                mask_path = batch["mask_path"][i] if isinstance(batch["mask_path"], list) else batch["mask_path"]

                # The unconditional path already composes the result in image space.
                inpainted_slice = inpainted[i, :1]
                out_name = to_submission_name(name)
                save_inpainted_with_inverse_transform(
                    volume=inpainted_slice,
                    ref_path=voided_path,
                    out_path=output_dir / out_name,
                    inverse_transform=inverse_transform,
                    source_tensor=batch["voided"][i],
                )

    print(f"Done. Results saved under: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full latent diffusion inpainting for all validation cases")
    parser.add_argument("--val_data_dir", type=str, required=True)
    parser.add_argument("--autoencoder_ckpt", type=str, default="./models/model_autoencoder.pt")
    parser.add_argument("--diffusion_ckpt", type=str, default="./models/model.pt")
    parser.add_argument("--output_dir", type=str, default="./output_val_all_diffusion_inpaint")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument(
        "--use_mask_conditioning",
        type=lambda x: x.lower() in {"1", "true", "yes", "y"},
        default=False,
        help="Whether to use mask as additional input to diffusion model (requires fine-tuned 9-channel model)",
    )
    args = parser.parse_args()
    main(args)
