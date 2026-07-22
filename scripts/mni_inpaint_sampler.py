from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List
import sys

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms as monai_transforms
from monai.data import DataLoader, Dataset, MetaTensor

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.apply_mni_transform import apply_transform
from scripts.mask_conditioned_ldm_trainer import load_mask_conditioned_diffusion
from scripts.mni_transforms import make_inpaint_mni_transforms


def _save_nifti_like(volume: np.ndarray, ref_path: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref = nib.load(ref_path)
    nib.save(nib.Nifti1Image(volume.astype(np.float32), ref.affine, ref.header), str(out_path))


def _copy_nifti(src_path: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.load(src_path), str(out_path))


def _invert_crop_to_full_mni(
    decoded_crop: torch.Tensor,
    source_voided_meta: torch.Tensor,
    preprocessing: monai_transforms.Compose,
) -> np.ndarray | None:
    """Use MONAI Invertd to reverse CenterSpatialCropd back to full MNI grid."""
    try:
        inv_tfm = monai_transforms.Invertd(
            keys=["inpaint"],
            transform=preprocessing,
            orig_keys=["voided"],
            nearest_interp=[False],
            to_tensor=True,
        )
        meta_dict = dict(getattr(source_voided_meta, "meta", {}) or {})
        applied_ops = getattr(source_voided_meta, "applied_operations", None)
        if applied_ops and isinstance(applied_ops, list) and len(applied_ops) > 0 and isinstance(applied_ops[0], list):
            applied_ops = applied_ops[0]
        meta_inpainted = MetaTensor(
            decoded_crop.detach().cpu().contiguous(),
            meta_dict=meta_dict,
            applied_operations=applied_ops,
        )
        result = inv_tfm({"inpaint": meta_inpainted, "voided": source_voided_meta.detach().cpu()})
        arr = result["inpaint"].detach().cpu().numpy()
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        return arr.astype(np.float32)
    except Exception:
        return None


class MNIInpaintSampler:
    def __init__(
        self,
        num_inference_steps: int = 50,
        scale_factor: float = 1.0,
        use_mask_conditioning: bool = True,
        concat_voided_latent: bool = False,
        concat_atlas_latent: bool = False,
        atlas_prior_scale: float = 1.0,
        region_checkpoint_paths: Dict[str, str] | None = None,
        default_checkpoint_path: str = "",
        save_mni_output: bool = True,
    ) -> None:
        self.num_inference_steps = num_inference_steps
        self.scale_factor = scale_factor
        self.use_mask_conditioning = use_mask_conditioning
        self.concat_voided_latent = concat_voided_latent
        self.concat_atlas_latent = concat_atlas_latent
        self.atlas_prior_scale = atlas_prior_scale
        self.region_checkpoint_paths = dict(region_checkpoint_paths or {})
        self.default_checkpoint_path = default_checkpoint_path or ""
        self.save_mni_output = save_mni_output

    @staticmethod
    def _normalize_mask(mask: torch.Tensor, target_size: tuple[int, ...]) -> torch.Tensor:
        mask = F.interpolate(mask.float(), size=target_size, mode="nearest")
        return (mask > 0.5).to(dtype=torch.float32)

    def _infer_region(self, case: Dict[str, str]) -> str:
        region = str(case.get("mask_region", "") or "").strip().lower()
        if region:
            return region
        mask_path = case.get("mask_mni_path") or case.get("mask") or ""
        if not mask_path:
            return "all"
        from scripts.mni_transforms import _infer_mask_region

        return _infer_mask_region(mask_path)

    def _resolve_diffusion_model(
        self,
        diffusion_model: torch.nn.Module,
        case: Dict[str, str],
        device: torch.device,
    ) -> torch.nn.Module:
        region = self._infer_region(case)
        checkpoint_path = self.region_checkpoint_paths.get(region)
        if not checkpoint_path:
            raise FileNotFoundError(f"No region checkpoint configured for region '{region}'")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Region checkpoint not found for region '{region}': {checkpoint_path}")

        model_copy = copy.deepcopy(diffusion_model)
        model_copy.to(device)
        model_copy.eval()
        load_mask_conditioned_diffusion(model_copy, checkpoint_path)
        return model_copy

    def _sample_latent(
        self,
        diffusion_model: torch.nn.Module,
        scheduler,
        z_known: torch.Tensor,
        mask_latent: torch.Tensor,
        z_voided: torch.Tensor | None,
        z_atlas: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        z_t = torch.randn_like(z_known, device=device)
        fixed_noise = torch.randn_like(z_known, device=device)
        scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)

        for t in scheduler.timesteps:
            t_tensor = torch.full((z_t.shape[0],), int(t), device=device, dtype=torch.long)
            model_inputs = [z_t]
            if self.use_mask_conditioning:
                model_inputs.append(mask_latent)
            if self.concat_voided_latent:
                if z_voided is None:
                    raise ValueError("concat_voided_latent=True requires z_voided during inference.")
                model_inputs.append(z_voided)
            if self.concat_atlas_latent:
                if z_atlas is None:
                    raise ValueError("concat_atlas_latent=True requires z_atlas during inference.")
                model_inputs.append(z_atlas)
            model_input = torch.cat(model_inputs, dim=1)
            model_output = diffusion_model(model_input, timesteps=t_tensor, context=None)
            z_prev, _ = scheduler.step(model_output, int(t), z_t)
            z_known_t = scheduler.add_noise(original_samples=z_known, noise=fixed_noise, timesteps=t_tensor)
            z_t = z_prev * mask_latent + z_known_t * (1.0 - mask_latent)
        return z_t

    @torch.no_grad()
    def run(
        self,
        cases: List[Dict[str, str]],
        autoencoder_model: torch.nn.Module,
        diffusion_model: torch.nn.Module,
        scheduler,
        device: torch.device,
        output_dir: str,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        prepared_cases = []
        for case in cases:
            prepared = dict(case)
            prepared.setdefault("voided", case.get("voided_mni_path", case["image"]))
            prepared_cases.append(prepared)

        preprocessing = make_inpaint_mni_transforms(include_voided=True, include_atlas=self.concat_atlas_latent)
        dataset = Dataset(
            data=prepared_cases,
            transform=preprocessing,
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        autoencoder_model.eval()
        diffusion_model.eval()

        case_index = 0
        for batch in loader:
            image = batch["image"].to(device)
            voided = batch["voided"].to(device)
            mask = batch["mask"].to(device)
            case = prepared_cases[case_index]
            case_index += 1
            case_diffusion_model = self._resolve_diffusion_model(diffusion_model, case, device)
            case_diffusion_model.eval()

            z_known = autoencoder_model.encode_stage_2_inputs(voided) * self.scale_factor
            z_voided = z_known if self.concat_voided_latent else None
            z_atlas = None
            if self.concat_atlas_latent:
                atlas = batch.get("atlas")
                if atlas is None:
                    raise ValueError("concat_atlas_latent=True requires 'atlas' in inference cases.")
                atlas = atlas.to(device)
                z_atlas = autoencoder_model.encode_stage_2_inputs(atlas) * self.scale_factor
                z_atlas = z_atlas * self.atlas_prior_scale
            mask_latent = self._normalize_mask(mask, target_size=z_known.shape[2:]).to(z_known.dtype)
            z_generated = self._sample_latent(
                case_diffusion_model,
                scheduler,
                z_known,
                mask_latent,
                z_voided,
                z_atlas,
                device,
            )
            decoded = autoencoder_model.decode_stage_2_outputs(z_generated / self.scale_factor)
            mask_img = self._normalize_mask(mask, target_size=image.shape[2:]).to(image.dtype)
            inpainted_mni = voided * (1.0 - mask_img) + decoded * mask_img

            batch_size_actual = image.shape[0]
            for index in range(batch_size_actual):
                case_name = batch["name"][index] if isinstance(batch["name"], list) else f"case_{index:04d}"
                case_dir = out_root / case_name
                case_dir.mkdir(parents=True, exist_ok=True)

                image_mni_path = batch["image_mni_path"][index] if isinstance(batch["image_mni_path"], list) else batch["image_mni_path"]
                voided_mni_path = batch.get("voided_mni_path", "")
                if isinstance(voided_mni_path, list):
                    voided_mni_path = voided_mni_path[index]
                mask_mni_path = batch.get("mask_mni_path", "")
                if isinstance(mask_mni_path, list):
                    mask_mni_path = mask_mni_path[index]
                original_image_path = (
                    batch["original_image_path"][index]
                    if isinstance(batch["original_image_path"], list)
                    else batch["original_image_path"]
                )
                transform_path = batch["transform_path"][index] if isinstance(batch["transform_path"], list) else batch["transform_path"]

                mni_output = case_dir / "inpainted_mni.nii.gz"
                voided_mni_output = case_dir / "voided_mni.nii.gz"
                mask_mni_output = case_dir / "mask_mni.nii.gz"
                native_output = case_dir / "inpainted_native.nii.gz"

                # Invert CenterSpatialCropd to recover full MNI grid position.
                source_voided_meta = batch["voided"][index]
                decoded_crop = decoded[index]  # (1, D, H, W)
                inverted_np = _invert_crop_to_full_mni(
                    decoded_crop=decoded_crop,
                    source_voided_meta=source_voided_meta,
                    preprocessing=preprocessing,
                )

                if self.save_mni_output:
                    # Save inpainted MNI: use inverted result if available, else fallback to cropped patch.
                    ref_mni = str(voided_mni_path) if voided_mni_path and Path(str(voided_mni_path)).exists() else image_mni_path
                    if inverted_np is not None:
                        ref_nii = nib.load(ref_mni)
                        mni_output.parent.mkdir(parents=True, exist_ok=True)
                        nib.save(
                            nib.Nifti1Image(inverted_np, ref_nii.affine, ref_nii.header),
                            str(mni_output),
                        )
                    else:
                        output_np = inpainted_mni[index, 0].detach().cpu().numpy().astype(np.float32)
                        _save_nifti_like(output_np, ref_mni, mni_output)

                    if voided_mni_path and Path(str(voided_mni_path)).exists():
                        _copy_nifti(str(voided_mni_path), voided_mni_output)
                    else:
                        voided_np = voided[index, 0].detach().cpu().numpy().astype(np.float32)
                        _save_nifti_like(voided_np, image_mni_path, voided_mni_output)

                    if mask_mni_path and Path(str(mask_mni_path)).exists():
                        _copy_nifti(str(mask_mni_path), mask_mni_output)
                    else:
                        mask_np = (mask[index, 0].detach().cpu().numpy() > 0.5).astype(np.float32)
                        _save_nifti_like(mask_np, image_mni_path, mask_mni_output)

                if original_image_path and transform_path:
                    apply_transform(
                        image_path=str(mni_output),
                        reference_path=original_image_path,
                        transform_path=transform_path,
                        output_path=str(native_output),
                        is_mask=False,
                        inverse=True,
                    )