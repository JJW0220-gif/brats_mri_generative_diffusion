from __future__ import annotations

import torch
import torch.nn.functional as F


class UncondInpaintSampler:
    """Unconditional inpainting sampler that keeps the 8-channel UNet input unchanged.

    The diffusion model is run exactly as an 8-channel latent denoiser. During each DDIM
    denoising step, the known region is explicitly restored in latent space by mixing the
    predicted latent with the known latent according to the mask. The non-masked region is
    then preserved again in image space after decoding.
    """

    def __init__(self, num_inference_steps: int = 50, scale_factor: float | None = None) -> None:
        self.num_inference_steps = num_inference_steps
        self.scale_factor = scale_factor

    @staticmethod
    def _prepare_mask(mask: torch.Tensor, target_size: tuple[int, ...]) -> torch.Tensor:
        mask = F.interpolate(mask.float(), size=target_size, mode="nearest")
        return (mask > 0.5).to(dtype=torch.float32)

    @torch.no_grad()
    def run(
        self,
        autoencoder,
        diffusion,
        scheduler,
        voided: torch.Tensor,
        mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        voided = voided.to(device)
        mask = mask.to(device)

        # Encode known image to latent space.
        z_known = autoencoder.encode_stage_2_inputs(voided)

        # Match the latent scale used during training if provided.
        if self.scale_factor is None:
            latent_std = torch.std(z_known).clamp_min(1e-6)
            self.scale_factor = 1.0 / float(latent_std.item())

        z_known = z_known * self.scale_factor

        # Resize mask to latent shape.
        mask_latent = self._prepare_mask(mask, target_size=z_known.shape[2:])

        # Initialize from noise and preserve the known region during denoising.
        z_t = torch.randn_like(z_known, device=device)
        fixed_noise = torch.randn_like(z_known, device=device)
        scheduler.set_timesteps(num_inference_steps=self.num_inference_steps)

        for t in scheduler.timesteps:
            t_tensor = torch.full((z_t.shape[0],), int(t), device=device, dtype=torch.long)

            # Keep the UNet input structure unchanged: 8-channel latent only.
            model_output = diffusion(z_t, timesteps=t_tensor, context=None)
            z_prev, _ = scheduler.step(model_output, int(t), z_t)

            z_known_t = scheduler.add_noise(original_samples=z_known, noise=fixed_noise, timesteps=t_tensor)
            z_t = z_prev * mask_latent + z_known_t * (1.0 - mask_latent)

        # Decode back to image space using the same latent scaling convention as training.
        z_t = z_t / self.scale_factor
        generated = autoencoder.decode_stage_2_outputs(z_t)

        mask_img = self._prepare_mask(mask, target_size=voided.shape[2:]).to(dtype=voided.dtype)
        inpainted = voided * (1.0 - mask_img) + generated * mask_img
        return inpainted
