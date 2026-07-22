from __future__ import annotations

from contextlib import nullcontext
import copy
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
import torch.nn.functional as F
from monai.engines.trainer import SupervisedTrainer
from monai.engines.utils import IterationEvents, default_metric_cmp_fn
from monai.utils import IgniteInfo, min_version, optional_import
from monai.utils.enums import CommonKeys
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from . import losses as seg_losses

Engine, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Engine")
Metric, _ = optional_import("ignite.metrics", IgniteInfo.OPT_IMPORT_VERSION, min_version, "Metric")
EventEnum, _ = optional_import("ignite.engine", IgniteInfo.OPT_IMPORT_VERSION, min_version, "EventEnum")


def prepare_mask_tensor(mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Normalize mask tensor dimensions for interpolation.

    Acceptable input shapes:
    - 5D: (B, C, D, H, W)
    - 4D: (B, C, H, W) or (B, D, H, W) or (C, D, H, W)
    - 3D: (D, H, W) or (C, H, W)
    """
    if mask.ndim == 5:
        return mask

    if mask.ndim == 4:
        if image.ndim == 5:
            if mask.shape[0] == image.shape[0]:
                if mask.shape[1] == image.shape[1]:
                    return mask[:, :, None, ...]
                return mask[:, None, ...]
            if mask.shape[1] == image.shape[1]:
                return mask[None, ...]
        return mask.unsqueeze(0)

    if mask.ndim == 3:
        return mask[None, None, ...]

    raise ValueError(
        f"Unable to prepare mask tensor for interpolation: expected 3-5 dims, got {mask.ndim} dims with shape {tuple(mask.shape)}"
    )


def _compat_autocast_kwargs(device: torch.device, amp_kwargs: dict | None) -> dict:
    kwargs = dict(amp_kwargs or {})
    if "device_type" not in kwargs:
        kwargs["device_type"] = device.type
    return kwargs


def load_mask_conditioned_diffusion(model: torch.nn.Module, ckpt_path: str) -> None:
    if not ckpt_path:
        return
    path = Path(ckpt_path)
    if not path.exists():
        return

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    target_in_channels = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv3d):
            target_in_channels = module.in_channels
            break

    if target_in_channels is None:
        raise ValueError("Could not find a Conv3d layer in diffusion model.")

    expanded_state = dict(state_dict)
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.ndim == 5 and value.shape[1] < target_in_channels:
            expanded = torch.zeros(
                value.shape[0],
                target_in_channels,
                value.shape[2],
                value.shape[3],
                value.shape[4],
                dtype=value.dtype,
            )
            expanded[:, : value.shape[1]] = value
            expanded_state[key] = expanded
            break

    model.load_state_dict(expanded_state, strict=False)


class MaskConditionedLDMTrainer(SupervisedTrainer):
    def __init__(
        self,
        device: str | torch.device,
        max_epochs: int,
        train_data_loader: Iterable | DataLoader,
        network: torch.nn.Module,
        autoencoder_model: torch.nn.Module,
        optimizer: Optimizer,
        loss_function: Callable,
        inferer: Any,
        epoch_length: int | None = None,
        non_blocking: bool = False,
        iteration_update: Callable[[Engine, Any], Any] | None = None,
        postprocessing: Any | None = None,
        key_train_metric: dict[str, Metric] | None = None,
        additional_metrics: dict[str, Metric] | None = None,
        metric_cmp_fn: Callable = default_metric_cmp_fn,
        train_handlers: Sequence | None = None,
        event_names: list[str | EventEnum] | None = None,
        event_to_attr: dict | None = None,
        decollate: bool = True,
        optim_set_to_none: bool = False,
        to_kwargs: dict | None = None,
        amp: bool = False,
        amp_kwargs: dict | None = None,
        scale_factor: float = 1.0,
        concat_voided_latent: bool = False,
        concat_atlas_latent: bool = False,
        atlas_prior_scale: float = 1.0,
        segmentor: torch.nn.Module | None = None,
        seg_guidance_weight: float = 0.0,
        seg_wm_index: int = 1,
        seg_gm_index: int = 2,
        fseg: int = 0,
        seg_denoise_steps: int = 10,
        initial_epoch: int = 0,
    ):
        super().__init__(
            device=device,
            max_epochs=max_epochs,
            train_data_loader=train_data_loader,
            network=network,
            optimizer=optimizer,
            loss_function=loss_function,
            inferer=inferer,
            epoch_length=epoch_length,
            non_blocking=non_blocking,
            iteration_update=iteration_update,
            postprocessing=postprocessing,
            key_train_metric=key_train_metric,
            additional_metrics=additional_metrics,
            metric_cmp_fn=metric_cmp_fn,
            train_handlers=train_handlers,
            event_names=event_names,
            event_to_attr=event_to_attr,
            decollate=decollate,
            to_kwargs=to_kwargs,
            amp=amp,
            amp_kwargs=amp_kwargs,
            optim_set_to_none=optim_set_to_none,
        )
        self.autoencoder_model = autoencoder_model
        self.scale_factor = scale_factor
        self.concat_voided_latent = concat_voided_latent
        self.concat_atlas_latent = concat_atlas_latent
        self.atlas_prior_scale = atlas_prior_scale
        self.segmentor = segmentor
        self.seg_guidance_weight = seg_guidance_weight
        self.seg_wm_index = seg_wm_index
        self.seg_gm_index = seg_gm_index
        self.fseg = fseg
        self.seg_denoise_steps = seg_denoise_steps
        self.initial_epoch = initial_epoch
        self.state.epoch = initial_epoch
        self.state.iteration = 0
        if self.segmentor is not None:
            self.segmentor.eval()
            for p in self.segmentor.parameters():
                p.requires_grad = False

    def _build_model_input(
        self,
        z_t: torch.Tensor,
        mask_latent: torch.Tensor,
        voided_latent: torch.Tensor,
        atlas_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        model_input = [z_t, mask_latent]
        if self.concat_voided_latent:
            model_input.append(voided_latent)
        if self.concat_atlas_latent:
            if atlas_latent is None:
                raise ValueError("concat_atlas_latent=True but atlas_latent was not computed.")
            model_input.append(atlas_latent)
        return torch.cat(model_input, dim=1)

    def _periodic_seg_guidance_loss(
        self,
        engine: MaskConditionedLDMTrainer,
        image: torch.Tensor,
        noisy_latent: torch.Tensor,
        mask_latent: torch.Tensor,
        voided_latent: torch.Tensor,
        atlas_latent: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.segmentor is None or self.seg_guidance_weight <= 0.0 or self.fseg <= 0:
            return torch.zeros(1, device=image.device, dtype=image.dtype)

        if engine.state.iteration % self.fseg != 0:
            return torch.zeros(1, device=image.device, dtype=image.dtype)

        scheduler = copy.deepcopy(engine.inferer.scheduler)
        if not hasattr(scheduler, "set_timesteps"):
            return torch.zeros(1, device=image.device, dtype=image.dtype)

        scheduler.set_timesteps(num_inference_steps=max(1, self.seg_denoise_steps))
        z_curr = noisy_latent.detach()
        for t in scheduler.timesteps:
            t_tensor = torch.full((z_curr.shape[0],), int(t), device=image.device, dtype=torch.long)
            model_input = self._build_model_input(z_curr, mask_latent, voided_latent, atlas_latent)
            eps_pred = engine.network(model_input, timesteps=t_tensor, context=None)
            z_prev, _ = scheduler.step(eps_pred, int(t), z_curr)
            z_curr = z_prev

        recon = self.autoencoder_model.decode_stage_2_outputs(z_curr / self.scale_factor)
        return seg_losses.anatomical_segmentation_loss(
            segmentor=self.segmentor,
            real_images=image,
            generated_images=recon,
            wm_index=self.seg_wm_index,
            gm_index=self.seg_gm_index,
        )

    def _iteration(self, engine: MaskConditionedLDMTrainer, batchdata: dict[str, torch.Tensor]) -> dict:
        if batchdata is None:
            raise ValueError("Must provide batch data for current iteration.")

        image = batchdata["image"].to(engine.state.device)
        mask = batchdata["mask"].to(engine.state.device)
        if mask.ndim != 5:
            mask = prepare_mask_tensor(mask, image)
        voided = batchdata.get("voided")
        if voided is None:
            voided = image * (1.0 - (mask > 0.5).to(image.dtype))
        else:
            voided = voided.to(engine.state.device)

        engine.state.output = {CommonKeys.IMAGE: image}

        with torch.no_grad():
            healthy_latent = self.autoencoder_model.encode_stage_2_inputs(image) * self.scale_factor
            voided_latent = self.autoencoder_model.encode_stage_2_inputs(voided) * self.scale_factor
            atlas_latent = None
            if self.concat_atlas_latent:
                atlas = batchdata.get("atlas")
                if atlas is None:
                    raise ValueError("concat_atlas_latent=True requires 'atlas' in batch data.")
                atlas = atlas.to(engine.state.device)
                atlas_latent = self.autoencoder_model.encode_stage_2_inputs(atlas) * self.scale_factor
                atlas_latent = atlas_latent * self.atlas_prior_scale

        mask_latent = F.interpolate(mask.float(), size=healthy_latent.shape[2:], mode="nearest")
        mask_latent = (mask_latent > 0.5).to(healthy_latent.dtype)

        noise = torch.randn_like(healthy_latent)
        timesteps = torch.randint(
            0,
            engine.inferer.scheduler.num_train_timesteps,
            (image.shape[0],),
            device=image.device,
        ).long()
        noisy_latent = engine.inferer.scheduler.add_noise(healthy_latent, noise, timesteps)

        model_input = self._build_model_input(noisy_latent, mask_latent, voided_latent, atlas_latent)

        engine.network.train()
        engine.optimizer.zero_grad(set_to_none=engine.optim_set_to_none)

        autocast_context = (
            torch.amp.autocast(**_compat_autocast_kwargs(image.device, engine.amp_kwargs))
            if engine.amp and engine.scaler is not None
            else nullcontext()
        )

        with autocast_context:
            prediction = engine.network(model_input, timesteps=timesteps, context=None)
            noise_loss = engine.loss_function(prediction, noise).mean()
            seg_loss = self._periodic_seg_guidance_loss(
                engine=engine,
                image=image,
                noisy_latent=noisy_latent,
                mask_latent=mask_latent,
                voided_latent=voided_latent,
                atlas_latent=atlas_latent,
            )
            loss = noise_loss + self.seg_guidance_weight * seg_loss
            engine.state.output[CommonKeys.PRED] = prediction
            engine.state.output[CommonKeys.LOSS] = loss
            engine.state.output["noise_loss"] = noise_loss
            engine.state.output["seg_loss"] = seg_loss
            engine.fire_event(IterationEvents.FORWARD_COMPLETED)
            engine.fire_event(IterationEvents.LOSS_COMPLETED)

        if engine.amp and engine.scaler is not None:
            engine.scaler.scale(loss).backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            engine.scaler.step(engine.optimizer)
            engine.scaler.update()
        else:
            loss.backward()
            engine.fire_event(IterationEvents.BACKWARD_COMPLETED)
            engine.optimizer.step()

        engine.fire_event(IterationEvents.MODEL_COMPLETED)
        return engine.state.output