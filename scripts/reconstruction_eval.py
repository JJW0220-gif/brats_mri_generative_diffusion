from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from monai.data import DataLoader, Dataset

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from generative.networks.nets import AutoencoderKL
from scripts.mni_transforms import collect_mni_autoencoder_cases, make_autoencoder_mni_transforms


def _load_ckpt_flexible(model: torch.nn.Module, ckpt_path: str) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if hasattr(model, "load_old_state_dict"):
        model.load_old_state_dict(checkpoint)
        return
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state_dict, strict=False)


def build_autoencoder(device: torch.device) -> AutoencoderKL:
    return AutoencoderKL(
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


@torch.no_grad()
def evaluate(autoencoder: AutoencoderKL, loader: DataLoader, device: torch.device, limit: int = 20) -> dict:
    autoencoder.eval()
    errors = []
    for index, batch in enumerate(loader):
        if index >= limit:
            break
        image = batch["image"].to(device)
        latent = autoencoder.encode_stage_2_inputs(image)
        recon = autoencoder.decode_stage_2_outputs(latent)
        error = torch.mean(torch.abs(recon - image)).item()
        errors.append(error)
    return {"mean_l1": float(np.mean(errors)), "num_cases": len(errors)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare autoencoder reconstruction quality on cached MNI cases.")
    parser.add_argument("--mni_cache_dir", required=True)
    parser.add_argument("--autoencoder_ckpt", required=True)
    parser.add_argument("--output_json", default="")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = collect_mni_autoencoder_cases(args.mni_cache_dir)
    dataset = Dataset(data=cases, transform=make_autoencoder_mni_transforms())
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    autoencoder = build_autoencoder(device)
    _load_ckpt_flexible(autoencoder, args.autoencoder_ckpt)
    metrics = evaluate(autoencoder, loader, device, limit=args.limit)
    print(json.dumps(metrics, indent=2))
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()