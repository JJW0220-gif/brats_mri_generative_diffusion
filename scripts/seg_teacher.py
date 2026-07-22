from __future__ import annotations

from pathlib import Path
from typing import Any
import urllib.request
import zipfile

import torch
from monai.bundle import ConfigParser

try:
    from monai.apps import download_url as _monai_download_url  # type: ignore
except Exception:
    _monai_download_url = None

try:
    from monai.apps import extractbundle as _monai_extractbundle  # type: ignore
except Exception:
    _monai_extractbundle = None


def _download_file(url: str, dst: str) -> None:
    if _monai_download_url is not None:
        _monai_download_url(url, dst)
        return
    urllib.request.urlretrieve(url, dst)


def _extract_bundle(zip_path: str, out_dir: str) -> None:
    if _monai_extractbundle is not None:
        _monai_extractbundle(zip_path, out_dir)
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _prepare_bundle(bundle_url: str, bundle_zip_path: str, bundle_dir: str) -> None:
    if not bundle_url:
        return

    bundle_root = Path(bundle_dir)
    if bundle_root.exists() and any(bundle_root.iterdir()):
        return

    zip_path = Path(bundle_zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_root.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        _download_file(bundle_url, str(zip_path))
    _extract_bundle(str(zip_path), str(bundle_root))


def get_frozen_monai_bundle_segmentor(
    bundle_url: str = "",
    bundle_zip_path: str = "",
    bundle_dir: str = "",
    config_rel_path: str = "configs/inference.json",
    checkpoint_rel_path: str = "models/model.pt",
    network_key: str = "network_def",
    device: str | torch.device = "cpu",
) -> torch.nn.Module | None:
    """Load a frozen MONAI bundle segmentor.

    If bundle_url is provided and bundle_dir is missing/empty, the bundle is downloaded
    and extracted via monai.apps.download_url/extractbundle.
    Returns None when config/checkpoint cannot be resolved.
    """

    if not bundle_dir:
        return None

    _prepare_bundle(bundle_url=bundle_url, bundle_zip_path=bundle_zip_path, bundle_dir=bundle_dir)

    root = Path(bundle_dir)
    config_path = root / config_rel_path
    checkpoint_path = root / checkpoint_rel_path
    if not config_path.exists() or not checkpoint_path.exists():
        # Some zip bundles extract into one additional top-level folder.
        for child in root.iterdir() if root.exists() else []:
            if not child.is_dir():
                continue
            alt_config = child / config_rel_path
            alt_ckpt = child / checkpoint_rel_path
            if alt_config.exists() and alt_ckpt.exists():
                root = child
                config_path = alt_config
                checkpoint_path = alt_ckpt
                break
    if not config_path.exists() or not checkpoint_path.exists():
        return None

    parser = ConfigParser()
    parser.read_config(str(config_path))

    try:
        network: Any = parser.get_parsed_content(network_key)
    except Exception:
        return None

    if network is None or not isinstance(network, torch.nn.Module):
        return None

    resolved_device = _resolve_device(device)
    network = network.to(resolved_device)

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    network.load_state_dict(state_dict, strict=False)
    network.eval()
    for param in network.parameters():
        param.requires_grad = False
    return network
