from pathlib import Path
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np
from monai import transforms
from monai.transforms import MapTransform


def _resolve_local_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _first_existing(case_dir: Path, names: Sequence[str]) -> Optional[Path]:
    for name in names:
        candidate = case_dir / name
        if candidate.exists():
            return candidate
    return None


def _read_metadata(case_dir: Path) -> Dict[str, str]:
    metadata_path = case_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        import json

        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_case_dir_native_cases(root: str) -> List[Dict[str, str]]:
    cases: List[Dict[str, str]] = []
    root_path = Path(root)
    for case_dir in sorted(path for path in root_path.iterdir() if path.is_dir()):
        case_name = case_dir.name
        image_candidates = sorted(
            list(case_dir.glob("*-t1n.nii.gz"))
            + list(case_dir.glob("*_t1n.nii.gz"))
            + list(case_dir.glob("*-t1n-voided.nii.gz"))
            + list(case_dir.glob("*_t1n_voided.nii.gz"))
            + list(case_dir.glob("*voided*.nii.gz"))
        )
        if not image_candidates:
            continue

        mask_candidates = sorted(list(case_dir.glob("*-mask*.nii.gz")) + list(case_dir.glob("*_mask*.nii.gz")))
        unhealthy_candidates = sorted(
            list(case_dir.glob("*unhealthy*mask*.nii.gz")) + list(case_dir.glob("*mask*unhealthy*.nii.gz"))
        )
        healthy_candidates = sorted(
            list(case_dir.glob("*healthy*mask*.nii.gz")) + list(case_dir.glob("*mask*healthy*.nii.gz"))
        )
        voided_candidates = sorted(list(case_dir.glob("*-t1n-voided.nii.gz")) + list(case_dir.glob("*voided*.nii.gz")))

        image_path = str(image_candidates[0])
        voided_path = str(voided_candidates[0]) if voided_candidates else ""
        if not voided_path and "voided" in Path(image_path).name:
            voided_path = image_path

        cases.append(
            {
                "name": case_name,
                "image": image_path,
                "mask": str(mask_candidates[0]) if mask_candidates else "",
                "healthy_mask": str(healthy_candidates[0]) if healthy_candidates else "",
                "unhealthy_mask": str(unhealthy_candidates[0]) if unhealthy_candidates else "",
                "voided": voided_path,
            }
        )
    return cases


def _collect_flat_native_cases(root: str) -> List[Dict[str, str]]:
    cases: List[Dict[str, str]] = []
    root_path = Path(root)
    image_candidates = sorted(
        list(root_path.glob("*-t1n.nii.gz"))
        + list(root_path.glob("*_t1n.nii.gz"))
        + list(root_path.glob("*-t1n-voided.nii.gz"))
        + list(root_path.glob("*_t1n_voided.nii.gz"))
        + list(root_path.glob("*voided*.nii.gz"))
    )
    if not image_candidates:
        return cases

    case_name = root_path.name
    image_path = str(image_candidates[0])
    mask_candidates = sorted(list(root_path.glob("*-mask*.nii.gz")) + list(root_path.glob("*_mask*.nii.gz")))
    healthy_candidates = sorted(
        list(root_path.glob("*healthy*mask*.nii.gz")) + list(root_path.glob("*mask*healthy*.nii.gz"))
    )
    unhealthy_candidates = sorted(
        list(root_path.glob("*unhealthy*mask*.nii.gz")) + list(root_path.glob("*mask*unhealthy*.nii.gz"))
    )
    cases.append(
        {
            "name": case_name,
            "image": image_path,
            "mask": str(mask_candidates[0]) if mask_candidates else "",
            "healthy_mask": str(healthy_candidates[0]) if healthy_candidates else "",
            "unhealthy_mask": str(unhealthy_candidates[0]) if unhealthy_candidates else "",
            "voided": image_path,
        }
    )
    return cases


def prepare_mni_inpaint_cases_from_native(
    native_data_dir: str,
    mni_cache_dir: str,
    template_path: str,
    case_name: str = "",
    overwrite_cache: bool = False,
    pseudo_healthy_mode: str = "hybrid",
    prefer_pseudo_healthy: bool = False,
    atlas_path: str = "",
) -> List[Dict[str, str]]:
    from scripts import build_mni_dataset

    root = _resolve_local_path(native_data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Native validation directory not found: {root}")
    template = _resolve_local_path(template_path)
    if not template.exists():
        raise FileNotFoundError(f"MNI template not found: {template}")

    has_case_dirs = any(path.is_dir() for path in root.iterdir())
    if has_case_dirs:
        try:
            cases = build_mni_dataset.collect_brats_cases(str(root))
        except ValueError:
            # Validation directories may only provide voided images + masks per case directory.
            cases = _collect_case_dir_native_cases(str(root))
    else:
        cases = _collect_flat_native_cases(str(root))

    requested_case = case_name.strip()
    if requested_case:
        cases = [case for case in cases if case.get("name", "") == requested_case]
        if not cases:
            raise ValueError(f"Requested case not found under native validation folder: {requested_case}")

    if not cases:
        raise ValueError(f"No native cases found under: {native_data_dir}")

    Path(mni_cache_dir).mkdir(parents=True, exist_ok=True)
    for case in cases:
        build_mni_dataset.build_case_cache(
            case=case,
            template_path=str(template),
            output_dir=mni_cache_dir,
            overwrite=overwrite_cache,
            pseudo_healthy_mode=pseudo_healthy_mode,
        )

    return collect_mni_inpaint_cases(
        cache_dir=mni_cache_dir,
        prefer_pseudo_healthy=prefer_pseudo_healthy,
        atlas_path=atlas_path,
        case_name=requested_case,
    )


def _load_mask_array(mask_path: str) -> np.ndarray:
    if not mask_path:
        return np.array([], dtype=np.float32)
    try:
        img = nib.load(mask_path)
        data = np.asarray(img.dataobj)
    except Exception:
        return np.array([], dtype=np.float32)

    if data.ndim == 4:
        if data.shape[-1] == 1:
            data = data[..., 0]
        elif data.shape[0] == 1:
            data = data[0, ...]
        else:
            data = np.squeeze(data)
    if data.ndim != 3:
        data = np.squeeze(data)
    return (data > 0.5).astype(np.float32)


def _infer_mask_region(mask_path: str) -> str:
    mask = _load_mask_array(mask_path)
    if mask.size == 0:
        return "all"

    coords = np.argwhere(mask > 0.0)
    if coords.size == 0:
        return "all"

    shape = np.array(mask.shape, dtype=np.float32)
    centroid = coords.mean(axis=0)
    center = shape / 2.0
    center_dist = np.linalg.norm((centroid - center) / np.maximum(shape, 1.0))
    edge_dist = min(
        float(centroid[0]),
        float(shape[0] - 1 - centroid[0]),
        float(centroid[1]),
        float(shape[1] - 1 - centroid[1]),
        float(centroid[2]),
        float(shape[2] - 1 - centroid[2]),
    )
    edge_norm = edge_dist / max(float(shape.max()), 1.0)

    if center_dist < 0.20:
        return "center"
    if edge_norm < 0.15:
        return "cortical"
    return "deep"


def _infer_mask_size_bucket(mask_path: str) -> str:
    mask = _load_mask_array(mask_path)
    if mask.size == 0:
        return "all"

    voxel_count = int(np.count_nonzero(mask > 0.0))
    total_voxels = int(mask.size)
    ratio = voxel_count / max(total_voxels, 1)
    if ratio < 0.003:
        return "small"
    if ratio < 0.01:
        return "medium"
    return "large"


def collect_mni_autoencoder_cases(cache_dir: str) -> List[Dict[str, str]]:
    root = Path(cache_dir)
    if not root.exists():
        raise FileNotFoundError(f"MNI cache directory not found: {root}")

    cases: List[Dict[str, str]] = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        image_path = _first_existing(case_dir, ["t1n_mni.nii.gz", "image_mni.nii.gz"])
        if image_path is None:
            continue
        cases.append({"image": str(image_path), "name": case_dir.name})

    if not cases:
        raise ValueError(f"No MNI autoencoder cases found under: {root}")
    return cases


def collect_mni_inpaint_cases(
    cache_dir: str,
    prefer_pseudo_healthy: bool = False,
    atlas_path: str = "",
    case_name: str = "",
) -> List[Dict[str, str]]:
    root = Path(cache_dir)
    if not root.exists():
        raise FileNotFoundError(f"MNI cache directory not found: {root}")

    cases: List[Dict[str, str]] = []
    requested_case = case_name.strip()
    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if requested_case:
        case_dirs = [path for path in case_dirs if path.name == requested_case]
        if not case_dirs:
            raise ValueError(f"Requested case not found under MNI cache: {requested_case}")

    for case_dir in case_dirs:
        metadata = _read_metadata(case_dir)
        target_image_path = _first_existing(case_dir, ["t1n_pseudo_healthy_mni.nii.gz"])
        image_path = target_image_path if prefer_pseudo_healthy and target_image_path is not None else _first_existing(case_dir, ["t1n_mni.nii.gz", "image_mni.nii.gz"])
        original_image_path = _first_existing(case_dir, ["t1n_mni.nii.gz", "image_mni.nii.gz"])
        unhealthy_mask_path = _first_existing(case_dir, ["unhealthy_mask_mni.nii.gz"])
        mask_path = unhealthy_mask_path or _first_existing(case_dir, ["mask_mni.nii.gz"])
        healthy_mask_path = _first_existing(case_dir, ["healthy_mask_mni.nii.gz"])
        voided_path = _first_existing(case_dir, ["t1n_voided_mni.nii.gz", "voided_mni.nii.gz"])
        transform_path = _first_existing(case_dir, ["mri_to_mni_transform.tfm"])
        if image_path is None or mask_path is None:
            continue

        case = {
            "image": str(image_path),
            "mask": str(mask_path),
            "name": case_dir.name,
            "mask_region": _infer_mask_region(str(mask_path)),
            "mask_size_bucket": _infer_mask_size_bucket(str(mask_path)),
            "image_mni_path": str(original_image_path if original_image_path is not None else image_path),
            "target_image_mni_path": str(image_path),
            "mask_mni_path": str(mask_path),
            "unhealthy_mask_mni_path": str(unhealthy_mask_path) if unhealthy_mask_path is not None else "",
            "healthy_mask_mni_path": str(healthy_mask_path) if healthy_mask_path is not None else "",
            "original_image_path": metadata.get("original_image_path", ""),
            "original_mask_path": metadata.get("original_mask_path", ""),
            "original_voided_path": metadata.get("original_voided_path", ""),
            "pseudo_healthy_mni_path": metadata.get("pseudo_healthy_mni_path", ""),
            "transform_path": str(transform_path) if transform_path is not None else "",
            "template_path": metadata.get("template_path", ""),
        }
        case["atlas"] = atlas_path if atlas_path else case["template_path"]
        if voided_path is not None:
            case["voided"] = str(voided_path)
            case["voided_mni_path"] = str(voided_path)
        cases.append(case)

    if not cases:
        raise ValueError(f"No MNI inpainting cases found under: {root}")
    return cases


def collect_mni_inpaint_cases_by_region(
    cache_dir: str,
    region: str = "all",
    size_bucket: str = "all",
    prefer_pseudo_healthy: bool = False,
    atlas_path: str = "",
    case_name: str = "",
) -> List[Dict[str, str]]:
    region_name = (region or "all").lower()
    size_name = (size_bucket or "all").lower()
    cases = collect_mni_inpaint_cases(
        cache_dir=cache_dir,
        prefer_pseudo_healthy=prefer_pseudo_healthy,
        atlas_path=atlas_path,
        case_name=case_name,
    )
    if region_name != "all":
        cases = [case for case in cases if case.get("mask_region", "all").lower() == region_name]
    if size_name != "all":
        cases = [case for case in cases if case.get("mask_size_bucket", "all").lower() == size_name]
    if not cases:
        suffix = ""
        if region_name != "all":
            suffix += f" region '{region_name}'"
        if size_name != "all":
            suffix += f" size '{size_name}'"
        raise ValueError(f"No MNI inpainting cases found{suffix} under: {cache_dir}")
    return cases


def make_autoencoder_mni_transforms(
    roi_size: Sequence[int] = (144, 176, 112),
    image_key: str = "image",
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=[image_key]),
            transforms.EnsureChannelFirstd(keys=[image_key]),
            transforms.EnsureTyped(keys=[image_key]),
            transforms.CenterSpatialCropd(keys=[image_key], roi_size=tuple(roi_size)),
            transforms.ScaleIntensityRangePercentilesd(
                keys=[image_key], lower=0, upper=99.5, b_min=0.0, b_max=1.0
            ),
        ]
    )


class RandomMaskAugmentd(MapTransform):
    """Randomly use one of three mask modes: original lesion, synthetic sphere-only, or mixed masks."""

    def __init__(
        self,
        keys: Sequence[str],
        prob: float = 0.7,
        synthetic_prob: float = 0.6,
        mixed_prob: float = 0.3,
        region_sampling: bool = False,
        region: str = None,
    ) -> None:
        super().__init__(keys)
        self.prob = prob
        self.synthetic_prob = synthetic_prob
        self.mixed_prob = mixed_prob
        self.region_sampling = region_sampling
        self.region = region

    def _ensure_channel_first(self, mask: np.ndarray) -> np.ndarray:
        arr = np.asarray(mask)
        if arr.ndim == 5:
            return arr.astype(np.float32)
        if arr.ndim == 4:
            if arr.shape[0] == 1:
                return arr.astype(np.float32)
            return arr[None, ...].astype(np.float32)
        if arr.ndim == 3:
            return arr[None, None, ...].astype(np.float32)
        raise ValueError(
            f"Unexpected mask shape for RandomMaskAugmentd: {tuple(arr.shape)}; expected 3-5 dims."
        )

    def _dilate_mask(self, mask: np.ndarray, radius: int) -> np.ndarray:
        if radius <= 0:
            return mask.astype(np.float32)

        spatial = np.asarray(mask, dtype=np.float32)
        if spatial.ndim == 4 and spatial.shape[0] == 1:
            spatial = spatial[0]
        if spatial.ndim not in {2, 3}:
            return mask.astype(np.float32)

        output = np.zeros_like(spatial, dtype=np.float32)
        if spatial.ndim == 2:
            for i in range(spatial.shape[0]):
                for j in range(spatial.shape[1]):
                    if spatial[i, j] <= 0.0:
                        continue
                    i0 = max(0, i - radius)
                    i1 = min(spatial.shape[0], i + radius + 1)
                    j0 = max(0, j - radius)
                    j1 = min(spatial.shape[1], j + radius + 1)
                    output[i0:i1, j0:j1] = 1.0
            return output

        for i in range(spatial.shape[0]):
            for j in range(spatial.shape[1]):
                for k in range(spatial.shape[2]):
                    if spatial[i, j, k] <= 0.0:
                        continue
                    i0 = max(0, i - radius)
                    i1 = min(spatial.shape[0], i + radius + 1)
                    j0 = max(0, j - radius)
                    j1 = min(spatial.shape[1], j + radius + 1)
                    k0 = max(0, k - radius)
                    k1 = min(spatial.shape[2], k + radius + 1)
                    output[i0:i1, j0:j1, k0:k1] = 1.0
        return output

    def _deform_mask(self, base: np.ndarray) -> np.ndarray:
        mask = np.asarray(base, dtype=np.float32)
        if not np.any(mask):
            return mask.astype(np.float32)

        keep_channel = mask.ndim == 4 and mask.shape[0] == 1
        spatial = mask[0] if keep_channel else mask
        if spatial.ndim not in {2, 3}:
            return mask.astype(np.float32)

        result = np.asarray(spatial, dtype=np.float32).copy()
        for _ in range(3):
            if np.random.rand() < 0.8:
                axis = int(np.random.randint(0, result.ndim))
                shift = int(np.random.randint(-3, 4))
                if shift != 0 and result.shape[axis] > 1:
                    rolled = np.roll(result, shift=shift, axis=axis)
                    result = np.logical_or(result, rolled).astype(np.float32)
            if np.random.rand() < 0.6:
                radius = int(np.random.randint(1, 4))
                result = self._dilate_mask(result, radius)

        if keep_channel:
            return result[None, ...].astype(np.float32)
        return result.astype(np.float32)

    def _add_random_synthetic_mask(self, base: np.ndarray) -> np.ndarray:
        spatial = np.asarray(base, dtype=np.float32)
        keep_channel = spatial.ndim == 4 and spatial.shape[0] == 1
        if keep_channel:
            spatial = spatial[0]
        if spatial.ndim not in {2, 3}:
            return base.astype(np.float32)

        result = np.zeros_like(spatial, dtype=np.float32)
        shape = result.shape
        n_shapes = int(np.random.randint(1, 4))
        for _ in range(n_shapes):
            shape_kind = np.random.choice(["box", "sphere", "blob"])
            if result.ndim == 2:
                size0 = int(np.random.randint(max(3, shape[0] // 12), max(5, shape[0] // 4)))
                size1 = int(np.random.randint(max(3, shape[1] // 12), max(5, shape[1] // 4)))
                selected_region = None
                if self.region is not None:
                    selected_region = self.region
                elif self.region_sampling:
                    selected_region = np.random.choice(["center", "cortical", "deep"])  # even weighting

                if selected_region == "center":
                    center0 = int(shape[0] // 2 + np.random.randint(-max(1, shape[0] // 10), max(1, shape[0] // 10)))
                    center1 = int(shape[1] // 2 + np.random.randint(-max(1, shape[1] // 10), max(1, shape[1] // 10)))
                elif selected_region == "cortical":
                    center0 = int(np.random.choice([np.random.randint(0, max(2, shape[0] // 8)), np.random.randint(shape[0] - max(2, shape[0] // 8), shape[0])]))
                    center1 = int(np.random.choice([np.random.randint(0, max(2, shape[1] // 8)), np.random.randint(shape[1] - max(2, shape[1] // 8), shape[1])]))
                elif selected_region == "deep":
                    center0 = int(shape[0] // 2 + np.random.randint(-max(1, shape[0] // 6), max(1, shape[0] // 6)))
                    center1 = int(shape[1] // 2 + np.random.randint(-max(1, shape[1] // 6), max(1, shape[1] // 6)))
                else:
                    center0 = int(np.random.randint(0, shape[0]))
                    center1 = int(np.random.randint(0, shape[1]))
                if shape_kind == "box":
                    for i in range(max(0, center0 - size0), min(shape[0], center0 + size0)):
                        for j in range(max(0, center1 - size1), min(shape[1], center1 + size1)):
                            if np.random.rand() < 0.8:
                                result[i, j] = 1.0
                elif shape_kind == "sphere":
                    radius = int(np.random.randint(max(2, min(shape) // 10), max(3, min(shape) // 4)))
                    yy, xx = np.ogrid[: shape[0], : shape[1]]
                    dist = (yy - center0) ** 2 + (xx - center1) ** 2
                    result[dist <= radius * radius] = 1.0
                else:
                    for i in range(max(0, center0 - size0), min(shape[0], center0 + size0)):
                        for j in range(max(0, center1 - size1), min(shape[1], center1 + size1)):
                            if np.random.rand() < 0.3:
                                result[i, j] = 1.0
                    radius = int(np.random.randint(max(2, min(shape) // 10), max(3, min(shape) // 4)))
                    yy, xx = np.ogrid[: shape[0], : shape[1]]
                    dist = (yy - center0) ** 2 + (xx - center1) ** 2
                    result[dist <= radius * radius] = 1.0
            else:
                size0 = int(np.random.randint(max(3, shape[0] // 12), max(5, shape[0] // 4)))
                size1 = int(np.random.randint(max(3, shape[1] // 12), max(5, shape[1] // 4)))
                size2 = int(np.random.randint(max(3, shape[2] // 12), max(5, shape[2] // 4)))
                selected_region = None
                if self.region is not None:
                    selected_region = self.region
                elif self.region_sampling:
                    selected_region = np.random.choice(["center", "cortical", "deep"])  # uniform

                if selected_region == "center":
                    center0 = int(shape[0] // 2 + np.random.randint(-max(1, shape[0] // 10), max(1, shape[0] // 10)))
                    center1 = int(shape[1] // 2 + np.random.randint(-max(1, shape[1] // 10), max(1, shape[1] // 10)))
                    center2 = int(shape[2] // 2 + np.random.randint(-max(1, shape[2] // 10), max(1, shape[2] // 10)))
                elif selected_region == "cortical":
                    axes = list(range(3))
                    edge_axis = np.random.choice(axes)
                    centers = [int(np.random.randint(0, shape[a] // 8)) if a == edge_axis and np.random.rand() < 0.5 else int(shape[a] - np.random.randint(0, shape[a] // 8) - 1) if a == edge_axis else int(np.random.randint(shape[a] // 8, shape[a] - shape[a] // 8)) for a in axes]
                    center0, center1, center2 = centers
                elif selected_region == "deep":
                    center0 = int(shape[0] // 2 + np.random.randint(-max(1, shape[0] // 6), max(1, shape[0] // 6)))
                    center1 = int(shape[1] // 2 + np.random.randint(-max(1, shape[1] // 6), max(1, shape[1] // 6)))
                    center2 = int(shape[2] // 2 + np.random.randint(-max(1, shape[2] // 6), max(1, shape[2] // 6)))
                else:
                    center0 = int(np.random.randint(0, shape[0]))
                    center1 = int(np.random.randint(0, shape[1]))
                    center2 = int(np.random.randint(0, shape[2]))
                if shape_kind == "box":
                    for i in range(max(0, center0 - size0), min(shape[0], center0 + size0)):
                        for j in range(max(0, center1 - size1), min(shape[1], center1 + size1)):
                            for k in range(max(0, center2 - size2), min(shape[2], center2 + size2)):
                                if np.random.rand() < 0.8:
                                    result[i, j, k] = 1.0
                elif shape_kind == "sphere":
                    radius = int(np.random.randint(max(2, min(shape) // 10), max(3, min(shape) // 4)))
                    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
                    dist = (zz - center0) ** 2 + (yy - center1) ** 2 + (xx - center2) ** 2
                    result[dist <= radius * radius] = 1.0
                else:
                    for i in range(max(0, center0 - size0), min(shape[0], center0 + size0)):
                        for j in range(max(0, center1 - size1), min(shape[1], center1 + size1)):
                            for k in range(max(0, center2 - size2), min(shape[2], center2 + size2)):
                                if np.random.rand() < 0.3:
                                    result[i, j, k] = 1.0
                    radius = int(np.random.randint(max(2, min(shape) // 10), max(3, min(shape) // 4)))
                    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
                    dist = (zz - center0) ** 2 + (yy - center1) ** 2 + (xx - center2) ** 2
                    result[dist <= radius * radius] = 1.0

        if keep_channel:
            return result[None, ...].astype(np.float32)
        return result.astype(np.float32)

    def _augment_mask(self, mask: np.ndarray) -> np.ndarray:
        base = (mask > 0.5).astype(np.float32)
        if np.random.rand() < self.prob:
            mode = np.random.choice(["original", "synthetic", "mixed"], p=[1.0 - self.synthetic_prob - self.mixed_prob, self.synthetic_prob, self.mixed_prob])
            if mode == "original":
                return base
            if mode == "synthetic":
                return self._add_random_synthetic_mask(base)
            synthetic = self._add_random_synthetic_mask(base)
            return np.logical_or(base, synthetic).astype(np.float32)
        return base

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key not in d:
                continue
            mask = self._ensure_channel_first(d[key])
            d[key] = self._augment_mask(mask)
        return d


def make_inpaint_mni_transforms(
    roi_size: Sequence[int] = (144, 176, 112),
    include_voided: bool = True,
    include_atlas: bool = False,
    augment_random_masks: bool = False,
    random_mask_prob: float = 0.7,
    synthetic_mask_prob: float = 0.6,
    mixed_mask_prob: float = 0.3,
    region_sampling: bool = False,
    region: str = None,
) -> transforms.Compose:
    keys = ["image", "mask"] + (["voided"] if include_voided else []) + (["atlas"] if include_atlas else [])
    intensity_keys = ["image"] + (["voided"] if include_voided else []) + (["atlas"] if include_atlas else [])
    transform_list = [
        transforms.LoadImaged(keys=keys),
        transforms.EnsureChannelFirstd(keys=keys),
        transforms.EnsureTyped(keys=keys),
        transforms.CenterSpatialCropd(keys=keys, roi_size=tuple(roi_size)),
    ]
    if augment_random_masks:
        transform_list.append(
            RandomMaskAugmentd(
                keys=["mask"],
                prob=random_mask_prob,
                synthetic_prob=synthetic_mask_prob,
                mixed_prob=mixed_mask_prob,
                region_sampling=region_sampling,
                region=region,
            )
        )
    transform_list.extend(
        [
            transforms.ScaleIntensityRangePercentilesd(
                keys=intensity_keys, lower=0, upper=99.5, b_min=0.0, b_max=1.0
            ),
            transforms.ScaleIntensityRanged(
                keys=["mask"], a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True
            ),
        ]
    )
    return transforms.Compose(transform_list)