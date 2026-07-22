from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import nibabel as nib
import numpy as np

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.apply_mni_transform import apply_transform
from scripts.register_to_mni import register_to_mni


def _box_blur3d(volume: np.ndarray, radius: int = 2, passes: int = 2) -> np.ndarray:
    out = volume.astype(np.float32)
    if radius <= 0:
        return out
    kernel_size = 2 * radius + 1
    kernel_vol = float(kernel_size**3)

    for _ in range(max(1, passes)):
        padded = np.pad(out, ((radius, radius), (radius, radius), (radius, radius)), mode="edge")
        acc = np.zeros_like(out, dtype=np.float32)
        for dz in range(kernel_size):
            for dy in range(kernel_size):
                for dx in range(kernel_size):
                    acc += padded[dz : dz + out.shape[0], dy : dy + out.shape[1], dx : dx + out.shape[2]]
        out = acc / kernel_vol
    return out


def _compute_blend_masks(
    unhealthy_mask: np.ndarray,
    healthy_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    unhealthy_f = unhealthy_mask.astype(np.float32)
    soft_mask = np.clip(_box_blur3d(unhealthy_f, radius=2, passes=3), 0.0, 1.0)
    soft_mask = np.maximum(soft_mask, unhealthy_f)

    ring_mask = (soft_mask > 0.0) & (~unhealthy_mask)
    if healthy_mask is not None:
        ring_mask &= healthy_mask
    if not np.any(ring_mask):
        ring_mask = (~unhealthy_mask) if healthy_mask is None else ((~unhealthy_mask) & healthy_mask)
    if not np.any(ring_mask):
        ring_mask = ~unhealthy_mask
    return soft_mask, ring_mask


def _match_source_to_local_stats(source: np.ndarray, image: np.ndarray, ring_mask: np.ndarray) -> np.ndarray:
    source_values = source[ring_mask]
    image_values = image[ring_mask]
    source_std = float(source_values.std()) if source_values.size > 0 else 0.0
    image_std = float(image_values.std()) if image_values.size > 0 else 0.0
    if source_std > 1e-6 and image_std > 1e-6:
        scale = image_std / source_std
    else:
        scale = 1.0
    scale = float(np.clip(scale, 0.5, 1.5))
    shift = float(image_values.mean()) - scale * float(source_values.mean()) if image_values.size > 0 else 0.0
    matched = source * scale + shift

    if image_values.size > 16:
        lo = float(np.percentile(image_values, 0.5))
        hi = float(np.percentile(image_values, 99.5))
        matched = np.clip(matched, lo, hi)
    return matched.astype(np.float32)


def _build_mirror_source(
    image: np.ndarray,
    unhealthy_mask: np.ndarray,
    healthy_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    # MNI cache is expected to be consistently oriented; axis 0 is treated as L/R mirror axis.
    mirrored_image = np.flip(image, axis=0)
    mirrored_unhealthy = np.flip(unhealthy_mask, axis=0)
    valid_mirror = ~mirrored_unhealthy
    if healthy_mask is not None:
        valid_mirror &= np.flip(healthy_mask, axis=0)

    unhealthy_voxels = int(np.sum(unhealthy_mask))
    if unhealthy_voxels > 0:
        mirror_core_valid_fraction = float(np.sum(valid_mirror & unhealthy_mask) / unhealthy_voxels)
    else:
        mirror_core_valid_fraction = 1.0

    return mirrored_image.astype(np.float32), valid_mirror, mirror_core_valid_fraction


def _pick_first_3d(candidates: List[Path]) -> Path | None:
    for path in sorted(candidates):
        try:
            if len(nib.load(str(path)).shape) == 3:
                return path
        except Exception:
            continue
    return None


def _find_optional_mask(patient_dir: Path, patterns: List[str]) -> Path | None:
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(patient_dir.glob(pattern))
    return _pick_first_3d(candidates)


def collect_brats_cases(data_dir: str) -> List[Dict[str, str]]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"BraTS data directory not found: {root}")

    cases: List[Dict[str, str]] = []
    for patient_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        image_path = _pick_first_3d(
            list(patient_dir.glob("*-t1n.nii.gz")) + [p for p in patient_dir.glob("*_t1n.nii.gz")]
        )
        if image_path is None:
            continue

        mask_path = _pick_first_3d(list(patient_dir.glob("*-mask.nii.gz")) + list(patient_dir.glob("*_mask.nii.gz")))
        healthy_mask_path = _find_optional_mask(
            patient_dir,
            [
                "*healthy*mask*.nii.gz",
                "*mask*healthy*.nii.gz",
                "*healthy_mask*.nii.gz",
                "*mask_healthy*.nii.gz",
            ],
        )
        unhealthy_mask_path = _find_optional_mask(
            patient_dir,
            [
                "*unhealthy*mask*.nii.gz",
                "*mask*unhealthy*.nii.gz",
                "*unhealthy_mask*.nii.gz",
                "*mask_unhealthy*.nii.gz",
            ],
        )
        voided_path = _pick_first_3d(
            list(patient_dir.glob("*-t1n-voided.nii.gz")) + list(patient_dir.glob("*_voided.nii.gz"))
        )
        cases.append(
            {
                "name": patient_dir.name,
                "image": str(image_path),
                "mask": str(mask_path) if mask_path is not None else "",
                "healthy_mask": str(healthy_mask_path) if healthy_mask_path is not None else "",
                "unhealthy_mask": str(unhealthy_mask_path) if unhealthy_mask_path is not None else "",
                "voided": str(voided_path) if voided_path is not None else "",
            }
        )

    if not cases:
        raise ValueError(f"No BraTS t1n cases found under: {root}")
    return cases


def _create_voided_from_image_and_mask(image_path: Path, mask_path: Path, output_path: Path) -> None:
    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    image = image_nii.get_fdata().astype(np.float32)
    mask = (mask_nii.get_fdata() > 0.5).astype(np.float32)
    voided = image * (1.0 - mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(voided, image_nii.affine, image_nii.header), str(output_path))


def _create_pseudo_healthy_from_unhealthy_mask(
    image_path: Path,
    unhealthy_mask_path: Path,
    template_path: Path,
    output_path: Path,
    healthy_mask_path: Path | None = None,
    pseudo_healthy_mode: str = "hybrid",
) -> Dict[str, float | str]:

    image_nii = nib.load(str(image_path))
    image = image_nii.get_fdata().astype(np.float32)
    template = nib.load(str(template_path)).get_fdata().astype(np.float32)
    unhealthy_mask = nib.load(str(unhealthy_mask_path)).get_fdata() > 0.5
    healthy_mask = None
    if healthy_mask_path is not None and healthy_mask_path.exists():
        healthy_mask = nib.load(str(healthy_mask_path)).get_fdata() > 0.5

    if template.shape != image.shape:
        raise ValueError(
            f"Template/image shape mismatch for pseudo-healthy fill: template={template.shape}, image={image.shape}."
        )

    soft_mask, ring_mask = _compute_blend_masks(unhealthy_mask, healthy_mask)
    template_matched = _match_source_to_local_stats(template, image, ring_mask)

    mirror_image, mirror_valid_mask, mirror_core_valid_fraction = _build_mirror_source(
        image=image,
        unhealthy_mask=unhealthy_mask,
        healthy_mask=healthy_mask,
    )
    mirror_matched = _match_source_to_local_stats(mirror_image, image, ring_mask)

    if pseudo_healthy_mode not in {"template", "mirror", "hybrid"}:
        raise ValueError(f"Unsupported pseudo_healthy_mode: {pseudo_healthy_mode}")

    if pseudo_healthy_mode == "template":
        fill_source = template_matched
        source_is_template = np.ones_like(unhealthy_mask, dtype=bool)
    elif pseudo_healthy_mode == "mirror":
        # Mirror-only mode still protects against invalid mirrored voxels by falling back to template.
        source_is_template = ~mirror_valid_mask
        fill_source = np.where(source_is_template, template_matched, mirror_matched)
    else:
        # Hybrid mode prefers mirrored anatomy and falls back to template for unreliable mirrored positions.
        source_is_template = ~mirror_valid_mask
        fill_source = np.where(source_is_template, template_matched, mirror_matched)

    pseudo_healthy = image.copy()
    blend_region = soft_mask > 0.0
    pseudo_healthy[blend_region] = (
        image[blend_region] * (1.0 - soft_mask[blend_region]) + fill_source[blend_region] * soft_mask[blend_region]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(pseudo_healthy, image_nii.affine, image_nii.header), str(output_path))

    blend_weights = soft_mask[blend_region]
    weight_sum = float(np.sum(blend_weights)) if blend_weights.size > 0 else 0.0
    if weight_sum > 0.0:
        template_weight = float(np.sum(blend_weights * source_is_template[blend_region].astype(np.float32)))
        template_fraction = template_weight / weight_sum
    else:
        template_fraction = 1.0 if pseudo_healthy_mode == "template" else 0.0

    return {
        "pseudo_healthy_mode": pseudo_healthy_mode,
        "pseudo_healthy_template_weight_fraction": float(np.clip(template_fraction, 0.0, 1.0)),
        "pseudo_healthy_mirror_weight_fraction": float(np.clip(1.0 - template_fraction, 0.0, 1.0)),
        "pseudo_healthy_mirror_core_valid_fraction": mirror_core_valid_fraction,
    }


def _resolve_template_path(case_dir: Path, fallback_template_path: str | None) -> Path:
    metadata_path = case_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            meta_template = metadata.get("template_path", "")
            if meta_template:
                candidate = Path(meta_template)
                if candidate.exists():
                    return candidate
        except Exception:
            pass

    if fallback_template_path:
        candidate = Path(fallback_template_path)
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve template path for {case_dir}. Provide --template_path or ensure metadata.json has a valid template_path."
    )


def _update_case_metadata(case_dir: Path, updates: Dict[str, object]) -> Dict[str, object]:
    metadata_path = case_dir / "metadata.json"
    metadata: Dict[str, object] = {}
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}

    metadata.update(updates)
    metadata.setdefault("case_name", case_dir.name)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def fill_pseudo_healthy_for_aligned_case(
    case_dir: Path,
    fallback_template_path: str | None = None,
    overwrite: bool = False,
    pseudo_healthy_mode: str = "hybrid",
) -> Dict[str, str]:
    image_mni_path = case_dir / "t1n_mni.nii.gz"
    unhealthy_mask_mni_path = case_dir / "unhealthy_mask_mni.nii.gz"
    healthy_mask_mni_path = case_dir / "healthy_mask_mni.nii.gz"
    pseudo_healthy_mni_path = case_dir / "t1n_pseudo_healthy_mni.nii.gz"

    if not image_mni_path.exists() or not unhealthy_mask_mni_path.exists():
        return {
            "case_name": case_dir.name,
            "status": "skipped",
            "reason": "missing t1n_mni.nii.gz or unhealthy_mask_mni.nii.gz",
        }

    if overwrite or not pseudo_healthy_mni_path.exists():
        template_path = _resolve_template_path(case_dir, fallback_template_path)
        fill_stats = _create_pseudo_healthy_from_unhealthy_mask(
            image_path=image_mni_path,
            unhealthy_mask_path=unhealthy_mask_mni_path,
            template_path=template_path,
            output_path=pseudo_healthy_mni_path,
            healthy_mask_path=healthy_mask_mni_path if healthy_mask_mni_path.exists() else None,
            pseudo_healthy_mode=pseudo_healthy_mode,
        )
    else:
        fill_stats = {
            "pseudo_healthy_mode": pseudo_healthy_mode,
        }

    metadata = _update_case_metadata(
        case_dir,
        {
            "image_mni_path": str(image_mni_path),
            "unhealthy_mask_mni_path": str(unhealthy_mask_mni_path),
            "healthy_mask_mni_path": str(healthy_mask_mni_path) if healthy_mask_mni_path.exists() else "",
            "pseudo_healthy_mni_path": str(pseudo_healthy_mni_path) if pseudo_healthy_mni_path.exists() else "",
            **fill_stats,
        },
    )
    metadata["status"] = "ok"
    return metadata


def fill_pseudo_healthy_for_aligned_cache(
    aligned_cache_dir: str,
    fallback_template_path: str | None = None,
    overwrite: bool = False,
    limit: int = 0,
    pseudo_healthy_mode: str = "hybrid",
) -> List[Dict[str, str]]:
    root = Path(aligned_cache_dir)
    if not root.exists():
        raise FileNotFoundError(f"Aligned cache directory not found: {root}")

    case_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if limit > 0:
        case_dirs = case_dirs[:limit]

    return [
        fill_pseudo_healthy_for_aligned_case(
            case_dir=case_dir,
            fallback_template_path=fallback_template_path,
            overwrite=overwrite,
            pseudo_healthy_mode=pseudo_healthy_mode,
        )
        for case_dir in case_dirs
    ]


def build_case_cache(
    case: Dict[str, str],
    template_path: str,
    output_dir: str,
    overwrite: bool = False,
    pseudo_healthy_mode: str = "hybrid",
) -> Dict[str, str]:
    case_dir = Path(output_dir) / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)

    transform_path = case_dir / "mri_to_mni_transform.tfm"
    image_mni_path = case_dir / "t1n_mni.nii.gz"
    mask_mni_path = case_dir / "mask_mni.nii.gz"
    healthy_mask_mni_path = case_dir / "healthy_mask_mni.nii.gz"
    unhealthy_mask_mni_path = case_dir / "unhealthy_mask_mni.nii.gz"
    voided_mni_path = case_dir / "t1n_voided_mni.nii.gz"
    pseudo_healthy_mni_path = case_dir / "t1n_pseudo_healthy_mni.nii.gz"

    if overwrite or not transform_path.exists():
        register_to_mni(
            moving_path=case["image"],
            template_path=template_path,
            output_transform_path=str(transform_path),
            output_warped_path=str(image_mni_path),
        )

    if (overwrite or not image_mni_path.exists()) and not transform_path.exists():
        raise FileNotFoundError(f"Missing transform after registration: {transform_path}")

    if overwrite or not image_mni_path.exists():
        apply_transform(
            image_path=case["image"],
            reference_path=template_path,
            transform_path=str(transform_path),
            output_path=str(image_mni_path),
            is_mask=False,
        )

    if case.get("mask"):
        if overwrite or not mask_mni_path.exists():
            apply_transform(
                image_path=case["mask"],
                reference_path=template_path,
                transform_path=str(transform_path),
                output_path=str(mask_mni_path),
                is_mask=True,
            )

        if case.get("healthy_mask"):
            if overwrite or not healthy_mask_mni_path.exists():
                apply_transform(
                    image_path=case["healthy_mask"],
                    reference_path=template_path,
                    transform_path=str(transform_path),
                    output_path=str(healthy_mask_mni_path),
                    is_mask=True,
                )

        if case.get("unhealthy_mask"):
            if overwrite or not unhealthy_mask_mni_path.exists():
                apply_transform(
                    image_path=case["unhealthy_mask"],
                    reference_path=template_path,
                    transform_path=str(transform_path),
                    output_path=str(unhealthy_mask_mni_path),
                    is_mask=True,
                )

        if case.get("voided"):
            if overwrite or not voided_mni_path.exists():
                apply_transform(
                    image_path=case["voided"],
                    reference_path=template_path,
                    transform_path=str(transform_path),
                    output_path=str(voided_mni_path),
                    is_mask=False,
                )
        elif overwrite or not voided_mni_path.exists():
            _create_voided_from_image_and_mask(image_mni_path, mask_mni_path, voided_mni_path)

        if unhealthy_mask_mni_path.exists() and (overwrite or not pseudo_healthy_mni_path.exists()):
            fill_stats = _create_pseudo_healthy_from_unhealthy_mask(
                image_path=image_mni_path,
                unhealthy_mask_path=unhealthy_mask_mni_path,
                template_path=Path(template_path),
                healthy_mask_path=healthy_mask_mni_path if healthy_mask_mni_path.exists() else None,
                output_path=pseudo_healthy_mni_path,
                pseudo_healthy_mode=pseudo_healthy_mode,
            )
        else:
            fill_stats = {
                "pseudo_healthy_mode": pseudo_healthy_mode,
            }
    else:
        fill_stats = {
            "pseudo_healthy_mode": pseudo_healthy_mode,
        }

    metadata = {
        "case_name": case["name"],
        "template_path": str(template_path),
        "original_image_path": case["image"],
        "original_mask_path": case.get("mask", ""),
        "original_healthy_mask_path": case.get("healthy_mask", ""),
        "original_unhealthy_mask_path": case.get("unhealthy_mask", ""),
        "original_voided_path": case.get("voided", ""),
        "image_mni_path": str(image_mni_path),
        "mask_mni_path": str(mask_mni_path) if mask_mni_path.exists() else "",
        "healthy_mask_mni_path": str(healthy_mask_mni_path) if healthy_mask_mni_path.exists() else "",
        "unhealthy_mask_mni_path": str(unhealthy_mask_mni_path) if unhealthy_mask_mni_path.exists() else "",
        "voided_mni_path": str(voided_mni_path) if voided_mni_path.exists() else "",
        "pseudo_healthy_mni_path": str(pseudo_healthy_mni_path) if pseudo_healthy_mni_path.exists() else "",
        "transform_path": str(transform_path),
        **fill_stats,
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cached MNI-space BraTS dataset.")
    parser.add_argument("--data_dir", default="", help="BraTS root directory containing case folders.")
    parser.add_argument("--template_path", default="", help="Path to the MNI template volume.")
    parser.add_argument("--output_dir", default="", help="Directory where cached MNI cases will be stored.")
    parser.add_argument(
        "--aligned_cache_dir",
        default="",
        help="If set, run fill-only mode on an existing aligned MNI cache and only generate pseudo-healthy targets.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick smoke tests.")
    parser.add_argument(
        "--pseudo_healthy_mode",
        default="hybrid",
        choices=["hybrid", "mirror", "template"],
        help="Pseudo-healthy fill mode: mirror (symmetric prior), template, or hybrid fallback.",
    )
    args = parser.parse_args()

    if args.aligned_cache_dir:
        summary = fill_pseudo_healthy_for_aligned_cache(
            aligned_cache_dir=args.aligned_cache_dir,
            fallback_template_path=args.template_path or None,
            overwrite=args.overwrite,
            limit=args.limit,
            pseudo_healthy_mode=args.pseudo_healthy_mode,
        )
        summary_path = Path(args.aligned_cache_dir) / "index.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        ok_count = sum(1 for item in summary if item.get("status") == "ok")
        print(f"Generated pseudo-healthy targets for {ok_count}/{len(summary)} cases under: {args.aligned_cache_dir}")
        return

    if not args.data_dir or not args.output_dir or not args.template_path:
        raise ValueError("Standard mode requires --data_dir, --template_path, and --output_dir.")

    cases = collect_brats_cases(args.data_dir)
    if args.limit > 0:
        cases = cases[: args.limit]

    summary = [
        build_case_cache(
            case,
            args.template_path,
            args.output_dir,
            overwrite=args.overwrite,
            pseudo_healthy_mode=args.pseudo_healthy_mode,
        )
        for case in cases
    ]
    summary_path = Path(args.output_dir) / "index.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Built {len(summary)} cached MNI cases under: {args.output_dir}")


if __name__ == "__main__":
    main()