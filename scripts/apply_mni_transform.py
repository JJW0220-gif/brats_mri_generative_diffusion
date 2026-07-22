from __future__ import annotations

import argparse
from pathlib import Path


def _require_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError(
            "SimpleITK is required to apply MNI transforms. Install it with `pip install SimpleITK`."
        ) from error
    return sitk


def apply_transform(
    image_path: str,
    reference_path: str,
    transform_path: str,
    output_path: str,
    is_mask: bool = False,
    inverse: bool = False,
    default_value: float = 0.0,
) -> str:
    sitk = _require_sitk()

    image = sitk.ReadImage(str(image_path), sitk.sitkFloat32)
    reference = sitk.ReadImage(str(reference_path), sitk.sitkFloat32)
    transform = sitk.ReadTransform(str(transform_path))
    if inverse:
        transform = transform.GetInverse()

    interpolator = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    resampled = sitk.Resample(
        image,
        reference,
        transform,
        interpolator,
        default_value,
        image.GetPixelID(),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(resampled, str(output))
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply an MNI transform to MRI or mask volumes.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--transform", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--is_mask", action="store_true")
    parser.add_argument("--inverse", action="store_true")
    parser.add_argument("--default_value", type=float, default=0.0)
    args = parser.parse_args()

    apply_transform(
        image_path=args.image,
        reference_path=args.reference,
        transform_path=args.transform,
        output_path=args.output,
        is_mask=args.is_mask,
        inverse=args.inverse,
        default_value=args.default_value,
    )


if __name__ == "__main__":
    main()