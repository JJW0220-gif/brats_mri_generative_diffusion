#!/usr/bin/env python
"""Export per-case native inpainting results into a submission-style folder.

The script scans a root directory for files named like ``inpainted_native.nii`` or
``inpainted_native.nii.gz`` under case subfolders and copies them into a target
folder with the naming convention:

    BraTS-GLI-[case_name]-t1n-inference.nii
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import nibabel as nib
import numpy as np


def find_native_outputs(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input root not found: {root}")

    candidates = sorted(root.rglob("inpainted_native*.nii*"))
    if not candidates:
        raise FileNotFoundError(
            f"No native inference outputs found under {root}. Expected files matching 'inpainted_native*.nii*'."
        )
    return [p for p in candidates if p.is_file()]


def build_output_name(case_name: str) -> str:
    case_name = case_name.strip()
    if not case_name:
        raise ValueError("Case name is empty")
    if case_name.startswith("BraTS-GLI-"):
        return f"{case_name}-t1n-inference.nii.gz"
    return f"BraTS-GLI-{case_name}-t1n-inference.nii.gz"


def export_native_outputs(input_root: Path, output_dir: Path, overwrite: bool = False) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: List[Path] = []

    for src in find_native_outputs(input_root):
        case_name = src.parent.name
        nii_name = build_output_name(case_name)
        dest_path = output_dir / nii_name

        if dest_path.exists() and not overwrite:
            print(f"Skipping existing file: {dest_path}")
            continue

        img = nib.load(str(src))
        data = np.asarray(img.dataobj)
        if data.ndim == 4 and data.shape[0] == 1:
            data = data[0]
        if data.ndim == 5 and data.shape[0] == 1:
            data = data[0]

        out_img = nib.Nifti1Image(
            data.astype(np.float32),
            affine=img.affine,
            header=img.header.copy(),
        )

        nib.save(out_img, str(dest_path))
        exported.append(dest_path)
        print(f"Exported: {src} -> {dest_path}")

    return exported


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Export native inpainting outputs into a submission-style folder")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=repo_root / "output_mni_inpaint",
        help="Directory that contains case subfolders with inpainted_native*.nii* files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "ASNR-MICCAI-BraTS2023-Local-Synthesis-Challenge-Validation-Results",
        help="Directory where per-case NIfTI files will be written",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing destination files")
    args = parser.parse_args()

    exported = export_native_outputs(args.input_root, args.output_dir, overwrite=args.overwrite)
    print(f"Finished exporting {len(exported)} file(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
