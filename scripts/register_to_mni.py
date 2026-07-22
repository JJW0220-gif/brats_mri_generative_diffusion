from __future__ import annotations

import argparse
from pathlib import Path


def _require_sitk():
    try:
        import SimpleITK as sitk
    except ImportError as error:
        raise ImportError(
            "SimpleITK is required for MNI registration. Install it with `pip install SimpleITK`."
        ) from error
    return sitk


def register_to_mni(
    moving_path: str,
    template_path: str,
    output_transform_path: str,
    output_warped_path: str | None = None,
    sampling_percentage: float = 0.2,
    learning_rate: float = 1.0,
    iterations: int = 150,
) -> str:
    sitk = _require_sitk()

    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    fixed = sitk.ReadImage(str(template_path), sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(sampling_percentage)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=1e-4,
        numberOfIterations=iterations,
        relaxationFactor=0.5,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed, moving)

    output_transform = Path(output_transform_path)
    output_transform.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteTransform(final_transform, str(output_transform))

    if output_warped_path is not None:
        resampled = sitk.Resample(
            moving,
            fixed,
            final_transform,
            sitk.sitkLinear,
            0.0,
            moving.GetPixelID(),
        )
        warped_path = Path(output_warped_path)
        warped_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(resampled, str(warped_path))

    return str(output_transform)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register one MRI volume into MNI space.")
    parser.add_argument("--moving", required=True, help="Path to the native-space MRI.")
    parser.add_argument("--template", required=True, help="Path to the MNI template image.")
    parser.add_argument("--output_transform", required=True, help="Path to save the transform (.tfm).")
    parser.add_argument("--output_warped", help="Optional path to save the MRI warped into MNI space.")
    parser.add_argument("--sampling_percentage", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=150)
    args = parser.parse_args()

    register_to_mni(
        moving_path=args.moving,
        template_path=args.template,
        output_transform_path=args.output_transform,
        output_warped_path=args.output_warped,
        sampling_percentage=args.sampling_percentage,
        learning_rate=args.learning_rate,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()