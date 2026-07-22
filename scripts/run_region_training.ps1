$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$config = "configs/train_diffusion_inpaint_mni.json"
Write-Host "==> Training unified MNI inpainting model with $config"
py -m monai.bundle run --config_file $config
if ($LASTEXITCODE -ne 0) {
    Write-Error "Training failed"
    exit $LASTEXITCODE
}
