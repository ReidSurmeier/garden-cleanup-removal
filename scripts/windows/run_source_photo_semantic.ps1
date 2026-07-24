param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-f]{7,40}$")]
    [string]$Commit,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^F:\\3d_scans\\cleanup\\garden-cleanup-removal-[^\\]+\\source-photo-semantic-[^\\]+$")]
    [string]$OutputRoot,
    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cuda",
    [ValidateRange(1, 64)]
    [int]$TorchThreads = 4
)

$ErrorActionPreference = "Stop"
$commitShort = $Commit.Substring(0, 7)
$repo = "F:\3d_scans\code\garden-cleanup-removal-$commitShort"
$python = "F:\3d_scans\code\garden-cleanup-removal-c01ecec\.venv\Scripts\python.exe"
$env:PYTHONPATH = Join-Path $repo "src"
$env:HF_HUB_DISABLE_XET = "1"

& $python -u `
    (Join-Path $repo "scripts\source_photo_semantic_batch.py") `
    "F:\3d_scans\cleanup\garden-cleanup-removal-50b7f86\inventory\projects.json" `
    "F:\3d_scans\cleanup\garden-cleanup-removal-631e1a8\full-stride8-v2\canonical-stride8" `
    "F:\3d_scans\cleanup\garden-cleanup-removal-de9c99e\native-full-v1" `
    "F:\3d_scans\cleanup\garden-cleanup-removal-d2be9a7\camera-inventory-v1" `
    $OutputRoot `
    --stride 8 `
    --camera-count 12 `
    --maximum-dimension 768 `
    --device $Device `
    --torch-threads $TorchThreads
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
