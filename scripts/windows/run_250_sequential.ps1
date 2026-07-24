param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-f]{7,40}$")]
    [string]$Commit,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    [string]$RunTag
)

$ErrorActionPreference = "Stop"
$commitShort = $Commit.Substring(0, 7)
$repo = "F:\3d_scans\code\garden-cleanup-removal-$commitShort"
$baselineReport = (
    "F:\3d_scans\cleanup\garden-cleanup-removal-631e1a8\" +
    "full-stride8-v2\cleanup\batch-report.json"
)
$semanticRoot = (
    "F:\3d_scans\cleanup\garden-cleanup-removal-$commitShort\" +
    "source-photo-semantic-$RunTag"
)
$semanticReport = Join-Path $semanticRoot "batch-report.json"
$sequentialRoot = (
    "F:\3d_scans\cleanup\garden-cleanup-sequential-$commitShort-$RunTag"
)
if (Test-Path -LiteralPath $sequentialRoot) {
    throw "Immutable sequential root already exists: $sequentialRoot"
}
New-Item -ItemType Directory -Path $sequentialRoot | Out-Null
$log = Join-Path $sequentialRoot "sequential.log"

function Write-ProgressLog([string]$message) {
    $line = "$(Get-Date -Format o) $message"
    Add-Content -LiteralPath $log -Value $line
    Write-Output $line
}

Write-ProgressLog "waiting for exclusive baseline GPU stage"
while (-not (Test-Path -LiteralPath $baselineReport)) {
    Start-Sleep -Seconds 30
}
Write-ProgressLog "baseline ready; starting calibrated CUDA semantics"

if (-not (Test-Path -LiteralPath $semanticReport)) {
    & powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File (Join-Path $repo "scripts\windows\run_source_photo_semantic.ps1") `
        -Commit $commitShort `
        -OutputRoot $semanticRoot `
        -Device cuda `
        -TorchThreads 4
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA semantic batch failed with exit code $LASTEXITCODE"
    }
}
Write-ProgressLog "calibrated CUDA semantics ready; starting cleanup"

& powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File (Join-Path $repo "scripts\windows\orchestrate_250.ps1") `
    -Commit $commitShort `
    -SemanticRoot $semanticRoot `
    -RunTag $RunTag
if ($LASTEXITCODE -ne 0) {
    throw "Cleanup orchestration failed with exit code $LASTEXITCODE"
}
Write-ProgressLog "sequential production processing complete"
