param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-f]{7,40}$")]
    [string]$Commit,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]*$")]
    [string]$RunTag,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[A-Za-z0-9][A-Za-z0-9._-]*\.json$")]
    [string]$AssignmentConfig,
    [string]$SemanticRoot = (
        "F:\3d_scans\cleanup\garden-cleanup-removal-f1aa8ba\" +
        "source-photo-semantic-photo12-stride8-v4"
    )
)

$ErrorActionPreference = "Stop"
$commit = $Commit.Substring(0, 7)
$repo = "F:\3d_scans\code\garden-cleanup-removal-$commit"
$python = (
    "F:\3d_scans\code\garden-cleanup-removal-c01ecec\" +
    ".venv\Scripts\python.exe"
)
$inventory = (
    "F:\3d_scans\cleanup\garden-cleanup-removal-50b7f86\" +
    "inventory\projects.json"
)
$baselineRoot = (
    "F:\3d_scans\cleanup\garden-cleanup-removal-631e1a8\" +
    "full-stride8-v2"
)
$baselineManifest = Join-Path `
    $baselineRoot "adaptive-profiles\cleanup-manifest.json"
$baselineOutput = Join-Path $baselineRoot "cleanup"
$assignment = Join-Path $repo "configs\$AssignmentConfig"
$catalog = Join-Path $repo "configs\scene-plan-catalog-202607-sf.json"
$versionRoot = (
    "F:\3d_scans\cleanup\garden-cleanup-removal-$commit-$RunTag"
)
$manifest = Join-Path $versionRoot "targeted-corrections.json"
$correctionRoot = Join-Path $versionRoot "object-corrections-v1"
$cleanupRoot = Join-Path $versionRoot "source-photo-cleanup-v1"
$reviewRoot = Join-Path $cleanupRoot "review-pages-v1"
$log = Join-Path $versionRoot "run.log"

foreach ($required in @(
    $repo,
    $python,
    $inventory,
    $baselineManifest,
    $baselineOutput,
    $assignment,
    $catalog,
    $SemanticRoot
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required input does not exist: $required"
    }
}
if (Test-Path -LiteralPath $versionRoot) {
    throw "Immutable targeted output root already exists: $versionRoot"
}
New-Item -ItemType Directory -Path $versionRoot | Out-Null
$env:PYTHONPATH = Join-Path $repo "src"
$env:HF_HUB_DISABLE_XET = "1"

function Write-ProgressLog([string]$message) {
    $line = "$(Get-Date -Format o) $message"
    Add-Content -LiteralPath $log -Value $line
    Write-Output $line
}

Write-ProgressLog "building targeted correction manifest"
& $python `
    (Join-Path $repo "scripts\build_targeted_correction_manifest.py") `
    $baselineManifest `
    $catalog `
    $assignment `
    $manifest
if ($LASTEXITCODE -ne 0) {
    throw "Targeted manifest failed with exit code $LASTEXITCODE"
}

Write-ProgressLog "running object-aware corrections"
& $python -u -m railing_removal.batch_cli $manifest $correctionRoot
if ($LASTEXITCODE -ne 0) {
    throw "Object correction batch failed with exit code $LASTEXITCODE"
}

Write-ProgressLog "carrying final decisions into source-photo cleanup"
& $python -u `
    (Join-Path $repo "scripts\source_photo_cleanup_batch.py") `
    $manifest `
    $baselineOutput `
    $SemanticRoot `
    $cleanupRoot `
    --baseline-correction-root $correctionRoot
if ($LASTEXITCODE -ne 0) {
    throw "Source-photo cleanup failed with exit code $LASTEXITCODE"
}

Write-ProgressLog "building before-after and full 3D review"
& $python `
    (Join-Path $repo "scripts\build_paginated_review.py") `
    (Join-Path $cleanupRoot "batch-report.json") `
    $reviewRoot `
    --page-size 20
if ($LASTEXITCODE -ne 0) {
    throw "Review build failed with exit code $LASTEXITCODE"
}

$corrections = Get-Content `
    -LiteralPath (Join-Path $correctionRoot "batch-report.json") `
    -Raw |
    ConvertFrom-Json
$cleanup = Get-Content `
    -LiteralPath (Join-Path $cleanupRoot "batch-report.json") `
    -Raw |
    ConvertFrom-Json
$completion = [ordered]@{
    schema_version = 1
    code_commit = $commit
    run_tag = $RunTag
    assignment_config = $AssignmentConfig
    finished_at = (Get-Date -Format o)
    source_files_deleted = 0
    source_directories_modified = 0
    correction_summary = $corrections.summary
    cleanup_summary = $cleanup.summary
    review = (Join-Path $reviewRoot "index.html")
}
$completion |
    ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath (Join-Path $versionRoot "complete.json") `
        -Encoding UTF8
Write-ProgressLog "targeted correction run complete; awaiting visual QA"
