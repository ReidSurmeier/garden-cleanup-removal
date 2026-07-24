param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[0-9a-f]{7,40}$")]
    [string]$Commit,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^F:\\3d_scans\\cleanup\\garden-cleanup-removal-[^\\]+\\source-photo-semantic-[^\\]+$")]
    [string]$SemanticRoot
)

$ErrorActionPreference = "Stop"
$commit = $Commit.Substring(0, 7)
$repo = "F:\3d_scans\code\garden-cleanup-removal-$commit"
$python = "F:\3d_scans\code\garden-cleanup-removal-c01ecec\.venv\Scripts\python.exe"
$inventory = "F:\3d_scans\cleanup\garden-cleanup-removal-50b7f86\inventory\projects.json"
$baselineRoot = "F:\3d_scans\cleanup\garden-cleanup-removal-631e1a8\full-stride8-v2"
$baselineManifest = Join-Path $baselineRoot "adaptive-profiles\cleanup-manifest.json"
$baselineOutput = Join-Path $baselineRoot "cleanup"
$baselineReport = Join-Path $baselineOutput "batch-report.json"
$semanticReport = Join-Path $semanticRoot "batch-report.json"
$versionRoot = "F:\3d_scans\cleanup\garden-cleanup-removal-$commit"
$correctionRoot = Join-Path $versionRoot "baseline-corrections-v2"
$correctionReport = Join-Path $correctionRoot "batch-report.json"
$cleanupRoot = Join-Path $versionRoot "source-photo-cleanup-v2"
$cleanupReport = Join-Path $cleanupRoot "batch-report.json"
$orchestration = "F:\3d_scans\cleanup\garden-cleanup-orchestration-$commit-v2"
$correctionManifest = Join-Path $orchestration "baseline-corrections-v2-manifest.json"
$log = Join-Path $orchestration "orchestration.log"

if (Test-Path -LiteralPath $orchestration) {
    throw "Immutable orchestration root already exists: $orchestration"
}
New-Item -ItemType Directory -Path $orchestration | Out-Null
$env:PYTHONPATH = Join-Path $repo "src"
$env:HF_HUB_DISABLE_XET = "1"

function Write-ProgressLog([string]$message) {
    $line = "$(Get-Date -Format o) $message"
    Add-Content -LiteralPath $log -Value $line
    Write-Output $line
}

Write-ProgressLog "waiting for frozen baseline and calibrated semantic reports"
while (
    -not (Test-Path -LiteralPath $baselineReport) -or
    -not (Test-Path -LiteralPath $semanticReport)
) {
    Start-Sleep -Seconds 30
}
Write-ProgressLog "dependency reports detected"

if (-not (Test-Path -LiteralPath $correctionManifest)) {
    Write-ProgressLog "selecting every partial or failed baseline scan"
    & $python `
        (Join-Path $repo "scripts\build_correction_manifest.py") `
        $baselineManifest `
        $baselineReport `
        $correctionManifest
    if ($LASTEXITCODE -ne 0) {
        throw "Correction manifest failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath $correctionReport)) {
    Write-ProgressLog "running additive calibrated-photo baseline corrections"
    & $python -u -m railing_removal.batch_cli `
        $correctionManifest `
        $correctionRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Baseline correction batch failed with exit code $LASTEXITCODE"
    }
}
Write-ProgressLog "baseline corrections ready"

if (-not (Test-Path -LiteralPath $cleanupReport)) {
    Write-ProgressLog "running calibrated source-photo cleanup for all scans"
    & $python -u `
        (Join-Path $repo "scripts\source_photo_cleanup_batch.py") `
        $inventory `
        $baselineOutput `
        $semanticRoot `
        $cleanupRoot `
        --baseline-correction-root $correctionRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Source-photo cleanup batch failed with exit code $LASTEXITCODE"
    }
}
Write-ProgressLog "source-photo cleanup batch ready"

$quality = Join-Path $cleanupRoot "quality-report-v2.json"
if (-not (Test-Path -LiteralPath $quality)) {
    Write-ProgressLog "building quantitative and calibrated-photo QA report"
    & $python `
        (Join-Path $repo "scripts\build_quality_report.py") `
        $cleanupReport `
        $quality
    if ($LASTEXITCODE -ne 0) {
        throw "Quality report failed with exit code $LASTEXITCODE"
    }
}

$review = Join-Path $cleanupRoot "review-pages-v2"
if (-not (Test-Path -LiteralPath $review)) {
    Write-ProgressLog "building paginated before-after and full 3D review"
    & $python `
        (Join-Path $repo "scripts\build_paginated_review.py") `
        $cleanupReport `
        $review `
        --page-size 20
    if ($LASTEXITCODE -ne 0) {
        throw "Paginated review failed with exit code $LASTEXITCODE"
    }
}

$baseline = Get-Content -LiteralPath $baselineReport -Raw |
    ConvertFrom-Json
$semantic = Get-Content -LiteralPath $semanticReport -Raw |
    ConvertFrom-Json
$correction = Get-Content -LiteralPath $correctionReport -Raw |
    ConvertFrom-Json
$cleanup = Get-Content -LiteralPath $cleanupReport -Raw |
    ConvertFrom-Json
$completion = [ordered]@{
    schema_version = 1
    code_commit = $commit
    finished_at = (Get-Date -Format o)
    source_files_deleted = 0
    source_directories_modified = 0
    baseline_summary = $baseline.summary
    source_photo_semantic_summary = $semantic.summary
    baseline_correction_summary = $correction.summary
    source_photo_cleanup_summary = $cleanup.summary
    quality_report = $quality
    paginated_review = (Join-Path $review "index.html")
}
$completion |
    ConvertTo-Json -Depth 6 |
    Set-Content -LiteralPath (Join-Path $orchestration "complete.json") `
        -Encoding UTF8
Write-ProgressLog "orchestration complete; awaiting visual QA and publication"
