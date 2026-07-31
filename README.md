# Garden Railing Removal

## Purpose

Garden Railing Removal is a conservative point-cloud pipeline for separating
plant geometry from line-family railings and other scene support. It combines
geometric evidence with multi-view semantic evidence, produces derived PLYs and
audit reports, and can prepare orientation candidates for a separate visual
review.

The GitHub repository retains the historical name
`garden-cleanup-removal`. This project is related to the broader
**Garden Scan Cleanup** work, but it has its own history, package, tests, and
production gates. Do not treat artifacts from one repository as inputs to the
other without an explicit manifest and source-identity check.

## Safety boundary

- Source captures, photos, videos, Metashape projects, and source point clouds
  are immutable inputs.
- Cleanup and review outputs are derived, versioned artifacts. Writers fail
  closed when a destination already exists.
- Ambiguous semantic or orientation evidence is retained for review; it is not
  permission to remove geometry or rotate a cloud.
- Cleanup, orientation selection, and publication are distinct decisions.
- A rendered review or an HTTP 200 proves only that review evidence is
  reachable. It does not authorize production writes.
- Production normalization is prohibited until every gate in
  [`docs/normalization-production-safety.md`](docs/normalization-production-safety.md)
  passes.

The legacy replacement helper in
`scripts/restore_replaced_orientation_batch.py` exists only to restore files
changed by an earlier replacement workflow. New orientation output must use
the non-overwriting corrected filename.

## Workflow

1. Inventory an explicitly configured source root and bind an adaptive profile
   to the source checksum.
2. Build plant, railing, floor, and scene evidence without changing the input.
3. Run cleanup into a new output directory and keep its decision report.
4. Render derived evidence for human review. Unresolved scans stay unresolved.
5. Build orientation candidates from independent camera, ground, photo, and
   structural evidence.
6. Select an orientation only after the canary and visual-review gates pass.
7. Publish a new versioned corrected output through the production safety
   contract; never replace the cleanup input.

The installed command-line entry points are:

```text
garden-railing-remove
garden-full-cleanup
garden-metashape-export
garden-normalize-cleanup
garden-adaptive-config
garden-batch-cleanup
garden-batch-review
```

Run any entry point with `--help` before preparing a real manifest. Examples,
tests, and local experiments must use disposable inputs; production roots are
not development fixtures.

## Development

Python 3.11 or newer and
[`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync --extra dev --locked
uv run python -m pytest -q
uv run python -m compileall -q src tests scripts
```

The full suite is the minimum gate for changes. Behavior changes follow a
red-green-refactor loop: first add a focused failing test, make the smallest
implementation change, then run the complete suite. Optional runtime groups
are available for reference-video, orientation, and vision workflows.

## Repository layout

```text
configs/                  Source-bound pipeline profiles
docs/                     Safety contract, ADRs, and agent conventions
scripts/                  Explicit batch and review entry points
src/plant_cleanup/        General plant-cleanup evidence and classification
src/railing_removal/      Railing, review, orientation, and publication domain
tests/                    Temporary-fixture unit and integration tests
```

Large datasets, model caches, generated runs, and review renders do not belong
in Git. Local `runs/` content is rebuildable evidence, not canonical source.

## Current evidence

As of 2026-07-31, a fresh `dev`-only environment collects and passes 142 tests.
The repository contains code for cleanup, review, and non-overwriting
orientation publication. That test result does **not** establish that the
five-scan SD-card canary has passed, that every scan is physically upright, or
that production normalization is authorized. Those remain explicit review
gates.

## Models

- `CIDAS/clipseg-rd64-refined`
- `facebook/sam2-hiera-tiny`
- `shi-labs/oneformer_ade20k_swin_large`

These model runtimes are optional. Installing a model dependency does not
download or grant access to model weights.

## Stack

The implementation uses Python, NumPy, SciPy, plyfile, Pillow, OpenCV, pytest,
and uv. Optional semantic workflows use Hugging Face Transformers and PyTorch.
Agisoft Metashape Professional 2.3.1 is an external source/export tool, and
WebGL/Blender outputs are review surfaces rather than sources of truth.
