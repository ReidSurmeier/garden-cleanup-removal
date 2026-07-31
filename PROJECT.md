# Project

## Identity

- Canonical name: Garden Railing Removal
- GitHub repository: `ReidSurmeier/garden-cleanup-removal`
- Python package: `garden-railing-removal`
- Status: active experimental pipeline; production normalization gated
- Deployment: external static review; application hosting is not owned here

The GitHub name is historical. Use **Garden Railing Removal** in new
documentation and issue titles.

## Objective

Produce conservative, auditable plant-cleanup outputs and orientation
candidates while preserving every source artifact and making uncertainty
visible.

## In scope

- read-only project and cloud inventory;
- source-bound adaptive configuration;
- geometric and multi-view semantic evidence;
- railing, floor, and scene cleanup decisions;
- derived batch reports and review artifacts;
- fail-closed orientation evidence and visual selection;
- versioned, non-overwriting corrected-output publication.

## Out of scope

- modifying captures or Metashape projects;
- silently deleting uncertain points;
- treating model output as physical ground truth;
- hosting a public application;
- authorizing a production batch from a review URL;
- merging custody with Garden Scan Cleanup without an explicit decision.

## Current state

The codebase has temporary-fixture coverage across inventory, cleanup, review,
normalization, and publication. A clean `dev` install passed 142 tests on
2026-07-31. Production evidence remains incomplete: the documented five-scan
canary, visual acceptance of the two known failures, and checkpoint-batch
inventory comparison are still required.

The repository can build static review artifacts. Their transport may be local,
tailnet-only, or separately published by an operator, but this repository does
not own a GitHub Pages site. On 2026-07-31 the separately operated 237-scan
review at `normals.reidsurmeier.wtf` returned HTTP 502 because its Windows
source volume was not attached. Cloudflare, Droplet nginx, Tailscale Serve, and
the WSL bridge remained present. Runtime recovery is tracked in GitHub issue
2; it must not be accomplished by mutating source scans.

## Next gates

1. Complete and record the five-scan SD-card canary (issue 1).
2. Restore and durably own the external review runtime (issue 2).
3. Confirm custody and backup policy for large ignored run data (issue 3).
4. Validate a representative semantic-cleanup pilot (issue 4).
5. Resolve the two known orientation failures through visual evidence.
6. Reconcile any historical replacement artifacts through the recovery-only
   path.
7. Run a small, inventoried production checkpoint only after the prior gates
   pass.

## Verification

```bash
uv sync --extra dev --locked
uv run python -m pytest -q
uv run python -m compileall -q src tests scripts
```

See `CONTRIBUTING.md` for change workflow and
`docs/normalization-production-safety.md` for production-specific verification.
