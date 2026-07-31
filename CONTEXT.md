# Garden Railing Removal context

## Domain

The system turns immutable scan evidence into auditable cleanup and
orientation candidates. It does not decide that a physical scan is correct
merely because a model, renderer, or server produced an output.

## Core language

- **Source project**: a Metashape project opened only for inventory or export.
- **Source cloud**: an immutable PLY or exported point set.
- **Source identity**: the checksum, point count, schema, and stable source
  indices that bind later artifacts to one source cloud.
- **Plant candidate**: points eligible for plant retention before railing and
  support evidence is reconciled.
- **Semantic evidence**: per-view model votes. Evidence is an input to a
  decision, not a deletion command.
- **Railing seed**: a low-green, low-saturation point where railing evidence
  exceeds paired plant evidence.
- **Rail line**: a long, narrow rigid structure supported by coherent railing
  seeds.
- **Completion**: inferred railing membership along an accepted rail line.
- **Cleanup run**: a new derived directory containing masks, reports, and
  optional review artifacts for one source identity.
- **Cleaned plant cloud**: a derived PLY; it is never a source capture.
- **Needs review**: the fail-closed state for ambiguous or competing evidence.
- **Orientation candidate**: a proposed right-handed transform supported by
  camera, ground, photo, or structural evidence.
- **Orientation selection**: an automatic gated result or an explicit visual
  choice, including the identity transform.
- **Orientation review**: rendered evidence used to inspect candidates. It is
  not production authorization.
- **Corrected output**: a new, versioned PLY created beside an unchanged
  cleaned plant cloud after all production gates pass.
- **Publication plan**: an exhaustive clean-or-flag decision for a manifest;
  it does not itself write production artifacts.

## Invariants

1. Source media and source clouds are never edited, renamed, moved, or deleted.
2. Every derived artifact is bound to an explicit source identity.
3. Writers create new paths and refuse to overwrite existing outputs.
4. Strong plant evidence protects intersecting plant points.
5. Ambiguous evidence becomes `needs_review`; uncertainty is not silently
   converted into removal or rotation.
6. Cleanup, orientation selection, review transport, and production publication
   are separate authority boundaries.
7. Orientation preserves point count, colors, classifications, source indices,
   finite values, scale, and handedness.
8. A production batch starts only after the SD-card canary and every gate in
   `docs/normalization-production-safety.md` pass.

## State transitions

```text
source evidence
  -> inventoried
  -> cleanup candidate
  -> cleaned or needs_review
  -> orientation candidate
  -> selected or needs_review
  -> corrected output (production gates only)
```

Review rendering and transport may occur after any derived stage, but they do
not advance the artifact to the next state.

## Repository relationship

The GitHub repository name `garden-cleanup-removal` predates the canonical
project name **Garden Railing Removal**. The separate Garden Scan Cleanup
repository has a different Git root and broader portfolio scope. Shared subject
matter does not imply shared artifact custody; cross-repository inputs require
an explicit manifest and checksum validation.

## Production boundary

The exact Windows production root, filenames, canary cohort, and batch
verification rules are intentionally centralized in
[`docs/normalization-production-safety.md`](docs/normalization-production-safety.md).
Local paths, Tailscale routes, public hostnames, and an HTTP response are
operational evidence only. None can relax that contract.
