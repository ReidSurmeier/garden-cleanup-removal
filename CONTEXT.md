# Railing removal

The input is a plant-candidate point cloud plus multi-view semantic evidence.
A confirmed seed is a point where railing evidence exceeds paired plant
evidence. A rail line is a long, narrow rigid structure supported by confirmed
low-green, low-saturation seeds. Completion fills unobserved points along an
accepted rail line. Strong plant evidence protects intersecting plant points.

The output is a rejection mask and an audit report. It never edits the source
cloud.

## Orientation normalization

Orientation normalization is a separate, gated stage after cleanup. Its
production input is the generated cleaned plant cloud, never a source capture.
The normalizer must first pass the SD-card canary and TDD gate documented in
[`docs/normalization-production-safety.md`](docs/normalization-production-safety.md).

On the production F: drive, the only mutable artifact is an existing generated
cleanup output named `plant-cleaned-garden-ec2fbd1-final-v2.ply` directly
inside a scan directory under `F:\3d_scans\scans\2026\202607_sf`. The
normalizer replaces that file atomically in place. It does not create a second
normalized PLY, modify source photos, videos, Metashape projects, or change the
scan directory structure.
