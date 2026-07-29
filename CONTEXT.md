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

On the production F: drive, the generated cleanup output named
`plant-cleaned-garden-ec2fbd1-final-v2.ply` is a read-only input. A reviewed
normalization is created beside it as
`plant-cleaned-garden-ec2fbd1-final-v2-orientation-corrected-v1.ply` under
`F:\3d_scans\scans\2026\202607_sf`. The writer refuses to overwrite either
file. It does not modify source photos, videos, Metashape projects, cleanup
outputs, or the existing scan directory structure.

The separately gated Blender review may create one non-overwriting inspection
scene beside an eligible cleanup output:
`plant-cleaned-garden-ec2fbd1-final-v2-orientation-review-v1.blend`. It embeds
review geometry but never changes the input PLY or any Metashape data.
