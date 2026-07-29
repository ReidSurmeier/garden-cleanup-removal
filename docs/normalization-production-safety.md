# Normalization production safety contract

## Objective

Correct the orientation of the plant clouds already produced by the cleanup
pipeline. Prove the method on the SD-card canary before changing any production
artifact on the F: drive.

This contract separates two phases. Phase 2 cannot start until every Phase 1
gate passes.

## Phase 1: SD-card verification

The SD card and its source scan material are read-only inputs. Development
outputs belong in the repository's ignored `runs/` directory and in the
tailnet-only review site.

The canary cohort contains five scans. It includes the two known orientation
failures:

- `2026-07-15 17.25.21`
- `2026-07-15 17.26.58`

The normalization method must not treat every point with decision code `2` as
a verified floor. That code means rejected support and may contain railings,
curved planters, distant fragments, and other non-floor geometry.

Phase 1 passes only when:

1. A regression test reproduces contaminated support evidence tilting an
   otherwise upright scan.
2. The correction chooses orientation from independent, coherent evidence and
   fails closed as `needs_review` when the evidence is ambiguous.
3. All repository tests pass.
4. All five canaries are rendered from front, side, and top.
5. The rejected/support evidence is rendered separately so a false floor fit
   is visible.
6. The two known failures no longer appear tilted in the visual review.
7. Point count, color, classification, source index, and handedness are
   unchanged by the transform.
8. The Tailscale review returns HTTP 200 and exposes the full-resolution
   evidence.

No production F: file may be opened for writing during Phase 1.

## Phase 2: F-drive production boundary

Production root:

`F:\3d_scans\scans\2026\202607_sf`

The only allowed writable path shape is:

`F:\3d_scans\scans\2026\202607_sf\<scan-directory>\plant-cleaned-garden-ec2fbd1-final-v2.ply`

The July 29 read-only preflight found 251 scan directories and 237 existing
files with that exact generated-cleanup filename. Missing generated outputs are
reported; the normalizer does not create substitutes.

The updater must refuse to run unless all of these checks pass:

- The resolved production root exactly matches the configured root.
- The target is an existing regular file.
- The target's parent is a direct child of the production root.
- The target filename exactly matches the generated-cleanup filename.
- Neither the target nor its parent is a symlink or Windows reparse point.
- The PLY schema matches the cleanup output contract.
- The file does not already contain this normalization version marker.
- The supplied SD validation report says `passed` and matches the running
  normalization version.

The updater may create one uniquely named temporary file beside the target
while it works. This is not a permanent normalized copy. Before replacement it
must validate:

- equal point counts;
- identical colors, classifications, and source indices;
- finite transformed coordinates and normals;
- a uniform, right-handed transform;
- a readable complete PLY;
- no mutation outside the temporary path.

After validation, the updater atomically replaces the same generated-cleanup
filename and removes any temporary residue. It does not retain a backup PLY or
create a second normalized PLY. The transform and original checksum are stored
as comments inside the replacement PLY so repeat application is rejected.

## Files that must never change

The production updater must not write, rename, move, or delete:

- source photos or extracted frames;
- source videos;
- `.psx`, `.psz`, or Metashape `.files` project data;
- original or intermediate point clouds;
- scan directories;
- any PLY whose name is not
  `plant-cleaned-garden-ec2fbd1-final-v2.ply`;
- any file outside the production root.

Review screenshots and reports remain outside the production scan tree. They
are served from the local Tailscale review directory.

## Verification around every production batch

Before a batch, capture a read-only inventory of every path, size, modified
time, and attributes beneath the production root. After the batch:

1. Compare the inventories.
2. Require every changed path to match the exact writable path shape.
3. Require the changed-path count to equal the successful target count.
4. Require no added or removed paths.
5. Re-read every replaced PLY and verify its embedded normalization marker.
6. Render the batch for visual review before continuing.

Start with a small checkpoint batch. Continue only when its invariant checks
and visual review pass. A failed or ambiguous scan is left unchanged and
reported for review.
