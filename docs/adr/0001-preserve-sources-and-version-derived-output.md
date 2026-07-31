# ADR 0001: Preserve sources and version derived output

- Status: accepted
- Date: 2026-07-31

## Context

The pipeline works with irreplaceable captures, Metashape projects, and point
clouds. Cleanup and normalization are probabilistic transformations, so an
apparently successful run cannot justify changing its input.

## Decision

All source media, projects, and source clouds are immutable. Derived writers
bind output to a source checksum and stable identity, create a new path, and
fail if that path exists. Production-corrected PLYs use an explicit versioned
filename beside an unchanged generated-cleanup PLY.

The replacement primitive retained in the recovery script is not a supported
publication path. It exists only to restore content changed by the superseded
replacement workflow.

## Consequences

- Reprocessing selects a new version rather than overwriting.
- Tests use temporary source fixtures and assert unrelated files stay
  byte-for-byte unchanged.
- Disk usage is higher, but provenance and rollback remain inspectable.
- Existing outputs create a hard stop instead of an implicit update.
