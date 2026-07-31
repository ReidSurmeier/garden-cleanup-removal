# ADR 0002: Separate cleanup, orientation, and publication

- Status: accepted
- Date: 2026-07-31

## Context

Railing evidence answers which points belong in a derived plant cloud.
Orientation evidence answers how that derived cloud should be transformed.
Publication decides whether a reviewed artifact may enter a production scan
directory. Combining these decisions lets confidence in one stage leak into
another.

## Decision

Cleanup, orientation selection, and publication are separate stages with
separate reports and gates:

1. Cleanup emits a derived cloud or `needs_review`.
2. Orientation emits candidates and a selected transform or `needs_review`.
3. Publication validates the source identity, review evidence, exact
   destination shape, and production safety report before creating a new file.

The identity transform is a valid visual selection. A cleanup acceptance never
implies orientation acceptance, and an orientation selection never implies
production authorization.

## Consequences

- Reports must preserve stage-specific status and provenance.
- Batch planners fail closed when inventories or reports are incomplete.
- Operators can review geometry without granting write authority.
- Additional steps are deliberate safety boundaries, not redundant workflow.
