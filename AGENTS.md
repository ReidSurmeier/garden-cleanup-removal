# Garden Railing Removal agent guide

This repository owns a conservative, non-overwriting point-cloud cleanup and
orientation-review pipeline. Read `PROJECT.md`, `CONTEXT.md`, and the relevant
ADRs before changing behavior. Read
`docs/normalization-production-safety.md` before touching any production-bound
path logic.

## Boundaries

- Treat captures, photos, videos, Metashape projects, and input PLYs as
  immutable.
- Use temporary fixtures in tests. Never target mounted scan media or a
  production root.
- Preserve uncertain evidence for review and fail closed when an output exists.
- A review transport is not production authority.
- Do not run production batch scripts, publish review artifacts, or move local
  datasets as a side effect of tests or documentation work.
- Never commit credentials, model tokens, scan datasets, generated runs, or
  private machine paths.

## Commands

```bash
uv sync --extra dev --locked
uv run python -m pytest -q
uv run python -m compileall -q src tests scripts
```

Run a focused failing test before implementation, then the complete suite.

## Agent skills

### Issue tracker

Issues live in this repository's GitHub Issues. See
`docs/agents/issue-tracker.md`.

### Triage labels

The tracker uses the five default Matt Pocock skill labels. See
`docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository. See `docs/agents/domain.md`, `CONTEXT.md`,
and `docs/adr/`.
