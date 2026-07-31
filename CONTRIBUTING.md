# Contributing

## Before changing code

Read `PROJECT.md`, `CONTEXT.md`, the applicable ADRs, and
`docs/normalization-production-safety.md` for production-bound behavior.
Confirm that examples and tests use disposable inputs.

## Test-driven workflow

1. Add the smallest test that demonstrates the missing behavior or regression.
2. Run it and record the expected failure.
3. Implement the smallest safe change.
4. Run the focused test until green.
5. Refactor without changing behavior.
6. Run the complete suite and compile check.

```bash
uv sync --extra dev --locked
uv run python -m pytest tests/path_to_test.py -q
uv run python -m pytest -q
uv run python -m compileall -q src tests scripts
```

Tests must not use mounted scan media, production roots, real credentials,
network publication, or existing generated outputs. Prefer `tmp_path` and
small synthetic point sets.

## Documentation and decisions

Use the terms in `CONTEXT.md`. Update that context when the domain model
changes. Add an ADR under `docs/adr/` when a change alters artifact custody,
source immutability, stage authority, output naming, or production safety.

## Pull requests

Describe the red test, the green implementation, complete-suite results, and
any unverified hardware, model, visual, or production boundary. Never describe
a rendered or reachable review as proof that a scan is physically correct.

No license is granted for reuse; see `LICENSE`.
