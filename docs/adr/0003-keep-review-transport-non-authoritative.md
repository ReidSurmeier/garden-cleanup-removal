# ADR 0003: Keep review transport non-authoritative

- Status: accepted
- Date: 2026-07-31

## Context

Static WebGL pages, Blender scenes, contact sheets, and tailnet or public HTTP
routes make evidence easier to inspect. Availability and visual correctness are
different facts, and neither grants permission to mutate production data.

## Decision

Review generation produces static, rebuildable artifacts. Transport is an
operational concern outside the domain state machine. An HTTP 200 proves only
reachability; reviewers must record an explicit selection against the matching
source identity. Production publication still requires the independent safety
contract.

This repository does not own a persistent public web deployment. Any separately
hosted review must record its runtime owner and may be removed without changing
canonical scan evidence.

## Consequences

- Tests build review artifacts in temporary directories without binding ports.
- Hostnames and local machine paths do not enter domain records.
- A broken review route blocks convenient inspection, not source recovery.
- A working review route does not advance a scan to production-ready.
