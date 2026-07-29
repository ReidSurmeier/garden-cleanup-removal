from __future__ import annotations

from typing import Any

import numpy as np


def _up(value: object) -> list[float]:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("selected orientation must be a finite 3D vector")
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        raise ValueError("selected orientation cannot be zero")
    return (vector / length).tolist()


def select_orientation(
    fused: dict[str, Any],
    *,
    visual_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    consensus = fused.get("consensus")
    if not isinstance(consensus, dict):
        raise ValueError("fused orientation lacks consensus")
    status = consensus.get("status")
    if (
        isinstance(visual_selection, dict)
        and visual_selection.get("candidate") == "identity"
    ):
        reviewer = str(visual_selection.get("reviewer", "")).strip()
        reason = str(visual_selection.get("reason", "")).strip()
        if not reviewer or not reason:
            raise ValueError("visual selection requires reviewer and reason")
        return {
            "schema_version": 1,
            "scan_id": str(fused["scan_id"]),
            "status": "selected",
            "up": [0.0, 0.0, 1.0],
            "selection_basis": "visual_identity_review",
            "visual_selection": {
                **visual_selection,
                "candidate": "identity",
                "reviewer": reviewer,
                "reason": reason,
            },
        }
    if status == "automatic":
        return {
            "schema_version": 1,
            "scan_id": str(fused["scan_id"]),
            "status": "selected",
            "up": _up(consensus["selected_up"]),
            "selection_basis": str(consensus["selection_basis"]),
            "visual_selection": None,
        }
    if status != "needs_review":
        raise ValueError(f"unsupported consensus status: {status!r}")
    if not isinstance(visual_selection, dict):
        raise ValueError("review orientation requires a visual selection")
    reviewer = str(visual_selection.get("reviewer", "")).strip()
    reason = str(visual_selection.get("reason", "")).strip()
    if not reviewer or not reason:
        raise ValueError("visual selection requires reviewer and reason")
    candidate_value = visual_selection.get("candidate", 0)
    candidate_number = int(candidate_value)
    candidates = consensus.get("ranked_candidates")
    if (
        not isinstance(candidates, list)
        or candidate_number < 1
        or candidate_number > len(candidates)
    ):
        raise ValueError("visual selection candidate is out of range")
    return {
        "schema_version": 1,
        "scan_id": str(fused["scan_id"]),
        "status": "selected",
        "up": _up(candidates[candidate_number - 1]["consensus_up"]),
        "selection_basis": "visual_candidate_review",
        "visual_selection": {
            **visual_selection,
            "candidate": candidate_number,
            "reviewer": reviewer,
            "reason": reason,
        },
    }
