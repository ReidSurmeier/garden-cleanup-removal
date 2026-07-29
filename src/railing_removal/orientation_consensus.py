from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class OrientationEvidence:
    """One independently-produced estimate of physical up."""

    family: str
    source: str
    up: np.ndarray
    confidence: float = 1.0
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        direction = np.asarray(self.up, dtype=np.float64)
        if direction.shape != (3,) or not np.all(np.isfinite(direction)):
            raise ValueError("orientation evidence up vector must be finite 3D")
        length = float(np.linalg.norm(direction))
        if length < 1e-12:
            raise ValueError("orientation evidence up vector cannot be zero")
        if not self.family or not self.source:
            raise ValueError("orientation evidence requires family and source")
        if not np.isfinite(self.confidence) or not 0 < self.confidence <= 1:
            raise ValueError("orientation confidence must be in (0, 1]")
        object.__setattr__(self, "up", direction / length)


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arccos(np.clip(float(left @ right), -1.0, 1.0))
        )
    )


def _candidate_for_seed(
    evidence: list[OrientationEvidence],
    seed: np.ndarray,
    maximum_agreement_degrees: float,
) -> dict[str, Any]:
    inliers = [
        item
        for item in evidence
        if _angle_degrees(item.up, seed) <= maximum_agreement_degrees
    ]
    weighted = np.sum(
        [item.confidence * item.up for item in inliers],
        axis=0,
    )
    consensus = weighted / np.linalg.norm(weighted)
    inliers = [
        item
        for item in evidence
        if _angle_degrees(item.up, consensus) <= maximum_agreement_degrees
    ]
    family_best: dict[str, OrientationEvidence] = {}
    for item in inliers:
        previous = family_best.get(item.family)
        if previous is None or item.confidence > previous.confidence:
            family_best[item.family] = item
    selected = list(family_best.values())
    weighted = np.sum(
        [item.confidence * item.up for item in selected],
        axis=0,
    )
    consensus = weighted / np.linalg.norm(weighted)
    residuals = {
        item.family: _angle_degrees(item.up, consensus)
        for item in selected
    }
    families = sorted(family_best)
    return {
        "consensus_up": consensus.tolist(),
        "family_count": len(families),
        "families": families,
        "score": float(sum(item.confidence for item in selected)),
        "maximum_residual_degrees": max(residuals.values(), default=0.0),
        "residual_degrees": residuals,
        "sources": sorted(item.source for item in selected),
    }


def _same_cluster(
    left: dict[str, Any],
    right: dict[str, Any],
    maximum_agreement_degrees: float,
) -> bool:
    return (
        _angle_degrees(
            np.asarray(left["consensus_up"], dtype=np.float64),
            np.asarray(right["consensus_up"], dtype=np.float64),
        )
        <= maximum_agreement_degrees
    )


def resolve_orientation_consensus(
    evidence: Iterable[OrientationEvidence],
    *,
    maximum_agreement_degrees: float = 20.0,
    minimum_automatic_families: int = 3,
) -> dict[str, Any]:
    """Rank physical-up hypotheses without letting one signal dominate."""

    items = list(evidence)
    if not items:
        raise ValueError("at least one orientation evidence item is required")
    if not 0 < maximum_agreement_degrees < 90:
        raise ValueError("maximum agreement must be between 0 and 90 degrees")
    if minimum_automatic_families < 2:
        raise ValueError("automatic consensus requires at least two families")

    candidates: list[dict[str, Any]] = []
    for item in items:
        candidate = _candidate_for_seed(
            items,
            item.up,
            maximum_agreement_degrees,
        )
        if not any(
            _same_cluster(
                candidate,
                existing,
                maximum_agreement_degrees,
            )
            for existing in candidates
        ):
            candidates.append(candidate)
    candidates.sort(
        key=lambda candidate: (
            -int(candidate["family_count"]),
            -float(candidate["score"]),
            float(candidate["maximum_residual_degrees"]),
            tuple(candidate["families"]),
        )
    )

    winner = candidates[0]
    enough_families = (
        int(winner["family_count"]) >= minimum_automatic_families
    )
    competing_consensus = any(
        int(candidate["family_count"]) >= int(winner["family_count"])
        for candidate in candidates[1:]
    )
    automatic = enough_families and not competing_consensus
    all_families = sorted({item.family for item in items})
    supporting = list(winner["families"])
    winner_up = np.asarray(winner["consensus_up"], dtype=np.float64)
    validated_ground = sorted(
        (
            item
            for item in items
            if item.family == "ground"
            and _angle_degrees(item.up, winner_up)
            <= maximum_agreement_degrees
        ),
        key=lambda item: -item.confidence,
    )
    if automatic and validated_ground:
        selected_up = validated_ground[0].up.tolist()
        selection_basis = "validated_ground_anchor"
    else:
        selected_up = list(winner["consensus_up"])
        selection_basis = (
            "weighted_family_consensus"
            if automatic
            else "ranked_review_hypothesis"
        )
    return {
        "schema_version": 1,
        "status": "automatic" if automatic else "needs_review",
        "reason": (
            "independent_family_consensus"
            if automatic
            else (
                "competing_consensus_hypotheses"
                if enough_families and competing_consensus
                else "insufficient_independent_family_agreement"
            )
        ),
        "consensus_up": list(winner["consensus_up"]),
        "selected_up": selected_up,
        "selection_basis": selection_basis,
        "supporting_families": supporting,
        "rejected_families": sorted(set(all_families) - set(supporting)),
        "ranked_candidates": candidates,
        "parameters": {
            "maximum_agreement_degrees": maximum_agreement_degrees,
            "minimum_automatic_families": minimum_automatic_families,
        },
    }
