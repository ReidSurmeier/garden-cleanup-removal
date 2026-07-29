from __future__ import annotations

import numpy as np

from railing_removal.orientation_consensus import (
    OrientationEvidence,
    resolve_orientation_consensus,
)


def _evidence(
    family: str,
    direction: tuple[float, float, float],
    *,
    confidence: float = 1.0,
) -> OrientationEvidence:
    return OrientationEvidence(
        family=family,
        source=f"{family}-test",
        up=np.asarray(direction, dtype=np.float64),
        confidence=confidence,
    )


def test_three_independent_families_reject_a_bad_ground_candidate() -> None:
    result = resolve_orientation_consensus(
        [
            _evidence("ground", (0.0, 1.0, 0.0)),
            _evidence("camera", (0.0, 0.0, 1.0)),
            _evidence("photo", (0.02, 0.0, 0.9998)),
            _evidence("vertical", (-0.01, 0.01, 0.9999)),
        ]
    )

    assert result["status"] == "automatic"
    assert result["supporting_families"] == ["camera", "photo", "vertical"]
    assert result["rejected_families"] == ["ground"]
    np.testing.assert_allclose(
        result["consensus_up"],
        (0.0, 0.0, 1.0),
        atol=0.02,
    )


def test_ground_vertical_and_photo_can_reject_bad_camera_orientation() -> None:
    result = resolve_orientation_consensus(
        [
            _evidence("ground", (0.0, 0.0, 1.0)),
            _evidence("camera", (1.0, 0.0, 0.0)),
            _evidence("photo", (0.01, -0.01, 0.9999)),
            _evidence("vertical", (-0.02, 0.0, 0.9998)),
        ]
    )

    assert result["status"] == "automatic"
    assert result["supporting_families"] == ["ground", "photo", "vertical"]
    assert result["rejected_families"] == ["camera"]


def test_two_against_two_is_ranked_but_never_automatic() -> None:
    result = resolve_orientation_consensus(
        [
            _evidence("ground", (0.0, 0.0, 1.0)),
            _evidence("vertical", (0.0, 0.0, 1.0)),
            _evidence("camera", (0.0, 1.0, 0.0)),
            _evidence("photo", (0.0, 1.0, 0.0)),
        ]
    )

    assert result["status"] == "needs_review"
    assert result["reason"] == "insufficient_independent_family_agreement"
    assert len(result["ranked_candidates"]) == 2
    assert result["ranked_candidates"][0]["family_count"] == 2
    assert result["ranked_candidates"][1]["family_count"] == 2


def test_competing_three_family_hypotheses_require_visual_review() -> None:
    result = resolve_orientation_consensus(
        [
            _evidence("ground", (0.0, 0.0, 1.0)),
            _evidence("ground", (0.0, 1.0, 0.0)),
            _evidence("vertical", (0.0, 0.0, 1.0)),
            _evidence("vertical", (0.0, 1.0, 0.0)),
            _evidence("camera", (0.0, 0.0, 1.0)),
            _evidence("photo", (0.0, 1.0, 0.0)),
        ]
    )

    assert result["status"] == "needs_review"
    assert result["reason"] == "competing_consensus_hypotheses"
    assert result["ranked_candidates"][0]["family_count"] == 3
    assert result["ranked_candidates"][1]["family_count"] == 3
