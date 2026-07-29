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


def test_validated_ground_is_the_rotation_anchor_not_an_averaged_vector() -> None:
    ground = np.array((0.0, -0.258819, 0.965926))
    result = resolve_orientation_consensus(
        [
            _evidence("ground", tuple(ground)),
            _evidence("camera", (0.0, 0.0, 1.0)),
            _evidence("photo", (0.0, -0.241922, 0.970296)),
            _evidence("vertical", (0.0, -0.275637, 0.961262)),
        ]
    )

    assert result["status"] == "automatic"
    assert result["selection_basis"] == "validated_ground_anchor"
    np.testing.assert_allclose(result["selected_up"], ground, atol=1e-6)
    assert not np.allclose(result["consensus_up"], ground)


def test_sloped_ground_is_not_gravity_without_a_vertical_structure_vote() -> None:
    ground = np.array((0.0, -0.258819, 0.965926))
    result = resolve_orientation_consensus(
        [
            _evidence("ground", tuple(ground)),
            _evidence("camera", (0.0, 0.0, 1.0)),
            _evidence("photo", (0.0, -0.104528, 0.994522)),
        ]
    )

    assert result["status"] == "automatic"
    assert result["selection_basis"] == "weighted_family_consensus"
    assert not np.allclose(result["selected_up"], ground)
