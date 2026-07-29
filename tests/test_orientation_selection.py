from __future__ import annotations

import pytest

from railing_removal.orientation_selection import select_orientation


def _fused(status: str) -> dict[str, object]:
    return {
        "scan_id": "scan-1",
        "consensus": {
            "status": status,
            "selected_up": [0.0, 0.0, 1.0],
            "selection_basis": "weighted_family_consensus",
            "ranked_candidates": [
                {"consensus_up": [1.0, 0.0, 0.0]},
                {"consensus_up": [0.0, 1.0, 0.0]},
            ],
        },
    }


def test_automatic_orientation_uses_the_gated_selected_vector() -> None:
    result = select_orientation(_fused("automatic"))

    assert result["status"] == "selected"
    assert result["up"] == [0.0, 0.0, 1.0]
    assert result["selection_basis"] == "weighted_family_consensus"


def test_review_orientation_uses_the_visually_chosen_candidate() -> None:
    result = select_orientation(
        _fused("needs_review"),
        visual_selection={
            "candidate": 2,
            "reviewer": "codex-visual-review",
            "reason": "ground and trunks are upright in two rendered views",
        },
    )

    assert result["status"] == "selected"
    assert result["up"] == [0.0, 1.0, 0.0]
    assert result["selection_basis"] == "visual_candidate_review"
    assert result["visual_selection"]["candidate"] == 2


def test_review_orientation_fails_closed_without_visual_selection() -> None:
    with pytest.raises(ValueError, match="requires a visual selection"):
        select_orientation(_fused("needs_review"))
