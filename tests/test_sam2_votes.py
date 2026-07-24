from __future__ import annotations

import numpy as np
from PIL import Image

from plant_cleanup.sam2_votes import _try_predict_competing_masks


def test_failed_sam2_anchors_become_a_skipped_view_not_a_failed_scan() -> None:
    def predictor(
        image: Image.Image,
        points: list[list[float]],
        labels: list[int],
    ) -> tuple[np.ndarray, float]:
        raise ValueError("no prompt-consistent foreground mask")

    result, report = _try_predict_competing_masks(
        predictor,
        Image.new("RGB", (8, 8)),
        planter_anchors=[[2.0, 2.0]],
        plant_anchors=[[6.0, 6.0]],
    )

    assert result is None
    assert report["status"] == "segmentation_failed"
    assert "prompt-consistent" in report["error"]
