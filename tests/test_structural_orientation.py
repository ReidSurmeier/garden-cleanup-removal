from __future__ import annotations

from railing_removal.structural_orientation import (
    build_structural_orientation_report,
)


def test_structural_orientation_combines_camera_and_connected_ground(
) -> None:
    inventory = {
        "project_opened_read_only": True,
        "cameras": [
            {
                "enabled": True,
                "aligned": True,
                "source_frame_up": [0.0, 0.0, 1.0],
            },
            {
                "enabled": True,
                "aligned": True,
                "source_frame_up": [0.0, 0.02, 0.9998],
            },
        ],
    }
    cleanup = {
        "source_opened_read_only": True,
        "floor": {
            "floor_components": [
                {
                    "label": 7,
                    "normal": [0.0, -0.01, 0.99995],
                    "matched_source_point_count": 2_400,
                    "seed_point_count": 1_000,
                }
            ]
        },
    }

    report = build_structural_orientation_report(
        "scan-a",
        inventory,
        cleanup,
    )

    assert report["scan_id"] == "scan-a"
    assert report["source_data_opened_read_only"]
    assert {item["family"] for item in report["evidence"]} == {
        "camera",
        "ground",
    }
    assert report["consensus"]["status"] == "needs_review"
    assert report["consensus"]["supporting_families"] == [
        "camera",
        "ground",
    ]
