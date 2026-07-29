from __future__ import annotations

import json
from pathlib import Path

from railing_removal.blender_orientation_review_site import (
    build_orientation_review_site,
)


def _report(root: Path, scan_id: str, angle: float) -> None:
    scan = root / "cohort-001" / scan_id
    scan.mkdir(parents=True)
    candidates = []
    for candidate, rotation in (("identity", 0.0), ("1", angle)):
        render_root = scan / f"candidate-{candidate}"
        render_root.mkdir()
        renders = {}
        for view in ("front", "side", "top"):
            render = render_root / f"{view}.png"
            render.write_bytes(b"render")
            renders[view] = str(render)
        candidates.append(
            {
                "candidate": candidate,
                "rotation_degrees": rotation,
                "selection_basis": (
                    "preserve_existing_orientation"
                    if candidate == "identity"
                    else "ranked_review_hypothesis"
                ),
                "renders": renders,
            }
        )
    (scan / "evaluation-report.json").write_text(
        json.dumps({"scan_id": scan_id, "candidates": candidates}),
        encoding="utf-8",
    )


def test_review_site_links_full_resolution_identity_and_candidates(
    tmp_path: Path,
) -> None:
    batch_root = tmp_path / "batch"
    _report(batch_root, "scan-a", 37.5)
    _report(batch_root, "scan-b", 12.0)
    destination = batch_root / "viewer.html"

    result = build_orientation_review_site(batch_root, destination)

    html = destination.read_text(encoding="utf-8")
    assert result == {"scan_count": 2, "candidate_count": 4}
    assert "scan-a" in html
    assert "scan-b" in html
    assert "37.5" in html
    assert (
        "cohort-001/scan-a/candidate-identity/front.png"
        in html
    )
    assert "loading=\"lazy\"" in html
    assert "data-view=\"side\"" in html
