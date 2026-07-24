from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from railing_removal.batch_review import build_batch_review


def test_build_batch_review_links_proofs_and_full_viewer(
    tmp_path: Path,
) -> None:
    output = tmp_path / "batch"
    scan = output / "scan with spaces"
    for layer in ("source", "plant", "conservative", "rejected"):
        proof = scan / "final" / f"render-{layer}" / "front-rgb.png"
        proof.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12), (20, 80, 30)).save(proof)
    report = output / "batch-report.json"
    report.write_text(
        json.dumps(
            {
                "output_root": str(output),
                "results": [
                    {
                        "scan_id": "scan with spaces",
                        "status": "complete",
                        "output": str(scan),
                        "counts": {"source": 100, "final": 40},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build_batch_review(report)

    document = (output / "index.html").read_text(encoding="utf-8")
    assert "scan%20with%20spaces/review/viewer.html" in document
    assert "render-plant/front-rgb.png" in document
    assert result["complete"] == 1
    with Image.open(output / "batch-review.jpg") as sheet:
        assert sheet.size == (1280, 282)
