from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from railing_removal.batch_review import (
    build_batch_review,
    build_paginated_review,
)


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


def test_paginated_review_builds_bounded_contact_sheets(
    tmp_path: Path,
) -> None:
    results = []
    for index in range(3):
        scan_id = f"scan-{index}"
        scan_output = tmp_path / "cleanup" / scan_id
        for layer in ("source", "plant", "conservative", "rejected"):
            destination = scan_output / "final" / f"render-{layer}"
            destination.mkdir(parents=True)
            Image.new("RGB", (32, 24), (index * 20, 80, 40)).save(
                destination / "front-rgb.png"
            )
        results.append(
            {
                "scan_id": scan_id,
                "status": "complete",
                "output": str(scan_output),
            }
        )
    report_path = tmp_path / "batch-report.json"
    report_path.write_text(
        json.dumps({"results": results}),
        encoding="utf-8",
    )
    output = tmp_path / "pages"

    report = build_paginated_review(
        report_path,
        output,
        page_size=2,
    )

    assert report["page_count"] == 2
    assert (output / "page-001.jpg").is_file()
    assert (output / "page-002.jpg").is_file()
    assert (output / "index.html").is_file()
    document = (output / "index.html").read_text(encoding="utf-8")
    assert "../cleanup/scan-0/review/viewer.html" in document
    assert "scan-0 3D" in document

    with pytest.raises(FileExistsError, match="already exists"):
        build_paginated_review(report_path, output, page_size=2)
