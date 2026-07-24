from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from railing_removal.slider_gallery import build_slider_gallery_assets


def test_build_slider_gallery_assets_exports_aligned_web_images_and_flags(
    tmp_path: Path,
) -> None:
    scan_output = tmp_path / "cleanup" / "scan-1"
    for layer, color in (
        ("source", (80, 90, 100)),
        ("plant", (30, 110, 50)),
    ):
        proof = scan_output / "final" / f"render-{layer}" / "front-rgb.png"
        proof.parent.mkdir(parents=True)
        Image.new("RGB", (2400, 1800), color).save(proof)

    batch_report = tmp_path / "batch-report.json"
    batch_report.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "scan_id": "scan-1",
                        "status": "complete",
                        "output": str(scan_output),
                        "counts": {"source": 100, "plant_cleaned": 60},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    publication_plan = tmp_path / "publication-plan.json"
    publication_plan.write_text(
        json.dumps(
            {
                "scans": [
                    {
                        "scan_id": "scan-1",
                        "action": "publish_flag",
                        "reason": "No coherent plant structure.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_slider_gallery_assets(
        batch_report,
        publication_plan,
        tmp_path / "public-gallery",
        max_width=1600,
    )

    manifest = json.loads(
        (tmp_path / "public-gallery" / "manifest.json").read_text()
    )
    scan = manifest["scans"][0]
    assert scan["before"] == "images/001-before.webp"
    assert scan["after"] == "images/001-after.webp"
    assert scan["action"] == "publish_flag"
    assert scan["reason"] == "No coherent plant structure."
    with Image.open(tmp_path / "public-gallery" / scan["before"]) as before:
        assert before.size == (1600, 1200)
    with Image.open(tmp_path / "public-gallery" / scan["after"]) as after:
        assert after.size == before.size
    assert result["total"] == 1
    assert result["flagged"] == 1
