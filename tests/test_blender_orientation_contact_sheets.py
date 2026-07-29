from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image


def test_contact_sheet_compares_identity_with_first_alternative(
    tmp_path: Path,
) -> None:
    batch_root = tmp_path / "batch"
    report_root = batch_root / "cohort-001" / "scan-a"
    report_root.mkdir(parents=True)
    renders = {}
    for candidate, color in (
        ("identity", (35, 120, 60)),
        ("candidate-1", (80, 65, 145)),
    ):
        candidate_root = report_root / candidate
        candidate_root.mkdir()
        renders[candidate] = {}
        for view in ("front", "side"):
            path = candidate_root / f"{view}.png"
            Image.new("RGB", (640, 480), color).save(path)
            renders[candidate][view] = str(path)
    (report_root / "evaluation-report.json").write_text(
        json.dumps(
            {
                "scan_id": "scan-a",
                "candidates": [
                    {
                        "candidate": "identity",
                        "rotation_degrees": 0.0,
                        "renders": renders["identity"],
                    },
                    {
                        "candidate": "candidate-1",
                        "rotation_degrees": 37.5,
                        "renders": renders["candidate-1"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "sheets"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_blender_orientation_contact_sheets.py",
            str(batch_root),
            str(output_root),
        ],
        check=True,
    )

    index = json.loads((output_root / "index.json").read_text())
    assert index["report_count"] == 1
    assert index["sheets"][0]["scan_ids"] == ["scan-a"]
    with Image.open(output_root / "sheet-01.jpg") as sheet:
        assert sheet.width == 1320
        assert sheet.height == 370
