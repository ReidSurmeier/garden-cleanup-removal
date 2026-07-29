from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from railing_removal.orientation_consensus import (  # noqa: E402
    OrientationEvidence,
    resolve_orientation_consensus,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")


def _record(item: OrientationEvidence) -> dict[str, Any]:
    return {
        "family": item.family,
        "source": item.source,
        "up": item.up.tolist(),
        "confidence": item.confidence,
        "metadata": item.metadata,
    }


def fuse_scan(
    photo_report: dict[str, Any],
    geometry_report: dict[str, Any],
) -> dict[str, Any]:
    evidence = [
        OrientationEvidence(
            family=str(item["family"]),
            source=str(item["source"]),
            up=np.asarray(item["up"], dtype=np.float64),
            confidence=float(item["confidence"]),
            metadata=item.get("metadata"),
        )
        for item in photo_report["evidence"]
    ]
    for index, candidate in enumerate(
        geometry_report["geometry"]["candidates"],
        start=1,
    ):
        support_fraction = float(candidate["support_fraction"])
        evidence.append(
            OrientationEvidence(
                family="vertical",
                source=f"rigid-surface-axis-{index}",
                up=np.asarray(candidate["axis"], dtype=np.float64),
                confidence=min(0.9, 0.55 + 4.0 * support_fraction),
                metadata={
                    "support_count": candidate["support_count"],
                    "support_fraction": support_fraction,
                    "median_residual_degrees": candidate[
                        "median_residual_degrees"
                    ],
                },
            )
        )
    return {
        "schema_version": 1,
        "scan_id": photo_report["scan_id"],
        "source_data_opened_read_only": True,
        "evidence": [_record(item) for item in evidence],
        "consensus": resolve_orientation_consensus(evidence),
    }


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: fuse_orientation_evidence_batch.py "
            "PROJECTS.json PHOTO_EVIDENCE_ROOT GEOMETRY_EVIDENCE_ROOT "
            "OUTPUT_ROOT"
        )
    manifest_path = Path(sys.argv[1]).resolve()
    photo_root = Path(sys.argv[2]).resolve()
    geometry_root = Path(sys.argv[3]).resolve()
    output_root = Path(sys.argv[4]).resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    manifest = _read_json(manifest_path)
    output_root.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    for item in manifest["projects"]:
        scan_id = str(item["scan_id"])
        fused = fuse_scan(
            _read_json(
                photo_root / scan_id / "orientation-evidence.json"
            ),
            _read_json(
                geometry_root / scan_id / "geometry-orientation.json"
            ),
        )
        _write_json(output_root / scan_id / "fused-orientation.json", fused)
        results.append(
            {
                "scan_id": scan_id,
                "status": fused["consensus"]["status"],
                "reason": fused["consensus"]["reason"],
                "consensus_up": fused["consensus"]["consensus_up"],
                "supporting_families": fused["consensus"][
                    "supporting_families"
                ],
                "ranked_candidates": fused["consensus"][
                    "ranked_candidates"
                ],
            }
        )
    report = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "source_data_opened_read_only": True,
        "results": results,
    }
    _write_json(output_root / "batch-report.json", report)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
