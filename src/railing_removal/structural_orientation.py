from __future__ import annotations

from typing import Any

import numpy as np

from railing_removal.orientation_consensus import (
    OrientationEvidence,
    resolve_orientation_consensus,
)


def _camera_evidence(
    inventory: dict[str, Any],
) -> OrientationEvidence | None:
    vectors = []
    for camera in inventory.get("cameras", []):
        value = camera.get("source_frame_up")
        if not camera.get("enabled") or not camera.get("aligned"):
            continue
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            continue
        length = float(np.linalg.norm(vector))
        if length > 1e-12:
            vectors.append(vector / length)
    if not vectors:
        return None
    consensus = np.sum(vectors, axis=0)
    if float(np.linalg.norm(consensus)) < 1e-12:
        return None
    return OrientationEvidence(
        family="camera",
        source="aligned-camera-image-up-consensus",
        up=consensus,
        confidence=0.65,
        metadata={"camera_count": len(vectors)},
    )


def _ground_evidence(
    cleanup: dict[str, Any],
) -> list[OrientationEvidence]:
    floor = cleanup.get("floor")
    components = (
        floor.get("floor_components", [])
        if isinstance(floor, dict)
        else []
    )
    result = []
    for component in components:
        points = max(
            int(component.get("matched_source_point_count", 0)),
            int(component.get("seed_point_count", 0)),
        )
        vector = np.asarray(component.get("normal"), dtype=np.float64)
        if (
            points < 500
            or vector.shape != (3,)
            or not np.all(np.isfinite(vector))
            or float(np.linalg.norm(vector)) < 1e-12
        ):
            continue
        if vector[2] < 0:
            vector = -vector
        result.append(
            OrientationEvidence(
                family="ground",
                source=(
                    "connected-floor-component-"
                    f"{component.get('label', 'unknown')}"
                ),
                up=vector,
                confidence=min(1.0, 0.55 + np.log10(points) / 12.0),
                metadata={"point_count": points},
            )
        )
    return result


def _empty_consensus() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "needs_review",
        "reason": "no_structural_orientation_evidence",
        "consensus_up": [0.0, 0.0, 1.0],
        "selected_up": [0.0, 0.0, 1.0],
        "selection_basis": "preserve_existing_orientation",
        "supporting_families": [],
        "rejected_families": [],
        "ranked_candidates": [],
        "parameters": {
            "maximum_agreement_degrees": 20.0,
            "minimum_automatic_families": 3,
        },
    }


def build_structural_orientation_report(
    scan_id: str,
    inventory: dict[str, Any],
    cleanup: dict[str, Any],
) -> dict[str, Any]:
    """Fuse read-only camera and connected-ground orientation evidence."""

    if not inventory.get("project_opened_read_only"):
        raise ValueError("camera inventory lacks read-only provenance")
    if not cleanup.get("source_opened_read_only"):
        raise ValueError("cleanup report lacks read-only provenance")
    evidence = _ground_evidence(cleanup)
    camera = _camera_evidence(inventory)
    if camera is not None:
        evidence.append(camera)
    return {
        "schema_version": 1,
        "scan_id": scan_id,
        "source_data_opened_read_only": True,
        "evidence": [
            {
                "family": item.family,
                "source": item.source,
                "up": item.up.tolist(),
                "confidence": item.confidence,
                "metadata": item.metadata,
            }
            for item in evidence
        ],
        "consensus": (
            resolve_orientation_consensus(evidence)
            if evidence
            else _empty_consensus()
        ),
    }
