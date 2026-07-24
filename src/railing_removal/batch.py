from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from plant_cleanup.clipseg_votes import HuggingFaceClipSegPredictor
from plant_cleanup.dense_semantic import HuggingFaceOneFormerPredictor
from plant_cleanup.sam2_votes import HuggingFaceSam2Predictor
from railing_removal.full_pipeline import run_full_cleanup


Progress = Callable[[str], None]


def load_batch_manifest(path: Path) -> list[dict[str, Any]]:
    manifest_path = path.resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1:
        raise ValueError("batch manifest schema_version must be 1")
    scans = value.get("scans")
    if not isinstance(scans, list) or not scans:
        raise ValueError("batch manifest requires a nonempty scans list")
    scan_ids: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in scans:
        scan_id = str(item.get("scan_id", "")).strip()
        if not scan_id or scan_id in scan_ids:
            raise ValueError("batch scan IDs must be nonempty and unique")
        scan_ids.add(scan_id)
        source = Path(item["source"]).resolve()
        config = Path(item["config"]).resolve()
        plan_value = item.get("scene_plan")
        scene_plan = (
            Path(plan_value).resolve()
            if plan_value is not None
            else None
        )
        review_artifacts = item.get("review_artifacts", True)
        if not isinstance(review_artifacts, bool):
            raise ValueError("review_artifacts must be a boolean")
        if not source.is_file():
            raise FileNotFoundError(source)
        if not config.is_file():
            raise FileNotFoundError(config)
        if scene_plan is not None and not scene_plan.is_file():
            raise FileNotFoundError(scene_plan)
        result.append(
            {
                "scan_id": scan_id,
                "source": source,
                "config": config,
                "scene_plan": scene_plan,
                "review_artifacts": review_artifacts,
            }
        )
    return result


def run_batch(
    manifest_path: Path,
    output_root: Path,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    scans = load_batch_manifest(manifest_path)
    output_root = output_root.resolve()
    report_path = output_root / "batch-report.json"
    if report_path.exists():
        raise FileExistsError(f"batch is already complete: {report_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda _: None)

    first_config = json.loads(
        scans[0]["config"].read_text(encoding="utf-8")
    )
    vision = first_config["vision"]
    clipseg_predictor = HuggingFaceClipSegPredictor(
        vision["clipseg_model"]
    )
    sam2_predictor = HuggingFaceSam2Predictor(vision["sam2_model"])
    dense_values = first_config.get("dense_semantic")
    dense_predictor = (
        HuggingFaceOneFormerPredictor(dense_values["model"])
        if dense_values
        else None
    )

    results: list[dict[str, Any]] = []
    for index, item in enumerate(scans, start=1):
        scan_id = item["scan_id"]
        destination = output_root / scan_id
        run_report = destination / "run-report.json"
        if run_report.is_file():
            progress(f"[{index}/{len(scans)}] already complete {scan_id}")
            report = json.loads(run_report.read_text(encoding="utf-8"))
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "complete",
                    "counts": report["counts"],
                    "output": str(destination),
                }
            )
            continue
        if destination.exists():
            progress(f"[{index}/{len(scans)}] partial output {scan_id}")
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "partial",
                    "output": str(destination),
                }
            )
            continue
        progress(f"[{index}/{len(scans)}] processing {scan_id}")
        try:
            report = run_full_cleanup(
                item["source"],
                destination,
                config_path=item["config"],
                railing_plan_path=item["scene_plan"],
                clipseg_predictor=clipseg_predictor,
                sam2_predictor=sam2_predictor,
                dense_predictor=dense_predictor,
                build_review_artifacts=item["review_artifacts"],
                progress=lambda stage, prefix=scan_id: progress(
                    f"[{prefix}] {stage}"
                ),
            )
        except Exception as error:
            progress(f"[{index}/{len(scans)}] failed {scan_id}: {error}")
            results.append(
                {
                    "scan_id": scan_id,
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "output": str(destination),
                }
            )
            continue
        results.append(
            {
                "scan_id": scan_id,
                "status": "complete",
                "counts": report["counts"],
                "output": str(destination),
            }
        )
        progress(f"[{index}/{len(scans)}] complete {scan_id}")

    summary = {
        status: sum(result["status"] == status for result in results)
        for status in ("complete", "partial", "failed")
    }
    report = {
        "schema_version": 1,
        "manifest": str(manifest_path.resolve()),
        "output_root": str(output_root),
        "summary": summary,
        "results": results,
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
