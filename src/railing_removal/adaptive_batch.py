from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from plant_cleanup.adaptive_profile import build_adaptive_config
from railing_removal.native_batch import _load_projects


ConfigBuilder = Callable[[Path, dict[str, Any]], dict[str, Any]]
Progress = Callable[[str], None]


def _write_json_atomic_new(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(f"{path.name}.partial-{uuid4().hex}")
    with partial.open("x", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")
    partial.rename(path)


def build_adaptive_batch(
    project_manifest_path: Path,
    canonical_root: Path,
    output_root: Path,
    base_config_path: Path,
    *,
    stride: int = 8,
    config_builder: ConfigBuilder = build_adaptive_config,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build resumable source-bound configs and one cleanup manifest."""

    if stride < 1:
        raise ValueError("stride must be positive")
    projects = _load_projects(project_manifest_path)
    canonical_root = canonical_root.resolve()
    output_root = output_root.resolve()
    base_config_path = base_config_path.resolve()
    manifest_path = output_root / "cleanup-manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"adaptive batch is already finalized: {manifest_path}"
        )
    if not base_config_path.is_file():
        raise FileNotFoundError(base_config_path)
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    config_root = output_root / "configs"
    config_root.mkdir(parents=True, exist_ok=True)
    progress = progress or (lambda _: None)
    scans: list[dict[str, Any]] = []

    for index, item in enumerate(projects, start=1):
        scan_id = item["scan_id"]
        source = (
            canonical_root
            / scan_id
            / f"source-stride{stride}-zup.ply"
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"canonical source is incomplete: {source}"
            )
        config_path = config_root / f"{scan_id}.json"
        if config_path.exists():
            progress(f"[{index}/{len(projects)}] config exists {scan_id}")
            value = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"invalid existing config: {config_path}")
        else:
            progress(f"[{index}/{len(projects)}] profile {scan_id}")
            config = config_builder(source, base_config)
            _write_json_atomic_new(config_path, config)
        scans.append(
            {
                "scan_id": scan_id,
                "source": str(source.resolve()),
                "config": str(config_path.resolve()),
            }
        )

    manifest = {
        "schema_version": 1,
        "project_manifest": str(project_manifest_path.resolve()),
        "canonical_root": str(canonical_root),
        "base_config": str(base_config_path),
        "stride": stride,
        "scans": scans,
    }
    _write_json_atomic_new(manifest_path, manifest)
    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "summary": {"complete": len(scans)},
    }
