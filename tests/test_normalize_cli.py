from __future__ import annotations

import json
import sys
from pathlib import Path

from railing_removal import normalize_cli


def test_normalize_cli_runs_cleanup_contract(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "source.ply"
    cleanup = tmp_path / "cleanup"
    inventory = tmp_path / "camera-inventory.json"
    output = tmp_path / "normalized"
    calls: list[tuple[Path, Path, Path, Path]] = []

    def normalize(
        source_path: Path,
        cleanup_run: Path,
        camera_inventory_path: Path,
        output_dir: Path,
    ) -> dict[str, object]:
        calls.append(
            (
                source_path,
                cleanup_run,
                camera_inventory_path,
                output_dir,
            )
        )
        return {
            "plan": {"status": "automatic", "scale": 0.5},
            "ground_evidence": {"point_count": 123},
        }

    monkeypatch.setattr(normalize_cli, "normalize_cleanup_run", normalize)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "garden-normalize-cleanup",
            str(source),
            str(cleanup),
            str(inventory),
            str(output),
        ],
    )

    normalize_cli.main()

    assert calls == [(source, cleanup, inventory, output)]
    assert json.loads(capsys.readouterr().out) == {
        "ground_point_count": 123,
        "scale": 0.5,
        "status": "automatic",
    }
