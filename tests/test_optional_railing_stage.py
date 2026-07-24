from __future__ import annotations

from pathlib import Path

from railing_removal.full_cli import main


def test_full_cleanup_cli_allows_scan_without_a_railing_plan(
    monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_run(
        source: Path,
        output: Path,
        *,
        config_path: Path,
        railing_plan_path: Path | None,
        progress,
    ) -> dict:
        observed["railing_plan_path"] = railing_plan_path
        return {"counts": {}}

    monkeypatch.setattr(
        "railing_removal.full_cli.run_full_cleanup",
        fake_run,
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "garden-full-cleanup",
            "source.ply",
            "output",
            "--config",
            "config.json",
        ],
    )

    main()

    assert observed["railing_plan_path"] is None
