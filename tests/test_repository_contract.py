from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from railing_removal import full_pipeline


ROOT = Path(__file__).resolve().parents[1]


def test_readme_remains_limited_to_models_and_stack() -> None:
    headings = re.findall(
        r"^## (.+)$",
        (ROOT / "README.md").read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert headings == ["Models", "Stack"]


def test_full_pipeline_refuses_mismatched_profile_without_creating_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.ply"
    source.write_bytes(b"source")
    output = tmp_path / "derived"
    config = ROOT / "configs" / "2026-07-15-172629-stride8.json"
    plan = ROOT / "configs" / "2026-07-15-172629-railing.json"
    cloud = np.zeros(
        1,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    monkeypatch.setattr(full_pipeline, "read_cloud", lambda _: cloud)
    monkeypatch.setattr(full_pipeline, "_sha256", lambda _: "wrong")

    with pytest.raises(ValueError, match="profile does not match"):
        full_pipeline.run_full_cleanup(
            source,
            output,
            config_path=config,
            railing_plan_path=plan,
        )

    assert not output.exists()
