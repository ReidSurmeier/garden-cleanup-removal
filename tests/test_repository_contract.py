from __future__ import annotations

import re
import tomllib
from pathlib import Path

import numpy as np
import pytest

from railing_removal import full_pipeline

ROOT = Path(__file__).resolve().parents[1]


def test_repository_exposes_the_documentation_and_validation_contract() -> None:
    required_paths = {
        "AGENTS.md",
        "CONTEXT.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "PROJECT.md",
        "README.md",
        ".github/workflows/validate.yml",
        "docs/adr/0001-preserve-sources-and-version-derived-output.md",
        "docs/adr/0002-separate-cleanup-orientation-and-publication.md",
        "docs/adr/0003-keep-review-transport-non-authoritative.md",
        "docs/agents/domain.md",
        "docs/agents/issue-tracker.md",
        "docs/agents/triage-labels.md",
        "docs/normalization-production-safety.md",
    }

    assert {
        path for path in required_paths if not (ROOT / path).is_file()
    } == set()


def test_readme_orients_people_before_listing_implementation_details() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    headings = re.findall(
        r"^## (.+)$",
        readme,
        flags=re.MULTILINE,
    )
    assert readme.startswith("# Garden Railing Removal\n")
    assert headings[:5] == [
        "Purpose",
        "Safety boundary",
        "Workflow",
        "Development",
        "Repository layout",
    ]
    assert "Garden Scan Cleanup" in readme
    assert "does not authorize production writes" in readme


def test_vision_extra_declares_sam2_runtime_dependencies() -> None:
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    vision = project["project"]["optional-dependencies"]["vision"]

    assert any(dependency.startswith("torchvision") for dependency in vision)


def test_dev_extra_contains_every_dependency_needed_to_collect_tests() -> None:
    project = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    dev = project["project"]["optional-dependencies"]["dev"]

    assert any(
        dependency.startswith("opencv-python-headless") for dependency in dev
    ), "tests import cv2, so a dev-only install must provide OpenCV"


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
