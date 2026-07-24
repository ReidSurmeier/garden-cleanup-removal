from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_metashape_reader_import_does_not_require_numpy() -> None:
    code = """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "numpy" or name.startswith("numpy."):
        raise ModuleNotFoundError("numpy blocked like embedded Metashape")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from railing_removal.metashape_reader import export_reader_cloud_readonly
assert callable(export_reader_cloud_readonly)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
