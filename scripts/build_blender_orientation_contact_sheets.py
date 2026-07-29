from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = (18, 21, 25)
PANEL = (31, 36, 43)
TEXT = (238, 242, 246)
MUTED = (166, 176, 187)
CELL = 260
LABEL_WIDTH = 280
ROW_HEIGHT = CELL + 58
ROWS_PER_SHEET = 20


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _thumbnail(path: Path) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    image.thumbnail((CELL, CELL), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (CELL, CELL), PANEL)
    canvas.paste(
        image,
        ((CELL - image.width) // 2, (CELL - image.height) // 2),
    )
    return canvas


def _candidate(
    report: dict[str, Any],
    identifier: str,
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in report["candidates"]
            if str(item["candidate"]) == identifier
        ),
        None,
    )


def _first_alternative(
    report: dict[str, Any],
) -> dict[str, Any] | None:
    return next(
        (
            item
            for item in report["candidates"]
            if str(item["candidate"]) != "identity"
        ),
        None,
    )


def _render_path(
    item: dict[str, Any] | None,
    view: str,
) -> Path | None:
    if item is None:
        return None
    path = Path(item["renders"][view])
    return path if path.is_file() else None


def _draw_sheet(
    reports: list[dict[str, Any]],
    output: Path,
) -> None:
    width = LABEL_WIDTH + 4 * CELL
    height = 52 + len(reports) * ROW_HEIGHT
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=15)
    headings = ("Identity front", "Identity side", "Alternative front", "Alternative side")
    for index, heading in enumerate(headings):
        draw.text(
            (LABEL_WIDTH + index * CELL + 10, 16),
            heading,
            fill=MUTED,
            font=small,
        )
    for row, report in enumerate(reports):
        top = 52 + row * ROW_HEIGHT
        if row % 2:
            draw.rectangle(
                (0, top, width, top + ROW_HEIGHT),
                fill=(23, 27, 32),
            )
        identity = _candidate(report, "identity")
        alternative = _first_alternative(report)
        draw.text(
            (12, top + 16),
            str(report["scan_id"]),
            fill=TEXT,
            font=font,
        )
        if alternative is None:
            detail = "identity only"
        else:
            detail = (
                f"alt {alternative['candidate']} · "
                f"{alternative['rotation_degrees']:.1f}°"
            )
        draw.text(
            (12, top + 46),
            detail,
            fill=MUTED,
            font=small,
        )
        paths = (
            _render_path(identity, "front"),
            _render_path(identity, "side"),
            _render_path(alternative, "front"),
            _render_path(alternative, "side"),
        )
        for column, path in enumerate(paths):
            left = LABEL_WIDTH + column * CELL
            if path is not None:
                sheet.paste(_thumbnail(path), (left, top))
            else:
                draw.rectangle(
                    (left, top, left + CELL - 1, top + CELL - 1),
                    fill=PANEL,
                    outline=(64, 72, 82),
                )
                draw.text(
                    (left + 18, top + CELL // 2),
                    "No alternative",
                    fill=MUTED,
                    font=small,
                )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "JPEG", quality=90, optimize=True)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: build_blender_orientation_contact_sheets.py "
            "BATCH_ROOT OUTPUT_ROOT"
        )
    batch_root = Path(sys.argv[1]).resolve()
    output_root = Path(sys.argv[2]).resolve()
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    report_paths = sorted(
        batch_root.glob("cohort-*/**/evaluation-report.json")
    )
    reports = [_read(path) for path in report_paths]
    if not reports:
        raise ValueError("no Blender evaluation reports found")
    output_root.mkdir(parents=True)
    sheets = []
    for offset in range(0, len(reports), ROWS_PER_SHEET):
        group = reports[offset : offset + ROWS_PER_SHEET]
        destination = output_root / f"sheet-{offset // ROWS_PER_SHEET + 1:02d}.jpg"
        _draw_sheet(group, destination)
        sheets.append(
            {
                "path": str(destination),
                "scan_ids": [item["scan_id"] for item in group],
            }
        )
    index = {
        "schema_version": 1,
        "batch_root": str(batch_root),
        "report_count": len(reports),
        "sheets": sheets,
    }
    with (output_root / "index.json").open("x", encoding="utf-8") as destination:
        json.dump(index, destination, indent=2, sort_keys=True)
        destination.write("\n")
    print(
        f"built {len(sheets)} contact sheets for {len(reports)} scans",
        flush=True,
    )


if __name__ == "__main__":
    main()
