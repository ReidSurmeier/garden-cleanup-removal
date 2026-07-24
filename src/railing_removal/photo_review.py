from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic"}
CARD_SIZE = (960, 360)
FRAME_SIZE = (320, 320)


def _save_jpeg_new(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        image.save(destination, format="JPEG", quality=86, optimize=True)


def _representative(paths: list[Path], count: int) -> list[Path]:
    if not paths:
        return []
    if len(paths) == 1:
        return [paths[0]] * count
    return [
        paths[
            min(
                len(paths) - 1,
                round((index + 1) * (len(paths) - 1) / (count + 1)),
            )
        ]
        for index in range(count)
    ]


def _scan_card(scan_id: str, paths: list[Path], count: int) -> Image.Image:
    card = Image.new("RGB", CARD_SIZE, (18, 20, 24))
    draw = ImageDraw.Draw(card)
    if not paths:
        draw.text(
            (24, 150),
            f"{scan_id} — original photos unavailable",
            fill=(240, 180, 100),
        )
        return card

    for index, path in enumerate(_representative(paths, count)):
        with Image.open(path) as source:
            frame = ImageOps.exif_transpose(source).convert("RGB")
            frame = ImageOps.fit(frame, FRAME_SIZE)
        card.paste(frame, (index * FRAME_SIZE[0], 40))
    draw.rectangle((0, 0, CARD_SIZE[0], 40), fill=(9, 10, 13))
    draw.text((12, 13), scan_id, fill=(245, 245, 245))
    return card


def build_reference_review(
    manifest_path: Path,
    output_root: Path,
    *,
    frames_per_scan: int = 3,
    scans_per_page: int = 20,
) -> dict[str, Any]:
    """Build compact visual evidence from photos without modifying them."""

    if frames_per_scan < 1 or scans_per_page < 1:
        raise ValueError("review sizes must be positive")
    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()
    report_path = output_root / "reference-review-report.json"
    if report_path.exists():
        raise FileExistsError(
            f"reference review is already finalized: {report_path}"
        )
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    projects = value.get("projects")
    if value.get("schema_version") != 1 or not isinstance(projects, list):
        raise ValueError("invalid project manifest")
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    card_paths: list[Path] = []
    for item in projects:
        scan_id = str(item["scan_id"])
        if Path(scan_id).name != scan_id or scan_id in {".", ".."}:
            raise ValueError(f"unsafe scan ID: {scan_id}")
        photo_value = item.get("photo_dir")
        photo_dir = Path(photo_value).resolve() if photo_value else None
        photos = (
            sorted(
                path
                for path in photo_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if photo_dir is not None and photo_dir.is_dir()
            else []
        )
        card_path = output_root / "scans" / f"{scan_id}.jpg"
        if not card_path.exists():
            _save_jpeg_new(
                _scan_card(scan_id, photos, frames_per_scan),
                card_path,
            )
        card_paths.append(card_path)
        results.append(
            {
                "scan_id": scan_id,
                "status": "complete" if photos else "missing_photos",
                "photo_count": len(photos),
                "selected_photos": [
                    str(path)
                    for path in _representative(photos, frames_per_scan)
                ],
                "card": str(card_path),
            }
        )

    columns = 2
    rows = math.ceil(scans_per_page / columns)
    page_size = (CARD_SIZE[0] * columns, CARD_SIZE[1] * rows)
    page_paths: list[str] = []
    for page_index, start in enumerate(
        range(0, len(card_paths), scans_per_page),
        start=1,
    ):
        page_path = output_root / "pages" / f"page-{page_index:03d}.jpg"
        if not page_path.exists():
            page = Image.new("RGB", page_size, (5, 6, 8))
            for tile_index, card_path in enumerate(
                card_paths[start : start + scans_per_page]
            ):
                with Image.open(card_path) as card:
                    x = (tile_index % columns) * CARD_SIZE[0]
                    y = (tile_index // columns) * CARD_SIZE[1]
                    page.paste(card, (x, y))
            _save_jpeg_new(page, page_path)
        page_paths.append(str(page_path))

    summary = {
        status: sum(result["status"] == status for result in results)
        for status in ("complete", "missing_photos")
    }
    report = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "frames_per_scan": frames_per_scan,
        "summary": summary,
        "pages": page_paths,
        "results": results,
    }
    with report_path.open("x", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return report
