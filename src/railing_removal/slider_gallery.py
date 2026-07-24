from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def _web_image(
    source: Path,
    destination: Path,
    *,
    max_width: int,
    quality: int,
) -> tuple[int, int]:
    if not source.is_file():
        raise FileNotFoundError(f"proof image does not exist: {source}")
    with Image.open(source) as original:
        image = original.convert("RGB")
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize(
                (max_width, height),
                Image.Resampling.LANCZOS,
            )
        image.save(destination, "WEBP", quality=quality, method=6)
        return image.size


def build_slider_gallery_assets(
    batch_report_path: Path,
    publication_plan_path: Path,
    output_dir: Path,
    *,
    max_width: int = 1920,
    quality: int = 92,
) -> dict[str, object]:
    if max_width < 1:
        raise ValueError("max_width must be positive")
    if not 1 <= quality <= 100:
        raise ValueError("quality must be between 1 and 100")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"gallery assets already exist: {output_dir}")

    batch = json.loads(
        batch_report_path.resolve().read_text(encoding="utf-8")
    )
    publication = json.loads(
        publication_plan_path.resolve().read_text(encoding="utf-8")
    )
    decisions = {
        str(scan["scan_id"]): scan for scan in publication["scans"]
    }
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True)

    scans: list[dict[str, object]] = []
    for number, result in enumerate(batch["results"], start=1):
        scan_id = str(result["scan_id"])
        scan_output = Path(str(result["output"]))
        before_source = (
            scan_output / "final" / "render-source" / "front-rgb.png"
        )
        after_source = (
            scan_output / "final" / "render-plant" / "front-rgb.png"
        )
        before_name = f"{number:03d}-before.webp"
        after_name = f"{number:03d}-after.webp"
        before_size = _web_image(
            before_source,
            images_dir / before_name,
            max_width=max_width,
            quality=quality,
        )
        after_size = _web_image(
            after_source,
            images_dir / after_name,
            max_width=max_width,
            quality=quality,
        )
        if before_size != after_size:
            raise ValueError(
                f"proof dimensions do not match for {scan_id}: "
                f"{before_size} != {after_size}"
            )
        decision = decisions[scan_id]
        scan = {
            "id": scan_id,
            "before": f"images/{before_name}",
            "after": f"images/{after_name}",
            "action": decision["action"],
            "width": before_size[0],
            "height": before_size[1],
            "counts": result.get("counts", {}),
        }
        if decision.get("reason"):
            scan["reason"] = decision["reason"]
        scans.append(scan)

    flagged = sum(scan["action"] == "publish_flag" for scan in scans)
    manifest = {
        "schema_version": 1,
        "summary": {
            "total": len(scans),
            "clean": len(scans) - flagged,
            "flagged": flagged,
        },
        "scans": scans,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "manifest": str(manifest_path),
        "total": len(scans),
        "clean": len(scans) - flagged,
        "flagged": flagged,
    }
