from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote

from PIL import Image, ImageDraw, ImageOps


LAYERS = (
    ("source", "Before"),
    ("plant", "Cleaned"),
    ("conservative", "Conservative"),
    ("rejected", "Removed"),
)


def _proof_path(scan_output: Path, layer: str) -> Path:
    return (
        scan_output
        / "final"
        / f"render-{layer}"
        / "front-rgb.png"
    )


def _build_contact_sheet(
    completed: list[dict[str, object]],
    destination: Path,
) -> None:
    cell_width = 320
    image_height = 240
    header_height = 42
    row_height = header_height + image_height
    sheet = Image.new(
        "RGB",
        (cell_width * len(LAYERS), row_height * len(completed)),
        (18, 19, 22),
    )
    draw = ImageDraw.Draw(sheet)
    for row, result in enumerate(completed):
        scan_id = str(result["scan_id"])
        scan_output = Path(str(result["output"]))
        y = row * row_height
        draw.text((12, y + 12), scan_id, fill=(244, 244, 239))
        for column, (layer, title) in enumerate(LAYERS):
            x = column * cell_width
            if column:
                draw.text((x + 12, y + 12), title, fill=(190, 194, 202))
            image_path = _proof_path(scan_output, layer)
            if not image_path.is_file():
                continue
            with Image.open(image_path) as source:
                image = ImageOps.contain(
                    source.convert("RGB"),
                    (cell_width, image_height),
                )
            offset = (
                x + (cell_width - image.width) // 2,
                y + header_height + (image_height - image.height) // 2,
            )
            sheet.paste(image, offset)
    sheet.save(destination, quality=92)


def build_batch_review(batch_report_path: Path) -> dict[str, object]:
    report_path = batch_report_path.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    output_root = Path(report["output_root"]).resolve()
    index_path = output_root / "index.html"
    contact_path = output_root / "batch-review.jpg"
    if index_path.exists() or contact_path.exists():
        raise FileExistsError("batch review already exists")

    completed = [
        result
        for result in report["results"]
        if result["status"] == "complete"
    ]
    _build_contact_sheet(completed, contact_path)

    sections: list[str] = []
    for result in report["results"]:
        scan_id = str(result["scan_id"])
        status = str(result["status"])
        encoded_id = quote(scan_id)
        cards: list[str] = []
        scan_output = Path(str(result["output"]))
        for layer, title in LAYERS:
            image_path = _proof_path(scan_output, layer)
            if image_path.is_file():
                relative = (
                    f"{encoded_id}/final/render-{layer}/front-rgb.png"
                )
                cards.append(
                    f'<figure><img src="{relative}" alt="{html.escape(title)}">'
                    f"<figcaption>{html.escape(title)}</figcaption></figure>"
                )
        viewer = f"{encoded_id}/review/viewer.html"
        counts = result.get("counts", {})
        count_text = html.escape(json.dumps(counts, sort_keys=True))
        sections.append(
            f'<section class="scan {html.escape(status)}">'
            f"<header><h2>{html.escape(scan_id)}</h2>"
            f'<span class="status">{html.escape(status)}</span>'
            f'<a href="{viewer}">Open full 3D view</a></header>'
            f'<div class="layers">{"".join(cards)}</div>'
            f"<details><summary>Point counts</summary><code>{count_text}</code>"
            f"</details></section>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Garden scan batch review</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, system-ui, sans-serif; }}
    body {{ margin: 0; background: #0d0e11; color: #f4f4ef; }}
    main {{ width: min(1600px, 96vw); margin: 32px auto 80px; }}
    h1 {{ margin-bottom: 4px; }}
    .summary {{ color: #afb4be; margin: 0 0 28px; }}
    .scan {{ border: 1px solid #30333a; border-radius: 14px; margin: 18px 0;
      overflow: hidden; background: #15171b; }}
    header {{ display: flex; gap: 18px; align-items: center; padding: 14px 18px; }}
    h2 {{ font-size: 17px; margin: 0; }}
    .status {{ color: #8bd3a8; }}
    header a {{ margin-left: auto; color: #92b8ff; }}
    .layers {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
      background: #30333a; }}
    figure {{ margin: 0; background: #0d0e11; }}
    img {{ width: 100%; aspect-ratio: 4 / 3; object-fit: contain; display: block; }}
    figcaption {{ padding: 7px 10px; color: #c7cad1; }}
    details {{ padding: 10px 18px 16px; color: #afb4be; }}
    code {{ white-space: pre-wrap; }}
    @media (max-width: 900px) {{ .layers {{ grid-template-columns: repeat(2, 1fr); }} }}
  </style>
</head>
<body><main>
  <h1>Garden scan cleanup review</h1>
  <p class="summary">{len(completed)} complete of {len(report["results"])} scans.
  Before, strict cleaned, conservative, and removed layers are shown together.</p>
  {"".join(sections)}
</main></body>
</html>
"""
    index_path.write_text(document, encoding="utf-8")
    return {
        "index": str(index_path),
        "contact_sheet": str(contact_path),
        "complete": len(completed),
        "total": len(report["results"]),
    }


def build_paginated_review(
    batch_report_path: Path,
    output_dir: Path,
    *,
    page_size: int = 20,
) -> dict[str, object]:
    if page_size < 1:
        raise ValueError("page_size must be positive")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"paginated review already exists: {output_dir}"
        )
    report = json.loads(
        batch_report_path.resolve().read_text(encoding="utf-8")
    )
    completed = [
        result
        for result in report["results"]
        if result["status"] == "complete"
    ]
    output_dir.mkdir(parents=True)
    pages: list[dict[str, object]] = []
    for offset in range(0, len(completed), page_size):
        page_results = completed[offset : offset + page_size]
        number = len(pages) + 1
        filename = f"page-{number:03d}.jpg"
        _build_contact_sheet(page_results, output_dir / filename)
        pages.append(
            {
                "number": number,
                "file": filename,
                "scan_ids": [
                    str(result["scan_id"]) for result in page_results
                ],
            }
        )
    figures = "".join(
        f'<figure><figcaption>Page {page["number"]}: '
        f'{html.escape(str(page["scan_ids"][0]))} through '
        f'{html.escape(str(page["scan_ids"][-1]))}</figcaption>'
        f'<a href="{page["file"]}"><img src="{page["file"]}" '
        f'alt="Review page {page["number"]}"></a></figure>'
        for page in pages
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paginated garden cleanup review</title>
<style>
body {{ margin: 0; background: #0d0e11; color: #f4f4ef;
font-family: Inter, system-ui, sans-serif; }}
main {{ width: min(1400px, 96vw); margin: 28px auto 80px; }}
figure {{ margin: 24px 0; }}
figcaption {{ margin-bottom: 8px; color: #c7cad1; }}
img {{ width: 100%; height: auto; display: block; }}
</style></head><body><main>
<h1>Garden cleanup proof pages</h1>
<p>{len(completed)} completed scans across {len(pages)} pages.</p>
{figures}
</main></body></html>
"""
    index_path = output_dir / "index.html"
    index_path.write_text(document, encoding="utf-8")
    return {
        "index": str(index_path),
        "page_count": len(pages),
        "complete": len(completed),
        "page_size": page_size,
        "pages": pages,
    }
