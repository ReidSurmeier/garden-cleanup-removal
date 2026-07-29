from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote


VIEWS = ("front", "side", "top")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _relative_url(path: str, batch_root: Path) -> str:
    render = Path(path).resolve()
    if not render.is_file():
        raise FileNotFoundError(render)
    if not render.is_relative_to(batch_root):
        raise ValueError(f"render escapes batch root: {render}")
    return quote(render.relative_to(batch_root).as_posix(), safe="/")


def _candidate_record(
    value: dict[str, Any],
    batch_root: Path,
) -> dict[str, Any]:
    renders = value.get("renders")
    if not isinstance(renders, dict):
        raise ValueError("candidate lacks renders")
    return {
        "id": str(value["candidate"]),
        "rotation_degrees": float(value["rotation_degrees"]),
        "selection_basis": str(value.get("selection_basis", "")),
        "renders": {
            view: _relative_url(str(renders[view]), batch_root)
            for view in VIEWS
        },
    }


def _scan_record(
    report_path: Path,
    batch_root: Path,
) -> dict[str, Any]:
    report = _read(report_path)
    candidates = [
        _candidate_record(item, batch_root)
        for item in report.get("candidates", [])
    ]
    if not candidates or candidates[0]["id"] != "identity":
        raise ValueError(
            f"first candidate must preserve identity: {report_path}"
        )
    return {
        "scan_id": str(report["scan_id"]),
        "candidates": candidates,
    }


def _option(candidate: dict[str, Any]) -> str:
    label = (
        "Identity (preserved)"
        if candidate["id"] == "identity"
        else (
            f"Candidate {candidate['id']} "
            f"({candidate['rotation_degrees']:.1f} degrees)"
        )
    )
    attributes = " ".join(
        f'data-{view}="{html.escape(candidate["renders"][view], quote=True)}"'
        for view in VIEWS
    )
    return (
        f'<option value="{html.escape(candidate["id"], quote=True)}" '
        f'{attributes}>{html.escape(label)}</option>'
    )


def _card(scan: dict[str, Any]) -> str:
    scan_id = html.escape(scan["scan_id"])
    identity = scan["candidates"][0]
    alternatives = scan["candidates"][1:]
    selected = alternatives[0] if alternatives else identity
    options = "\n".join(_option(item) for item in scan["candidates"])
    buttons = "\n".join(
        (
            f'<button type="button" data-view="{view}"'
            f'{" class=\"active\"" if view == "front" else ""}>'
            f"{view.title()}</button>"
        )
        for view in VIEWS
    )
    return f"""
    <article class="scan" data-search="{html.escape(scan["scan_id"].lower())}">
      <header>
        <div>
          <h2>{scan_id}</h2>
          <p>{len(alternatives)} structural alternative(s); identity is the safe default.</p>
        </div>
        <div class="controls">
          <label>Right panel
            <select>{options}</select>
          </label>
          <div class="views">{buttons}</div>
        </div>
      </header>
      <div class="comparison">
        <figure>
          <a href="{identity["renders"]["front"]}" target="_blank">
            <img class="identity" src="{identity["renders"]["front"]}"
              loading="lazy" alt="{scan_id} identity">
          </a>
          <figcaption>Identity — preserved source orientation</figcaption>
        </figure>
        <figure>
          <a href="{selected["renders"]["front"]}" target="_blank">
            <img class="candidate" src="{selected["renders"]["front"]}"
              loading="lazy" alt="{scan_id} candidate">
          </a>
          <figcaption class="candidate-label">
            {html.escape("Identity" if selected["id"] == "identity" else f"Candidate {selected['id']}")}
          </figcaption>
        </figure>
      </div>
    </article>
    """


def _document(scans: list[dict[str, Any]]) -> str:
    cards = "\n".join(_card(scan) for scan in scans)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SF garden orientation review</title>
  <style>
    :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #0f1216; color: #eef2f6; }}
    .masthead {{ position: sticky; top: 0; z-index: 4; padding: 18px 24px;
      background: rgba(15,18,22,.96); border-bottom: 1px solid #303741; }}
    .masthead h1 {{ margin: 0 0 6px; font-size: 22px; }}
    .masthead p {{ margin: 0 0 12px; color: #aeb8c4; }}
    #search {{ width: min(520px, 100%); padding: 10px 12px; border-radius: 7px;
      border: 1px solid #46515f; background: #1b2027; color: inherit; }}
    main {{ width: min(1600px, 100%); margin: 0 auto; padding: 18px; }}
    .scan {{ margin: 0 0 22px; border: 1px solid #303741; border-radius: 10px;
      overflow: hidden; background: #171b21; }}
    .scan header {{ display: flex; gap: 18px; justify-content: space-between;
      align-items: end; padding: 14px 16px; }}
    h2 {{ margin: 0; font-size: 18px; }}
    header p {{ margin: 5px 0 0; color: #aeb8c4; font-size: 13px; }}
    .controls {{ display: flex; gap: 12px; align-items: end; flex-wrap: wrap; }}
    label {{ color: #aeb8c4; font-size: 12px; }}
    select, button {{ border: 1px solid #46515f; background: #222933;
      color: #eef2f6; border-radius: 6px; padding: 8px 10px; }}
    button.active {{ background: #2d6cdf; border-color: #5d91ef; }}
    .comparison {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1px;
      background: #303741; }}
    figure {{ margin: 0; min-width: 0; background: #111419; }}
    img {{ display: block; width: 100%; height: auto; min-height: 260px;
      object-fit: contain; background: #20242a; }}
    figcaption {{ padding: 9px 12px; color: #bdc7d2; font-size: 13px; }}
    @media (max-width: 800px) {{
      .scan header {{ align-items: start; flex-direction: column; }}
      .comparison {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="masthead">
    <h1>SF garden orientation review — {len(scans)} scans</h1>
    <p>Identity is never replaced automatically. Compare fixed front, side, and top views; click an image for the full-resolution PNG.</p>
    <input id="search" type="search" placeholder="Filter by scan date or time">
  </div>
  <main>{cards}</main>
  <script>
    const views = ["front", "side", "top"];
    document.querySelectorAll(".scan").forEach(card => {{
      let view = "front";
      const select = card.querySelector("select");
      const identity = card.querySelector("img.identity");
      const candidate = card.querySelector("img.candidate");
      const candidateLabel = card.querySelector(".candidate-label");
      function option() {{ return select.options[select.selectedIndex]; }}
      function update() {{
        const chosen = option();
        const identityOption = select.options[0];
        identity.src = identityOption.dataset[view];
        identity.closest("a").href = identity.src;
        candidate.src = chosen.dataset[view];
        candidate.closest("a").href = candidate.src;
        candidateLabel.textContent = chosen.textContent;
      }}
      select.addEventListener("change", update);
      card.querySelectorAll("[data-view]").forEach(button => {{
        button.addEventListener("click", () => {{
          view = button.dataset.view;
          card.querySelectorAll("[data-view]").forEach(item =>
            item.classList.toggle("active", item === button));
          update();
        }});
      }});
    }});
    document.querySelector("#search").addEventListener("input", event => {{
      const query = event.target.value.trim().toLowerCase();
      document.querySelectorAll(".scan").forEach(card => {{
        card.hidden = !card.dataset.search.includes(query);
      }});
    }});
  </script>
</body>
</html>
"""


def build_orientation_review_site(
    batch_root: Path,
    output_path: Path,
) -> dict[str, int]:
    batch_root = batch_root.resolve()
    output_path = output_path.resolve()
    if not batch_root.is_dir():
        raise FileNotFoundError(batch_root)
    if output_path.parent != batch_root or output_path.name != "viewer.html":
        raise ValueError("review site must be BATCH_ROOT/viewer.html")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite review site: {output_path}")
    report_paths = sorted(
        batch_root.glob("cohort-*/**/evaluation-report.json")
    )
    scans = sorted(
        (
            _scan_record(report_path, batch_root)
            for report_path in report_paths
        ),
        key=lambda item: item["scan_id"],
    )
    if not scans:
        raise ValueError("no Blender evaluation reports found")
    output_path.write_text(_document(scans), encoding="utf-8")
    return {
        "scan_count": len(scans),
        "candidate_count": sum(len(scan["candidates"]) for scan in scans),
    }
