from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageOps


def _video_duration(path: Path) -> float:
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
    finally:
        reader.close()
    duration = float(metadata.get("duration") or 0.0)
    if duration <= 0.0:
        raise ValueError(f"video has no usable duration: {path}")
    return duration


def _extract_frame(
    executable: str,
    video: Path,
    destination: Path,
    *,
    timestamp: float,
) -> None:
    subprocess.run(
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            "scale=640:-2",
            "-q:v",
            "2",
            str(destination),
        ],
        check=True,
    )
    if not destination.is_file():
        raise RuntimeError(f"ffmpeg did not create frame: {destination}")


def build_reference_contacts(
    photo_root: Path,
    output_root: Path,
    scan_ids: list[str],
) -> dict[str, object]:
    photo_root = photo_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    if not scan_ids or len(set(scan_ids)) != len(scan_ids):
        raise ValueError("scan IDs must be nonempty and unique")
    output_root.mkdir(parents=True, exist_ok=False)
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    results: list[dict[str, object]] = []

    for scan_id in scan_ids:
        if not scan_id or Path(scan_id).name != scan_id:
            raise ValueError(f"unsafe scan ID: {scan_id!r}")
        scan_root = photo_root / scan_id
        videos = sorted(scan_root.glob("*.mov"))
        if len(videos) != 1:
            raise ValueError(
                f"expected exactly one reference video for {scan_id}"
            )
        video = videos[0].resolve()
        try:
            video.relative_to(photo_root)
        except ValueError as error:
            raise ValueError(f"video lies outside photo root: {video}") from error

        destination = output_root / scan_id
        destination.mkdir()
        duration = _video_duration(video)
        frame_paths: list[Path] = []
        for index, fraction in enumerate(
            (0.08, 0.20, 0.32, 0.44, 0.56, 0.68, 0.80, 0.92),
            start=1,
        ):
            frame_path = destination / f"frame-{index:02d}.jpg"
            _extract_frame(
                executable,
                video,
                frame_path,
                timestamp=min(duration - 0.05, duration * fraction),
            )
            frame_paths.append(frame_path)

        frames = [Image.open(path).convert("RGB") for path in frame_paths]
        width = max(image.width for image in frames)
        height = max(image.height for image in frames)
        contact = Image.new(
            "RGB",
            (width * 4 + 40, height * 2 + 24),
            (20, 20, 20),
        )
        for index, frame in enumerate(frames):
            tile = ImageOps.pad(frame, (width, height), color=(0, 0, 0))
            x = 8 + (index % 4) * (width + 8)
            y = 8 + (index // 4) * (height + 8)
            contact.paste(tile, (x, y))
        contact_path = destination / "contact.jpg"
        contact.save(contact_path, quality=88)
        for frame in frames:
            frame.close()
        results.append(
            {
                "scan_id": scan_id,
                "source_video": str(video),
                "duration_seconds": duration,
                "contact": str(contact_path),
                "frames": [str(path) for path in frame_paths],
            }
        )

    report = {
        "schema_version": 1,
        "source_photos_opened_read_only": True,
        "photo_root": str(photo_root),
        "output_root": str(output_root),
        "results": results,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build read-only reference-video contact sheets."
    )
    parser.add_argument("photo_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--scan", action="append", required=True)
    args = parser.parse_args()
    report = build_reference_contacts(
        args.photo_root,
        args.output_root,
        args.scan,
    )
    print(json.dumps({"contacts": len(report["results"])}, indent=2))


if __name__ == "__main__":
    main()
