from __future__ import annotations

import argparse
import json
from pathlib import Path

from plant_cleanup.dense_semantic import HuggingFaceOneFormerPredictor
from plant_cleanup.source_photo_semantic import aggregate_source_photo_votes


MODEL = "shi-labs/oneformer_ade20k_swin_large"
PLANT_LABELS = (
    "plant",
    "tree",
    "grass",
    "flower",
    "palm, palm tree",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-cloud", required=True, type=Path)
    parser.add_argument("--native-cloud", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--photo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--camera-count", type=int, default=8)
    parser.add_argument("--maximum-dimension", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    predictor = HuggingFaceOneFormerPredictor(
        MODEL,
        device=args.device,
    )
    report = aggregate_source_photo_votes(
        args.canonical_cloud,
        args.native_cloud,
        args.inventory,
        args.photo_root,
        args.output,
        predictor=predictor,
        model_id=MODEL,
        plant_labels=PLANT_LABELS,
        camera_count=args.camera_count,
        maximum_dimension=args.maximum_dimension,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
