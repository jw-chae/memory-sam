#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = lambda x, **_: x

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt_sam.metrics import binary_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Memory-SAM on a CSV split.")
    parser.add_argument("--manifest", required=True, help="CSV with image_path,mask_path columns")
    parser.add_argument("--memory-dir", default="memory_m20")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l")
    parser.add_argument("--dinov3-model", default="dinov3_vitl16")
    parser.add_argument("--dinov3-repo", default="third_party/dinov3")
    parser.add_argument("--dinov3-weights", default="assets/dinov3_weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from mt_sam import MTSAMConfig, MTSAMPredictor

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    predictor = MTSAMPredictor(
        MTSAMConfig(
            sam_checkpoint=args.sam_checkpoint,
            sam_config=args.sam_config,
            memory_dir=args.memory_dir,
            dinov3_model=args.dinov3_model,
            dinov3_repo=args.dinov3_repo,
            dinov3_weights=args.dinov3_weights,
            device=args.device,
        )
    )

    with open(args.manifest, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    per_image = []
    for idx, row in enumerate(tqdm(rows, desc="Memory-SAM")):
        result = predictor.predict_file(row["image_path"])
        mask_path = row.get("mask_path") or row.get("gt_mask_path")
        if not mask_path:
            raise KeyError("manifest must contain mask_path or gt_mask_path")
        gt = np.array(Image.open(mask_path).convert("L")) > 0
        gt = cv2.resize(gt.astype(np.uint8), result["mask"].shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
        metrics = binary_metrics(result["mask"], gt)
        mask_path = pred_dir / f"{idx:05d}_{Path(row['image_path']).stem}.png"
        predictor.save_mask(str(mask_path), result["mask"])
        per_image.append(
            {
                "image_path": row["image_path"],
                "mask_path": mask_path,
                "pred_mask_path": str(mask_path),
                "picked_item_id": result["picked_item_id"],
                "latency_ms": result["latency_ms"],
                **metrics,
            }
        )

    keys = ["mIoU", "mPA", "Acc", "Precision", "Recall", "Dice", "IoU_fg", "IoU_bg", "latency_ms"]
    summary = {key: float(np.mean([float(row[key]) for row in per_image])) for key in keys}
    summary["n"] = len(per_image)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image[0].keys()))
        writer.writeheader()
        writer.writerows(per_image)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
