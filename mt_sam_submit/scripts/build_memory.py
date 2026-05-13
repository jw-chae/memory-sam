#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Memory-SAM labeled memory bank.")
    parser.add_argument("--manifest", required=True, help="CSV with image_path,mask_path columns")
    parser.add_argument("--memory-dir", required=True)
    parser.add_argument("--max-items", type=int, default=20)
    parser.add_argument("--dinov3-model", default="dinov3_vitl16")
    parser.add_argument("--dinov3-repo", default="third_party/dinov3")
    parser.add_argument("--dinov3-weights", default="assets/dinov3_weights")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from mt_sam.features import DINOv3FeatureExtractor
    from mt_sam.memory import MemoryBank
    from mt_sam.predictor import resize_square

    extractor = DINOv3FeatureExtractor(args.dinov3_model, args.dinov3_repo, args.dinov3_weights, device=args.device)
    memory = MemoryBank(args.memory_dir)
    with open(args.manifest, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows[: max(1, int(args.max_items))]:
        mask_path = row.get("mask_path") or row.get("gt_mask_path")
        if not mask_path:
            raise KeyError("manifest must contain mask_path or gt_mask_path")
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        image = resize_square(image, 512)
        mask = resize_square(mask, 512)
        patch_features, grid = extractor.extract_patch_features(image)
        global_features = extractor.extract_global_features(image)
        item_id = memory.add(image, mask, global_features, patch_features, grid)
        print(f"[OK] added item_id={item_id} image={row['image_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
