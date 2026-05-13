#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main() -> int:
    parser = argparse.ArgumentParser(description="Run Memory-SAM on a single image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out-mask", required=True)
    parser.add_argument("--out-meta", default="")
    parser.add_argument("--memory-dir", default="memory_m20")
    parser.add_argument("--sam-checkpoint", required=True)
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l")
    parser.add_argument("--dinov3-model", default="dinov3_vitl16")
    parser.add_argument("--dinov3-repo", default="third_party/dinov3")
    parser.add_argument("--dinov3-weights", default="assets/dinov3_weights")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from mt_sam import MTSAMConfig, MTSAMPredictor

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
    result = predictor.predict_file(args.image)
    predictor.save_mask(args.out_mask, result["mask"])
    if args.out_meta:
        meta = {
            "score": result["score"],
            "picked_item_id": result["picked_item_id"],
            "retrieved": result["retrieved"],
            "separability": result["separability"],
            "latency_ms": result["latency_ms"],
            "prompt_points": result["prompt"]["points"].tolist(),
            "prompt_labels": result["prompt"]["labels"].tolist(),
        }
        Path(args.out_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[OK] mask={args.out_mask}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
