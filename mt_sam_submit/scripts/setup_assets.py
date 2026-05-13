#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


SUBMIT_ROOT = Path(__file__).resolve().parents[1]


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    src = src.resolve()
    if not src.exists():
        raise FileNotFoundError(f"source not found: {src}")
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        dst.symlink_to(src, target_is_directory=src.is_dir())


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare local SAM2/DINOv3 assets for mt_sam_submit.")
    parser.add_argument("--source-root", default=str(SUBMIT_ROOT.parent), help="Existing memory-sam repository root")
    parser.add_argument("--copy", action="store_true", help="Copy assets instead of creating symlinks")
    parser.add_argument("--sam-checkpoint", default="", help="Optional SAM2 Hiera-L checkpoint path")
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"source root not found: {source_root}")

    mappings = [
        (source_root / "dinov3", SUBMIT_ROOT / "third_party" / "dinov3"),
        (source_root / "sam2", SUBMIT_ROOT / "third_party" / "sam2"),
        (source_root / "configs", SUBMIT_ROOT / "configs"),
        (source_root / "dinov3_weights", SUBMIT_ROOT / "assets" / "dinov3_weights"),
    ]
    for src, dst in mappings:
        _link_or_copy(src, dst, copy=bool(args.copy))
        print(f"[OK] {'copied' if args.copy else 'linked'} {dst} -> {src}")

    checkpoint = Path(args.sam_checkpoint).expanduser() if args.sam_checkpoint else source_root / "MedSAM2" / "checkpoints" / "sam2.1_hiera_large.pt"
    if not checkpoint.exists():
        checkpoint = source_root / "checkpoints" / "sam2.1_hiera_large.pt"
    _link_or_copy(checkpoint, SUBMIT_ROOT / "checkpoints" / "sam2.1_hiera_large.pt", copy=bool(args.copy))
    print(f"[OK] {'copied' if args.copy else 'linked'} {SUBMIT_ROOT / 'checkpoints' / 'sam2.1_hiera_large.pt'} -> {checkpoint}")

    print("\nUse:")
    print("  --dinov3-repo mt_sam_submit/third_party/dinov3")
    print("  --dinov3-weights mt_sam_submit/assets/dinov3_weights")
    print("  --sam-checkpoint mt_sam_submit/checkpoints/sam2.1_hiera_large.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
