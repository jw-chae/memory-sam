#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


SUBMIT_ROOT = Path(__file__).resolve().parents[1]
SAM2_REPO_URL = "https://github.com/facebookresearch/sam2.git"
DINOV3_REPO_URL = "https://github.com/facebookresearch/dinov3.git"
SAM21_HIERA_L_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
DINOV3_VITL16_FILENAME = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[SKIP] exists: {dst}")
        return
    print(f"[DOWNLOAD] {url}")
    print(f"           -> {dst}")
    urllib.request.urlretrieve(url, str(dst))
    print(f"[OK] {dst}")


def clone_repo(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[SKIP] exists: {dst}")
        return
    run(["git", "clone", "--depth", "1", url, str(dst)])


def copy_or_link(src: Path, dst: Path, symlink: bool) -> None:
    src = src.expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        print(f"[SKIP] exists: {dst}")
        return
    if symlink:
        dst.symlink_to(src, target_is_directory=src.is_dir())
        print(f"[OK] linked {dst} -> {src}")
    else:
        shutil.copy2(src, dst)
        print(f"[OK] copied {dst} <- {src}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare public Memory-SAM assets from official SAM2/DINOv3 sources."
    )
    parser.add_argument("--skip-code", action="store_true", help="Do not clone SAM2/DINOv3 code repositories")
    parser.add_argument("--sam2-repo-url", default=SAM2_REPO_URL)
    parser.add_argument("--dinov3-repo-url", default=DINOV3_REPO_URL)
    parser.add_argument("--sam-checkpoint-url", default=SAM21_HIERA_L_URL)
    parser.add_argument(
        "--dinov3-weight-url",
        default="",
        help="DINOv3 ViT-L/16 weight URL from Meta's approved access email",
    )
    parser.add_argument(
        "--dinov3-weight-file",
        default="",
        help="Existing local DINOv3 ViT-L/16 .pth file to copy or symlink",
    )
    parser.add_argument("--symlink-dinov3-weight", action="store_true", help="Symlink local DINOv3 weight file")
    args = parser.parse_args()

    if not args.skip_code:
        clone_repo(args.sam2_repo_url, SUBMIT_ROOT / "third_party" / "sam2")
        clone_repo(args.dinov3_repo_url, SUBMIT_ROOT / "third_party" / "dinov3")

    download(args.sam_checkpoint_url, SUBMIT_ROOT / "checkpoints" / "sam2.1_hiera_large.pt")

    dinov3_dst = SUBMIT_ROOT / "assets" / "dinov3_weights" / DINOV3_VITL16_FILENAME
    if args.dinov3_weight_file:
        copy_or_link(Path(args.dinov3_weight_file), dinov3_dst, symlink=bool(args.symlink_dinov3_weight))
    elif args.dinov3_weight_url:
        download(args.dinov3_weight_url, dinov3_dst)
    else:
        print("\n[DINOv3 ACTION REQUIRED]")
        print("DINOv3 weights are gated by Meta. Request access, then rerun with one of:")
        print(f"  --dinov3-weight-url '<URL_FROM_META_EMAIL>'")
        print(f"  --dinov3-weight-file /path/to/{DINOV3_VITL16_FILENAME}")
        print(f"\nExpected destination:\n  {dinov3_dst}")

    print("\n[DEFAULT RUNTIME PATHS]")
    print("  --dinov3-repo third_party/dinov3")
    print("  --dinov3-weights assets/dinov3_weights")
    print("  --sam-checkpoint checkpoints/sam2.1_hiera_large.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
