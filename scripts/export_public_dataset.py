#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


INPUT_SUFFIX = "__input.png"
MASK_SUFFIX = "__mask_morphed.png"
OVERLAY_SUFFIX = "__overlay_morphed.jpg"
IMAGE_EXTENSIONS = (INPUT_SUFFIX, MASK_SUFFIX, OVERLAY_SUFFIX)


@dataclass(frozen=True)
class Sample:
    source_id: str
    image_path: Path
    mask_path: Path
    overlay_path: Path


def collect_samples(source_dir: Path) -> tuple[list[Sample], dict[str, list[str]]]:
    grouped: dict[str, dict[str, Path]] = {}
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        for suffix in IMAGE_EXTENSIONS:
            if name.endswith(suffix):
                source_id = name[: -len(suffix)]
                grouped.setdefault(source_id, {})[suffix] = path
                break

    complete: list[Sample] = []
    incomplete: dict[str, list[str]] = {}
    for source_id, files in sorted(grouped.items()):
        missing = [suffix for suffix in IMAGE_EXTENSIONS if suffix not in files]
        if missing:
            incomplete[source_id] = missing
            continue
        complete.append(
            Sample(
                source_id=source_id,
                image_path=files[INPUT_SUFFIX],
                mask_path=files[MASK_SUFFIX],
                overlay_path=files[OVERLAY_SUFFIX],
            )
        )
    return complete, incomplete


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"failed to read mask: {path}")
    return mask > 0


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return binary.astype(bool)
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    return labels == largest


def bbox_from_mask(mask: np.ndarray, margin_ratio: float, min_margin: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("empty mask")
    height, width = mask.shape[:2]
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    box_w = x1 - x0
    box_h = y1 - y0
    margin = max(int(round(max(box_w, box_h) * float(margin_ratio))), int(min_margin))
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(width, x1 + margin)
    y1 = min(height, y1 + margin)
    return x0, y0, x1, y1


def anonymize_crop(
    image: np.ndarray,
    mask: np.ndarray,
    margin_ratio: float,
    min_margin: int,
    visible_context_dilate: int,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    mask = largest_component(mask)
    x0, y0, x1, y1 = bbox_from_mask(mask, margin_ratio=margin_ratio, min_margin=min_margin)
    crop_image = image[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1].copy()

    kernel_size = max(1, int(visible_context_dilate))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    visible = cv2.dilate(crop_mask.astype(np.uint8), kernel, iterations=1) > 0

    private_removed = np.zeros_like(crop_image)
    private_removed[visible] = crop_image[visible]

    if output_size > 0:
        private_removed = cv2.resize(private_removed, (output_size, output_size), interpolation=cv2.INTER_AREA)
        crop_mask = cv2.resize(crop_mask.astype(np.uint8), (output_size, output_size), interpolation=cv2.INTER_NEAREST) > 0

    return private_removed, crop_mask


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = image.copy()
    color = np.zeros_like(out)
    color[..., 1] = 255
    mask_b = mask > 0
    out[mask_b] = ((1.0 - alpha) * out[mask_b] + alpha * color[mask_b]).astype(np.uint8)
    return out


def write_rgb_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])


def write_mask_png(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (mask > 0).astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 6])


def write_rgb_jpg(path: Path, image: np.ndarray, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_release_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix != ".tar":
            yield path


def write_checksums(root: Path) -> None:
    checksum_path = root / "checksums_sha256.txt"
    rows = []
    for path in iter_release_files(root):
        if path.name == checksum_path.name:
            continue
        rows.append((sha256_file(path), path.relative_to(root).as_posix()))
    checksum_path.write_text("".join(f"{digest}  {rel}\n" for digest, rel in rows), encoding="utf-8")


def resize_original(image: np.ndarray, mask: np.ndarray, output_size: int) -> tuple[np.ndarray, np.ndarray]:
    if output_size <= 0:
        return image.copy(), mask.copy()
    clean_image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
    clean_mask = cv2.resize(mask.astype(np.uint8), (output_size, output_size), interpolation=cv2.INTER_NEAREST) > 0
    return clean_image, clean_mask


def write_readme(
    root: Path,
    dataset_name: str,
    sample_count: int,
    skipped_count: int,
    output_size: int,
    export_mode: str,
) -> None:
    if export_mode == "original_resize":
        image_description = "Full filtered input images resized to the release resolution"
        deid_steps = """- Original filenames are not included.
- EXIF and image metadata are stripped by re-encoding all images.
- Each sample is renamed to `sm_tongue_XXXXXX`.
- Images and masks are resized to the release resolution.
- Overlays are regenerated from the released images and masks.

This mode preserves the filtered input image content. Use it only when the source set has already removed identifiable facial regions according to the release protocol."""
    else:
        image_description = "De-identified tongue-context input crops"
        deid_steps = """- Original filenames are not included.
- EXIF and image metadata are stripped by re-encoding all images.
- Each sample is renamed to `sm_tongue_XXXXXX`.
- Images are cropped around the tongue mask.
- Pixels outside a dilated tongue-context region are set to black.
- Overlays are regenerated from the released de-identified images and masks."""

    text = f"""# {dataset_name}

This release contains de-identified tongue segmentation samples exported from a smartphone tongue-image collection.

## Contents

```text
images/      {image_description}
masks/       Binary tongue masks
overlays/    Visualization overlays generated from images/ and masks/
metadata.csv Public sample manifest
checksums_sha256.txt
```

## De-identification And Release Processing

{deid_steps}

The released images are intended for tongue segmentation research. They should not be used for re-identification or subject matching.

## Statistics

- Released samples: {sample_count}
- Skipped incomplete/invalid samples: {skipped_count}
- Image size: {output_size}x{output_size}
- Export mode: `{export_mode}`

## File Pairing

For sample `sm_tongue_000001`:

```text
images/sm_tongue_000001.png
masks/sm_tongue_000001.png
overlays/sm_tongue_000001.jpg
```

## License

Add the final dataset license before public release. For broad research reuse, consider `CC BY-NC 4.0` or a custom research-only data use agreement if consent/IRB terms restrict commercial reuse.

## Citation

If you use this dataset, cite the associated Memory-SAM paper.
"""
    (root / "README.md").write_text(text, encoding="utf-8")


def write_license_placeholder(root: Path) -> None:
    text = """Dataset license placeholder.

Before public release, replace this file with the final dataset license approved by the data owner, consent terms, and ethics/IRB requirements.

Recommended conservative default for human-subject image data:
Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) or a custom research-only data use agreement.
"""
    (root / "LICENSE").write_text(text, encoding="utf-8")


def make_archive(root: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists():
        archive_path.unlink()
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(root, arcname=root.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a de-identified public tongue segmentation dataset.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-name", default="SM-Tongue-Public")
    parser.add_argument(
        "--export-mode",
        choices=["tongue_context_crop", "original_resize"],
        default="tongue_context_crop",
        help="Public image export policy",
    )
    parser.add_argument("--margin-ratio", type=float, default=0.35)
    parser.add_argument("--min-margin", type=int, default=40)
    parser.add_argument("--visible-context-dilate", type=int, default=31)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--archive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    release_root = out_dir / args.dataset_name
    if release_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {release_root}")
        shutil.rmtree(release_root)
    release_root.mkdir(parents=True, exist_ok=True)

    samples, incomplete = collect_samples(source_dir)
    rows = []
    skipped = []
    for index, sample in enumerate(samples, start=1):
        public_id = f"sm_tongue_{index:06d}"
        try:
            image = read_rgb(sample.image_path)
            mask = read_mask(sample.mask_path)
            if args.export_mode == "original_resize":
                clean_image, clean_mask = resize_original(image=image, mask=mask, output_size=args.output_size)
            else:
                clean_image, clean_mask = anonymize_crop(
                    image=image,
                    mask=mask,
                    margin_ratio=args.margin_ratio,
                    min_margin=args.min_margin,
                    visible_context_dilate=args.visible_context_dilate,
                    output_size=args.output_size,
                )
            clean_overlay = overlay_mask(clean_image, clean_mask)
            image_rel = Path("images") / f"{public_id}.png"
            mask_rel = Path("masks") / f"{public_id}.png"
            overlay_rel = Path("overlays") / f"{public_id}.jpg"
            write_rgb_png(release_root / image_rel, clean_image)
            write_mask_png(release_root / mask_rel, clean_mask)
            write_rgb_jpg(release_root / overlay_rel, clean_overlay)
            rows.append(
                {
                    "id": public_id,
                    "image_path": image_rel.as_posix(),
                    "mask_path": mask_rel.as_posix(),
                    "overlay_path": overlay_rel.as_posix(),
                    "height": args.output_size,
                    "width": args.output_size,
                }
            )
        except Exception as exc:
            skipped.append({"source_index": index, "reason": str(exc)})

    manifest_path = release_root / "metadata.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image_path", "mask_path", "overlay_path", "height", "width"])
        writer.writeheader()
        writer.writerows(rows)

    private_report = {
        "source_dir": str(source_dir),
        "released_samples": len(rows),
        "incomplete_source_sets": len(incomplete),
        "invalid_or_failed_samples": len(skipped),
        "incomplete_source_ids_not_released": list(incomplete.keys()),
        "failed_samples_not_released": skipped,
        "deidentification": {
            "renamed": True,
            "exif_stripped": True,
            "export_mode": args.export_mode,
            "mask_bbox_crop": args.export_mode == "tongue_context_crop",
            "outside_dilated_tongue_context_black": args.export_mode == "tongue_context_crop",
            "original_content_preserved": args.export_mode == "original_resize",
            "margin_ratio": args.margin_ratio,
            "min_margin": args.min_margin,
            "visible_context_dilate": args.visible_context_dilate,
            "output_size": args.output_size,
        },
    }
    private_report_path = out_dir / f"{args.dataset_name}_private_report.json"
    private_report_path.write_text(json.dumps(private_report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(
        root=release_root,
        dataset_name=args.dataset_name,
        sample_count=len(rows),
        skipped_count=len(incomplete) + len(skipped),
        output_size=args.output_size,
        export_mode=args.export_mode,
    )
    write_license_placeholder(release_root)
    write_checksums(release_root)

    archive_path = None
    if args.archive:
        archive_path = out_dir / f"{args.dataset_name}.tar.gz"
        make_archive(release_root, archive_path)
        (out_dir / f"{archive_path.name}.sha256").write_text(
            f"{sha256_file(archive_path)}  {archive_path.name}\n",
            encoding="utf-8",
        )

    print(json.dumps({"release_root": str(release_root), "samples": len(rows), "archive": str(archive_path) if archive_path else ""}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
