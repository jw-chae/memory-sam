# Memory-SAM

**MICCAI 2026 accepted release package** for:

> Memory-SAM: Memory-Augmented Retrieval-to-Prompt for Training-Free Tongue Segmentation

Memory-SAM is a human-prompt-free, training-free retrieval-to-prompt framework for automatic tongue segmentation with SAM2. Here, **training-free** means that no task-specific parameter optimization or fine-tuning is performed at deployment time. Users provide a small labeled memory bank, and Memory-SAM retrieves references from that memory to generate SAM2 point prompts automatically.

This repository is the cleaned user-facing release package. It is designed to run independently from the original research workspace after assets are prepared.

![Memory-SAM architecture](figures/figure_architecture.png)

## What Is Included

- Paper-aligned Memory-SAM inference pipeline.
- English Gradio UI for practical use.
- Empty user memory by default.
- Single-image and batch segmentation.
- Reference image/mask memory management.
- SAM2 point-mask tool with live mask preview.
- Memory table, reference image preview, and reference overlay preview.
- Fixed post-processing to remove small disconnected spill regions and smooth masks.
- CLI scripts for inference, evaluation, asset setup, and dataset export.

The UI intentionally exposes the best/default paper setting only. Ablation switches are not exposed to end users.

## Public Links

- Code: https://github.com/jw-chae/memory-sam
- Dataset: https://huggingface.co/datasets/Mark-CHAE/SM-Tongue-Public-Original512

## Public Dataset

The public dataset release is **SM-Tongue Public Original512**. The dataset has been updated to **2,334** image/mask pairs for public release.

- Updated public release size: **2,334** image/mask pairs.
- Paper benchmark size: **2,155** SM-Tongue images used in the paper experiments.
- Images are filtered source images resized to `512x512`.
- Masks are binary tongue masks resized to `512x512`.
- Overlays, metadata, and SHA256 checksums are included.

Download with Hugging Face CLI:

```bash
pip install -U huggingface_hub
hf download Mark-CHAE/SM-Tongue-Public-Original512 \
  SM-Tongue-Public-Original512.tar.gz \
  SM-Tongue-Public-Original512.tar.gz.sha256 \
  --repo-type dataset
sha256sum -c SM-Tongue-Public-Original512.tar.gz.sha256
tar -xzf SM-Tongue-Public-Original512.tar.gz
```

## Folder Layout

```text
memory-sam/
  README.md
  requirements.txt
  figures/
    figure_architecture.png
  mt_sam/
    features.py       # DINOv3 feature extraction
    memory.py         # FAISS memory bank and memory manager
    metrics.py        # Evaluation metrics
    paths.py          # Package-local path resolution
    predictor.py      # End-to-end Memory-SAM predictor
    prompting.py      # Retrieval-to-prompt generation
  scripts/
    build_memory.py
    download_assets.py
    evaluate_split.py
    export_public_dataset.py
    infer_image.py
    setup_assets.py
  ui/
    app.py            # Gradio UI
  assets/             # DINOv3 weights, not committed
  checkpoints/        # SAM2 checkpoint, not committed
  configs/            # SAM2 configs, not committed
  third_party/        # SAM2/DINOv3 source repos, not committed
  user_memory/        # User-created memory bank, not committed
  results/            # Output masks/overlays/metadata, not committed
```

## Environment

On this workstation, use the existing conda environment:

```bash
source /home/jjack/miniconda3/etc/profile.d/conda.sh
conda activate medsam_env
cd /media/jjack/Extreme\ SSD/paper_codes/memory-sam
```

Important: this machine does not provide a `python` command. Use `python3` globally or use `python` only after activating the conda environment.

For a new machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

If using CUDA, install the PyTorch build that matches your CUDA driver before running Memory-SAM.

## Asset Setup

Memory-SAM requires SAM2.1 and DINOv3 assets. These are not committed to GitHub.

Expected local structure:

```text
memory-sam/
  checkpoints/sam2.1_hiera_large.pt
  configs/sam2.1/sam2.1_hiera_l.yaml
  third_party/sam2/
  third_party/dinov3/
  assets/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

On this workstation, populate assets from the parent research workspace:

```bash
python scripts/setup_assets.py
```

Use copies instead of symlinks if needed:

```bash
python scripts/setup_assets.py --copy
```

For a clean external machine, download public assets:

```bash
python scripts/download_assets.py
```

SAM2.1 Hiera-L is downloaded automatically from Meta's public URL. DINOv3 weights require Meta approval. After approval, use either the approved URL:

```bash
python scripts/download_assets.py \
  --skip-code \
  --dinov3-weight-url '<URL_FROM_META_APPROVAL_EMAIL>'
```

or a local downloaded file:

```bash
python scripts/download_assets.py \
  --skip-code \
  --dinov3-weight-file /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Official upstream sources:

- SAM2 code: https://github.com/facebookresearch/sam2
- SAM2.1 Hiera-L checkpoint: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
- DINOv3 code/access: https://github.com/facebookresearch/dinov3 and https://ai.meta.com/dinov3/

Users must follow the upstream SAM2 and DINOv3 licenses and access terms.

## Launch UI

Recommended command on this workstation:

```bash
source /home/jjack/miniconda3/etc/profile.d/conda.sh
conda activate medsam_env
cd /media/jjack/Extreme\ SSD/paper_codes/memory-sam
python ui/app.py \
  --device cuda
```

Equivalent explicit command:

```bash
python ui/app.py \
  --memory-dir user_memory \
  --results-dir results \
  --sam-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_l \
  --dinov3-repo third_party/dinov3 \
  --dinov3-weights assets/dinov3_weights \
  --device cuda
```

If running directly inside the Memory-SAM package folder:

```bash
cd /media/jjack/Extreme\ SSD/paper_codes/memory-sam
python ui/app.py --device cuda
```

Open the local Gradio URL printed in the terminal, usually:

```text
http://127.0.0.1:7860
```

## UI Workflow

### 1. Build Memory

The memory bank starts empty. Add user-owned reference examples before segmentation.

Options:

- `Build Memory`: upload a reference tongue image and its validated binary mask, then click `Add Reference To Memory`.
- `Point Mask Tool`: upload an image, click foreground/background points, inspect the live SAM2 mask preview, then click `Save Generated Mask To Memory`.

### 2. Segment One Image

1. Open the `Segment` tab.
2. Upload a query image.
3. Click `Run Memory-SAM`.
4. Inspect the overlay, binary mask, prompt-point visualization, and debug similarity maps.

### 3. Segment A Batch

1. Open the `Batch Segment` tab.
2. Upload multiple images or enter a folder path.
3. Set a results directory.
4. Click `Run Batch`.

The UI writes masks, overlays, per-image metadata, and `batch_summary.json`.

### 4. Manage Memory

Open `Memory Manager` to:

- refresh the memory table,
- preview the reference image,
- preview the reference overlay,
- delete a selected item,
- clear all memory after typing `CLEAR`.

## CLI Usage

Single-image inference:

```bash
python scripts/infer_image.py \
  --image /abs/path/query.png \
  --out-mask results/query_mask.png \
  --out-meta results/query_meta.json \
  --device cuda
```

Build memory from a CSV manifest:

```bash
python scripts/build_memory.py \
  --manifest /abs/path/train_memory.csv \
  --memory-dir user_memory \
  --max-items 20 \
  --device cuda
```

CSV format:

```csv
image_path,mask_path
/abs/path/image_001.png,/abs/path/mask_001.png
/abs/path/image_002.png,/abs/path/mask_002.png
```

Evaluate a split:

```bash
python scripts/evaluate_split.py \
  --manifest /abs/path/test.csv \
  --memory-dir user_memory \
  --out-dir results/eval \
  --device cuda
```

Metrics reported:

```text
mIoU, mPA, Acc, Precision, Recall, Dice, IoU_fg, IoU_bg, latency_ms
```

## Method Summary

For each query image, Memory-SAM:

1. Resizes the image to `512x512`.
2. Extracts DINOv3 ViT/16 global and dense patch descriptors.
3. Retrieves top references from the user memory bank with FAISS cosine search.
4. Reranks retrieved references by foreground/background separability.
5. Builds foreground/background similarity maps from the selected reference mask.
6. Builds the contrast map `S(i)=s_fg(i)-s_bg(i)`.
7. Selects the highest-scoring foreground-only contrast points.
8. Runs SAM2 with those point prompts.
9. Keeps the largest connected component, fills holes, and applies fixed morphology cleanup.

## Reproducibility Notes

- This package performs no training or fine-tuning.
- The UI does not expose ablation settings.
- The memory bank is user-created and starts empty.
- For paper-style evaluation, never insert test images into memory.
- Memory quality directly controls retrieval-to-prompt quality.
- DINOv3 weights are gated by Meta and must be obtained by each user under the official access terms.
