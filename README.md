# Memory-SAM

This repository contains the user-facing release version of Memory-SAM:

> Memory-Augmented Retrieval-to-Prompt for Training-Free Tongue Segmentation

The repository keeps the paper-aligned Memory-SAM core and a practical Gradio UI. It does not ship with a prebuilt memory bank. Users create their own memory items from reference images and masks through the UI.

## What This Version Includes

- Fully automatic Memory-SAM inference after memory items are added.
- Empty user memory by default.
- English Gradio UI.
- Single-image segmentation.
- Batch segmentation for uploaded files or a folder path.
- Reference image/mask memory insertion.
- SAM2 point-prompt mask creation for building memory items.
- Memory preview, deletion, and clearing.
- Debug visualization for prompt points and FG/BG/contrast maps.
- Fixed mask cleanup to remove small disconnected spill regions and smooth boundaries.

The paper configuration is fixed in code as the default operating mode. Experimental configuration switches are intentionally not exposed in the UI.

## Method Summary

For each query image, Memory-SAM:

1. Resizes the image to `512x512`.
2. Extracts DINOv3 ViT/16 global and dense patch descriptors.
3. Retrieves labeled references from the user memory bank with FAISS cosine search.
4. Selects the reference with the strongest foreground/background separability.
5. Builds foreground and background similarity maps using the selected reference mask.
6. Builds the final contrast map `S(i)=s_fg(i)-s_bg(i)`.
7. Sends the highest-scoring foreground-only contrast points to SAM2.
8. Keeps the largest connected mask component, fills holes, and applies fixed morphology cleanup.

If no valid prompt candidates remain, the run reports an explicit error so the user can add better reference memory.

## Folder Layout

```text
memory-sam/
  mt_sam/
    features.py       # DINOv3 feature extraction
    memory.py         # FAISS memory bank and memory management
    metrics.py        # Evaluation metrics
    predictor.py      # End-to-end Memory-SAM and SAM2 point-mask helper
    prompting.py      # Paper-aligned prompt generation
  scripts/
    build_memory.py   # Optional CSV-based memory builder for evaluation
    download_assets.py# Public setup from official SAM2/DINOv3 sources
    infer_image.py    # Single-image CLI inference
    evaluate_split.py # CSV split evaluation
    setup_assets.py   # Copy or symlink local assets into this folder
  ui/
    app.py            # English Gradio UI
```

## Quick Start

```bash
git clone https://github.com/jw-chae/memory-sam.git
cd memory-sam

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/download_assets.py
```

## Public Dataset

SM-Tongue Public Original512 is available on Hugging Face:

- https://huggingface.co/datasets/Mark-CHAE/SM-Tongue-Public-Original512
- 2,334 de-identified 512x512 image/mask pairs
- filtered input images, binary tongue masks, regenerated overlays, metadata, and checksums

Download:

```bash
hf download Mark-CHAE/SM-Tongue-Public-Original512 \
  SM-Tongue-Public-Original512.tar.gz \
  SM-Tongue-Public-Original512.tar.gz.sha256 \
  --repo-type dataset
sha256sum -c SM-Tongue-Public-Original512.tar.gz.sha256
tar -xzf SM-Tongue-Public-Original512.tar.gz
```

`download_assets.py` downloads public SAM2.1 assets and clones the official SAM2/DINOv3 code repositories. DINOv3 weights require Meta approval, so the script prints the required next step when no DINOv3 weight URL or local weight file is provided.

After DINOv3 access is approved, rerun one of:

```bash
python scripts/download_assets.py \
  --skip-code \
  --dinov3-weight-url '<URL_FROM_META_APPROVAL_EMAIL>'
```

or:

```bash
python scripts/download_assets.py \
  --skip-code \
  --dinov3-weight-file /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Then launch:

```bash
python ui/app.py \
  --memory-dir user_memory \
  --results-dir results \
  --sam-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_l \
  --device cuda
```

## Assets And Licenses

After assets are prepared, the repository should contain:

```text
memory-sam/
  third_party/
    sam2/
    dinov3/
  assets/
    dinov3_weights/
  checkpoints/
    sam2.1_hiera_large.pt
  configs/
```

Do not commit these folders to GitHub. They are ignored by `.gitignore`:

```text
assets/dinov3_weights/
checkpoints/
third_party/
configs/
user_memory/
results/
```

### SAM2.1

Official sources:

- Code: `https://github.com/facebookresearch/sam2`
- Project page: `https://ai.meta.com/research/sam2/`
- SAM2.1 Hiera-L checkpoint: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt`
- Hugging Face model card: `https://huggingface.co/facebook/sam2.1-hiera-large`

The helper script downloads `sam2.1_hiera_large.pt` into:

```text
checkpoints/sam2.1_hiera_large.pt
```

and clones SAM2 code into:

```text
third_party/sam2/
```

The official SAM2.1 model card lists Apache-2.0 licensing for the Hugging Face release. Check the upstream repository/model card before redistribution.

### DINOv3

Official sources:

- Code: `https://github.com/facebookresearch/dinov3`
- Project page and access request: `https://ai.meta.com/dinov3/`

DINOv3 weights are not a plain public direct-download dependency. The official DINOv3 README says users must request access; after approval, Meta sends an email containing the available weight URLs. Use that approved URL with:

```bash
python scripts/download_assets.py \
  --skip-code \
  --dinov3-weight-url '<URL_FROM_META_APPROVAL_EMAIL>'
```

If the weight is already downloaded locally:

```bash
python scripts/download_assets.py \
  --skip-code \
  --dinov3-weight-file /path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Expected local file:

```text
assets/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

DINOv3 code and model weights are released under the DINOv3 License. Users must follow Meta's DINOv3 access and license terms.

### Local Workstation Setup

On the original workstation, you can populate assets from the parent repository instead of downloading them again:

```bash
cd /media/jjack/Extreme\ SSD/paper_codes/memory-sam
python scripts/setup_assets.py --copy
```

Use symlinks instead of copies when the filesystem supports them:

```bash
python scripts/setup_assets.py
```

## Environment

On the current machine, use the existing conda environment:

```bash
source /home/jjack/miniconda3/etc/profile.d/conda.sh
conda activate medsam_env
cd /media/jjack/Extreme\ SSD/paper_codes/memory-sam
```

For a new machine, install the required runtime packages in your own environment:

```bash
pip install torch torchvision opencv-python pillow numpy faiss-cpu gradio tqdm
```

Use the CUDA-compatible `faiss-gpu`/PyTorch builds if GPU FAISS or CUDA inference is required.

## Launch The UI

```bash
python ui/app.py \
  --memory-dir user_memory \
  --results-dir results \
  --sam-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_l \
  --device cuda
```

Open the local Gradio URL printed in the terminal.

## UI Workflow

### 1. Build Memory

Use one of the two memory-building paths:

- `Build Memory`: upload a reference tongue image and a validated binary mask, then click `Add Reference To Memory`.
- `Point Mask Tool`: upload an image, click foreground/background points, click `Generate Mask`, then click `Save Generated Mask To Memory`.

The memory bank starts empty. Memory-SAM uses only the memory items that the user adds.

### 2. Segment One Image

1. Open the `Segment` tab.
2. Upload a query image.
3. Optionally set a results directory.
4. Click `Run Memory-SAM`.

Outputs include overlay, binary mask, prompt-point visualization, and optional debug heatmaps.

### 3. Segment A Batch

1. Open the `Batch Segment` tab.
2. Upload multiple images or enter a folder path.
3. Set a results directory.
4. Click `Run Batch`.

The UI writes masks, overlays, per-image metadata, and `batch_summary.json`.

### 4. Manage Memory

Open `Memory Manager` to:

- refresh the memory list,
- preview a memory item,
- delete a selected item,
- clear all memory after typing `CLEAR`.

## Optional CLI Usage

Single-image inference:

```bash
python scripts/infer_image.py \
  --image /abs/path/query.png \
  --memory-dir user_memory \
  --sam-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_l \
  --out-mask results/query_mask.png \
  --out-meta results/query_meta.json \
  --device cuda
```

Evaluation on a CSV split:

```bash
python scripts/evaluate_split.py \
  --manifest /abs/path/test.csv \
  --memory-dir user_memory \
  --sam-checkpoint checkpoints/sam2.1_hiera_large.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_l \
  --out-dir results/eval \
  --device cuda
```

CSV format:

```csv
image_path,mask_path
/abs/path/image_001.png,/abs/path/mask_001.png
/abs/path/image_002.png,/abs/path/mask_002.png
```

`evaluate_split.py` reports:

```text
mIoU, mPA, Acc, Precision, Recall, Dice, IoU_fg, IoU_bg, latency_ms
```

## Notes For Reviewers And Users

- This is a training-free inference package.
- The UI does not expose experimental configuration controls.
- The memory bank is user-owned and starts empty.
- Test images should not be inserted into memory when reproducing paper-style evaluation.
- Better memory quality directly improves retrieval-to-prompt quality.
