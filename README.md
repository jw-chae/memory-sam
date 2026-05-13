# Memory-SAM

**Memory-SAM: human-prompt-free retrieval-to-prompt tongue segmentation with SAM2.**

Memory-SAM is a training-free, memory-augmented segmentation framework for automatic tongue segmentation. It retrieves labeled exemplars from a memory bank using DINOv3 descriptors, converts mask-constrained feature correspondences into foreground-only SAM2 prompts, and provides a practical UI for memory construction and fully automatic inference.

## Abstract

Accurate tongue segmentation is a prerequisite for reliable tongue-image analysis in Traditional Chinese Medicine (TCM) and related screening workflows. While supervised segmentation networks can achieve high accuracy, they require substantial pixel-wise annotations and dataset-specific retraining. Recent foundation models such as SAM2 reduce the need for task-specific model development; however, they remain prompt-driven, creating a practical bottleneck for fully automatic deployment. We propose **Memory-SAM**, a human-prompt-free retrieval-to-prompt framework for automatic tongue segmentation with SAM2. Here, “training-free” means that the framework requires no parameter optimization or task-specific fine-tuning at deployment time, while relying on a labeled memory bank constructed from the training split. Memory-SAM retrieves exemplars from the memory bank using DINOv3 global descriptors, transfers mask-constrained correspondences to generate foreground/background point candidates, and applies contrastive foreground-only prompting for robust SAM2 inference. We evaluate Memory-SAM on the public HIT-Tongue dataset and **SM-Tongue**, a smartphone-captured in-the-wild tongue image benchmark with expert masks. The public SM-Tongue release currently contains **2,334** de-identified 512×512 image/mask pairs. Memory-SAM achieves competitive accuracy in controlled settings and substantially improves robustness over box-based SAM prompting in unconstrained scenes, while eliminating interactive prompting. We release SM-Tongue and the codebase to support reproducible research on automated tongue segmentation.

## Public Dataset

SM-Tongue Public Original512 is available on Hugging Face:

- Dataset: https://huggingface.co/datasets/Mark-CHAE/SM-Tongue-Public-Original512
- Samples: 2,334
- Image size: 512×512
- Contents: filtered input images, binary tongue masks, regenerated overlays, metadata, checksums

Download manually from Hugging Face, or with the Hugging Face Hub CLI:

```bash
hf download Mark-CHAE/SM-Tongue-Public-Original512 \
  SM-Tongue-Public-Original512.tar.gz \
  SM-Tongue-Public-Original512.tar.gz.sha256 \
  --repo-type dataset
sha256sum -c SM-Tongue-Public-Original512.tar.gz.sha256
tar -xzf SM-Tongue-Public-Original512.tar.gz
```

## Repository Layout

```text
memory-sam/
  mt_sam_submit/       # release-ready Memory-SAM package and UI
  README.md            # project overview
```

The current release-ready implementation is in `mt_sam_submit/`. The directory name is kept for compatibility with the working submission package, but all user-facing documentation and UI refer to the method as **Memory-SAM**.

## Quick Start

```bash
git clone https://github.com/jw-chae/memory-sam.git
cd memory-sam

python -m venv .venv
source .venv/bin/activate
pip install -r mt_sam_submit/requirements.txt

python mt_sam_submit/scripts/download_assets.py
```

DINOv3 weights require Meta approval. After approval, rerun:

```bash
python mt_sam_submit/scripts/download_assets.py \
  --skip-code \
  --dinov3-weight-url '<URL_FROM_META_APPROVAL_EMAIL>'
```

Launch the UI:

```bash
python mt_sam_submit/ui/app.py \
  --memory-dir mt_sam_submit/user_memory \
  --results-dir mt_sam_submit/results \
  --sam-checkpoint mt_sam_submit/checkpoints/sam2.1_hiera_large.pt \
  --sam-config configs/sam2.1/sam2.1_hiera_l \
  --device cuda
```

## How Memory-SAM Works

1. Resize the query image to 512×512.
2. Extract DINOv3 ViT-L/16 global and dense patch descriptors.
3. Retrieve top memory exemplars from a labeled memory bank using FAISS cosine search.
4. Select the exemplar with the strongest foreground/background separability.
5. Build foreground and background similarity maps from mask-constrained reference features.
6. Compute the contrast map `S(i)=s_fg(i)-s_bg(i)`.
7. Use the highest-scoring foreground-only contrast points as SAM2 prompts.
8. Apply fixed mask cleanup: largest component, hole filling, and morphology smoothing.

## UI Features

- Empty user memory by default.
- Add reference image/mask pairs through the UI.
- Create memory masks with live SAM2 point-click previews.
- Run single-image or batch Memory-SAM inference.
- Inspect retrieved references, prompt points, masks, overlays, and similarity maps.
- Manage memory items with preview, delete, and clear operations.

## Assets And Licenses

Memory-SAM uses external foundation-model assets:

- SAM2 code/checkpoints: https://github.com/facebookresearch/sam2
- SAM2.1 Hiera-L checkpoint: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
- DINOv3 code/access: https://github.com/facebookresearch/dinov3

DINOv3 weights require access approval from Meta. Follow the upstream DINOv3 license and access terms. Do not commit checkpoints, DINOv3 weights, user memory, or results to GitHub.

## Citation

Citation information will be updated after the Memory-SAM paper is public.
