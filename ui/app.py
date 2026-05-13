#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mt_sam import NoPromptCandidatesError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def as_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    if arr.shape[-1] == 4:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2RGB)
    return arr.astype(np.uint8)


def as_binary_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return arr > 0


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    image = as_rgb(image).copy()
    mask_b = as_binary_mask(mask)
    color_img = np.zeros_like(image)
    color_img[..., 0] = int(color[0])
    color_img[..., 1] = int(color[1])
    color_img[..., 2] = int(color[2])
    image[mask_b] = ((1.0 - alpha) * image[mask_b] + alpha * color_img[mask_b]).astype(np.uint8)
    return image


def draw_points(image: np.ndarray, points: Sequence[Tuple[float, float]], labels: Sequence[int]) -> np.ndarray:
    canvas = as_rgb(image).copy()
    for (x, y), label in zip(points, labels):
        color = (0, 255, 0) if int(label) == 1 else (255, 0, 0)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 7, color, -1)
        cv2.circle(canvas, (int(round(x)), int(round(y))), 9, (255, 255, 255), 2)
    return canvas


def heatmap(values: np.ndarray, grid_size: Tuple[int, int], image_shape=(512, 512)) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(int(grid_size[0]), int(grid_size[1]))
    arr = arr - float(np.min(arr))
    max_value = float(np.max(arr))
    if max_value > 0:
        arr = arr / max_value
    arr = cv2.resize(arr, (int(image_shape[1]), int(image_shape[0])), interpolation=cv2.INTER_CUBIC)
    return cv2.applyColorMap((arr * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]


def safe_slug(path: Path, idx: int) -> str:
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)
    return f"{idx:04d}_{stem}"


def file_path(uploaded) -> str:
    if uploaded is None:
        return ""
    if isinstance(uploaded, str):
        return uploaded
    return getattr(uploaded, "name", "") or getattr(uploaded, "path", "")


def collect_image_paths(files, folder_text: str) -> List[Path]:
    paths: List[Path] = []
    for uploaded in files or []:
        path = Path(file_path(uploaded))
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.exists():
            paths.append(path)
    if folder_text:
        folder = Path(folder_text).expanduser()
        if folder.exists():
            paths.extend(path for path in sorted(folder.rglob("*")) if path.suffix.lower() in IMAGE_EXTENSIONS)
    unique = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def build_app(predictor, default_results_dir: str):
    import gradio as gr
    globals()["gr"] = gr

    def memory_status() -> str:
        return f"Memory items: {predictor.memory.count()}"

    def memory_table():
        rows = []
        for item in predictor.memory.list_items():
            rows.append(
                [
                    int(item["id"]),
                    item["image_path"],
                    item["mask_path"],
                    item["features_path"],
                ]
            )
        return rows

    def memory_gallery():
        items = []
        for item in predictor.memory.list_items():
            try:
                image, mask = predictor.memory.load_image_mask(int(item["id"]))
                items.append((overlay_mask(image, mask), f"ID {int(item['id'])}"))
            except Exception as exc:
                items.append((np.zeros((64, 64, 3), dtype=np.uint8), f"ID {int(item['id'])}: {exc}"))
        return items

    def refresh_memory():
        return memory_status(), memory_table(), memory_gallery()

    def add_reference(reference_image, reference_mask):
        if reference_image is None:
            return memory_status(), memory_table(), memory_gallery(), "Reference image is required."
        if reference_mask is None:
            return memory_status(), memory_table(), memory_gallery(), "Reference mask is required."
        item_id = predictor.add_reference(reference_image, reference_mask)
        return memory_status(), memory_table(), memory_gallery(), f"Added memory item {item_id}."

    def preview_memory(item_id_text):
        if not str(item_id_text).strip():
            return None, None, "Enter a memory item ID."
        item_id = int(item_id_text)
        image, mask = predictor.memory.load_image_mask(item_id)
        return image, overlay_mask(image, mask), json.dumps(predictor.memory.get_item(item_id), indent=2)

    def delete_memory(item_id_text):
        if not str(item_id_text).strip():
            return memory_status(), memory_table(), memory_gallery(), "Enter a memory item ID."
        predictor.memory.delete(int(item_id_text))
        return memory_status(), memory_table(), memory_gallery(), f"Deleted memory item {int(item_id_text)}."

    def clear_memory(confirm_text):
        if str(confirm_text).strip() != "CLEAR":
            return memory_status(), memory_table(), memory_gallery(), "Type CLEAR to remove every memory item."
        predictor.memory.clear()
        return memory_status(), memory_table(), memory_gallery(), "Memory is empty."

    def run_single(image, results_dir):
        if image is None:
            return None, None, None, None, None, None, "No image provided."
        try:
            result = predictor.predict_array(image)
        except NoPromptCandidatesError as exc:
            return image, None, None, None, None, None, f"Memory-SAM could not produce prompts: {exc}. Add memory items first."
        mask = result["mask"].astype(np.uint8) * 255
        overlay = overlay_mask(result["image"], result["mask"])
        prompt_overlay = draw_points(result["image"], result["prompt"]["points"], result["prompt"]["labels"])
        maps = result["similarity_maps"]
        fg_map = heatmap(maps["fg"], maps["grid_size"], result["image"].shape[:2])
        bg_map = heatmap(maps["bg"], maps["grid_size"], result["image"].shape[:2])
        contrast_map = heatmap(maps["contrast"], maps["grid_size"], result["image"].shape[:2])
        saved = ""
        if str(results_dir).strip():
            out_dir = Path(str(results_dir)).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(out_dir / f"{stamp}_mask.png"), mask)
            cv2.imwrite(str(out_dir / f"{stamp}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            meta = {
                "score": result["score"],
                "picked_item_id": result["picked_item_id"],
                "retrieved": result["retrieved"],
                "separability": result["separability"],
                "latency_ms": result["latency_ms"],
                "prompt_points": result["prompt"]["points"].tolist(),
                "prompt_labels": result["prompt"]["labels"].tolist(),
            }
            (out_dir / f"{stamp}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            saved = f"Saved to {out_dir}"
        status = (
            f"Selected memory item: {result['picked_item_id']} | "
            f"SAM score: {result['score']:.4f} | "
            f"Separability: {result['separability']:.4f} | "
            f"Latency: {result['latency_ms']:.1f} ms"
        )
        if saved:
            status = f"{status}\n{saved}"
        return overlay, mask, prompt_overlay, fg_map, bg_map, contrast_map, status

    def run_batch(files, folder_text, results_dir):
        paths = collect_image_paths(files, folder_text)
        if not paths:
            return [], "No images found."
        out_dir = Path(str(results_dir or default_results_dir)).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        gallery = []
        records = []
        failures = []
        for idx, path in enumerate(paths):
            try:
                image = predictor.load_image(str(path))
                result = predictor.predict_array(image)
                mask = result["mask"].astype(np.uint8) * 255
                overlay = overlay_mask(result["image"], result["mask"])
                slug = safe_slug(path, idx)
                mask_path = out_dir / f"{slug}_mask.png"
                overlay_path = out_dir / f"{slug}_overlay.png"
                meta_path = out_dir / f"{slug}_meta.json"
                cv2.imwrite(str(mask_path), mask)
                cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                meta = {
                    "image_path": str(path),
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "score": result["score"],
                    "picked_item_id": result["picked_item_id"],
                    "retrieved": result["retrieved"],
                    "separability": result["separability"],
                    "latency_ms": result["latency_ms"],
                    "prompt_points": result["prompt"]["points"].tolist(),
                }
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
                records.append(meta)
                gallery.append((overlay, path.name))
            except Exception as exc:
                failures.append(f"{path}: {exc}")
        summary_path = out_dir / "batch_summary.json"
        summary_path.write_text(json.dumps({"results": records, "failures": failures}, indent=2), encoding="utf-8")
        status = f"Processed {len(records)}/{len(paths)} images. Results: {out_dir}"
        if failures:
            status += "\nFailures:\n" + "\n".join(failures[:20])
        return gallery, status

    def point_summary(points, labels):
        fg = sum(1 for label in labels if int(label) == 1)
        bg = sum(1 for label in labels if int(label) == 0)
        return f"Points: {len(points)} | Foreground: {fg} | Background: {bg}"

    def preview_point_mask(image, points, labels):
        if image is None:
            return None, None, None, "Upload an image first.", None
        point_preview = draw_points(image, points, labels)
        if not points:
            return point_preview, None, None, point_summary(points, labels), None
        try:
            result = predictor.generate_mask_from_points(image, points, labels)
        except Exception as exc:
            return point_preview, None, None, f"{point_summary(points, labels)}\nMask preview failed: {exc}", None
        mask = result["mask"].astype(np.uint8) * 255
        overlay = overlay_mask(result["image"], result["mask"])
        status = f"{point_summary(points, labels)} | Live SAM score: {result['score']:.4f}"
        return point_preview, overlay, mask, status, mask

    def add_point(image, points, labels, point_type, evt: gr.SelectData):
        if image is None:
            return None, points, labels, None, None, "Upload an image first.", None
        points = list(points or [])
        labels = list(labels or [])
        x, y = evt.index
        label = 1 if point_type == "Foreground" else 0
        points.append((float(x), float(y)))
        labels.append(label)
        point_preview, overlay, mask, status, mask_state = preview_point_mask(image, points, labels)
        return point_preview, points, labels, overlay, mask, status, mask_state

    def undo_point(image, points, labels):
        points = list(points or [])
        labels = list(labels or [])
        if points:
            points.pop()
            labels.pop()
        point_preview, overlay, mask, status, mask_state = preview_point_mask(image, points, labels)
        return point_preview, points, labels, overlay, mask, status, mask_state

    def clear_points(image):
        return image, [], [], None, None, "Points: 0 | Foreground: 0 | Background: 0", None

    def generate_point_mask(image, points, labels):
        if image is None:
            return None, None, "Upload an image first.", None
        if not points:
            return None, None, "Add at least one point.", None
        result = predictor.generate_mask_from_points(image, points, labels)
        mask = result["mask"].astype(np.uint8) * 255
        overlay = overlay_mask(result["image"], result["mask"])
        return overlay, mask, f"SAM score: {result['score']:.4f}", mask

    def save_generated_memory(image, generated_mask):
        if image is None:
            return memory_status(), memory_table(), memory_gallery(), "Upload an image first."
        if generated_mask is None:
            return memory_status(), memory_table(), memory_gallery(), "Generate a mask first."
        item_id = predictor.add_reference(image, generated_mask)
        return memory_status(), memory_table(), memory_gallery(), f"Saved generated mask as memory item {item_id}."

    with gr.Blocks(title="Memory-SAM UI") as demo:
        gr.Markdown(
            """
            # Memory-SAM UI
            Training-free retrieval-to-prompt tongue segmentation with SAM2.

            The memory bank starts empty. Add your own reference image/mask pairs, or create masks with point prompts, then run segmentation.
            """
        )
        memory_box = gr.Textbox(label="Memory Status", value=memory_status(), interactive=False)
        with gr.Accordion("Memory Items", open=False):
            memory_table_box = gr.Dataframe(
                headers=["ID", "Image", "Mask", "Features"],
                value=memory_table(),
                interactive=False,
                label="Current Memory Bank",
            )
            memory_gallery_box = gr.Gallery(
                label="Reference Overlay Gallery",
                value=memory_gallery(),
                columns=5,
                height=260,
            )

        with gr.Tab("Segment"):
            with gr.Row():
                input_image = gr.Image(type="numpy", label="Query Image")
                single_overlay = gr.Image(type="numpy", label="Overlay")
            with gr.Row():
                single_mask = gr.Image(type="numpy", label="Mask")
                prompt_overlay = gr.Image(type="numpy", label="Prompt Points")
            with gr.Accordion("Memory-SAM Debug Visualization", open=False):
                with gr.Row():
                    fg_heatmap = gr.Image(type="numpy", label="Foreground Similarity")
                    bg_heatmap = gr.Image(type="numpy", label="Background Similarity")
                    contrast_heatmap = gr.Image(type="numpy", label="Contrast")
            result_dir = gr.Textbox(label="Results Directory", value=default_results_dir)
            segment_status = gr.Textbox(label="Status", lines=4, interactive=False)
            gr.Button("Run Memory-SAM").click(
                run_single,
                inputs=[input_image, result_dir],
                outputs=[single_overlay, single_mask, prompt_overlay, fg_heatmap, bg_heatmap, contrast_heatmap, segment_status],
            )

        with gr.Tab("Batch Segment"):
            batch_files = gr.Files(label="Upload Images")
            batch_folder = gr.Textbox(label="Or Folder Path")
            batch_results_dir = gr.Textbox(label="Results Directory", value=default_results_dir)
            batch_gallery = gr.Gallery(label="Batch Overlays", columns=4, height=420)
            batch_status = gr.Textbox(label="Batch Status", lines=8, interactive=False)
            gr.Button("Run Batch").click(
                run_batch,
                inputs=[batch_files, batch_folder, batch_results_dir],
                outputs=[batch_gallery, batch_status],
            )

        with gr.Tab("Build Memory"):
            gr.Markdown("Add validated reference image/mask pairs. These are the only items used by Memory-SAM retrieval.")
            with gr.Row():
                reference_image = gr.Image(type="numpy", label="Reference Image")
                reference_mask = gr.Image(type="numpy", label="Reference Binary Mask")
            add_status = gr.Textbox(label="Add Status", interactive=False)
            gr.Button("Add Reference To Memory").click(
                add_reference,
                inputs=[reference_image, reference_mask],
                outputs=[memory_box, memory_table_box, memory_gallery_box, add_status],
            )

        with gr.Tab("Point Mask Tool"):
            gr.Markdown("Create a reference mask with SAM2 point prompts. The mask preview updates after every click.")
            point_points = gr.State([])
            point_labels = gr.State([])
            generated_mask_state = gr.State(None)
            with gr.Row():
                point_image = gr.Image(type="numpy", label="Image For Mask Creation")
                point_preview = gr.Image(type="numpy", label="Point Overlay")
            point_type = gr.Radio(["Foreground", "Background"], value="Foreground", label="Next Point Type")
            point_status = gr.Textbox(label="Prompt Summary", value="Points: 0 | Foreground: 0 | Background: 0", interactive=False)
            with gr.Row():
                generated_overlay = gr.Image(type="numpy", label="Live Mask Overlay")
                generated_mask = gr.Image(type="numpy", label="Live Mask")
            point_image.select(
                add_point,
                inputs=[point_image, point_points, point_labels, point_type],
                outputs=[point_preview, point_points, point_labels, generated_overlay, generated_mask, point_status, generated_mask_state],
            )
            with gr.Row():
                gr.Button("Undo Last Point").click(
                    undo_point,
                    inputs=[point_image, point_points, point_labels],
                    outputs=[point_preview, point_points, point_labels, generated_overlay, generated_mask, point_status, generated_mask_state],
                )
                gr.Button("Clear Points").click(
                    clear_points,
                    inputs=[point_image],
                    outputs=[point_preview, point_points, point_labels, generated_overlay, generated_mask, point_status, generated_mask_state],
                )
            generated_status = gr.Textbox(label="Generated Mask Status", interactive=False)
            gr.Button("Generate Mask").click(
                generate_point_mask,
                inputs=[point_image, point_points, point_labels],
                outputs=[generated_overlay, generated_mask, generated_status, generated_mask_state],
            )
            save_generated_status = gr.Textbox(label="Save Status", interactive=False)
            gr.Button("Save Generated Mask To Memory").click(
                save_generated_memory,
                inputs=[point_image, generated_mask_state],
                outputs=[memory_box, memory_table_box, memory_gallery_box, save_generated_status],
            )

        with gr.Tab("Memory Manager"):
            with gr.Row():
                selected_id = gr.Textbox(label="Memory Item ID")
                gr.Button("Refresh").click(refresh_memory, outputs=[memory_box, memory_table_box, memory_gallery_box])
            with gr.Row():
                memory_image = gr.Image(type="numpy", label="Reference Image")
                memory_overlay = gr.Image(type="numpy", label="Reference Overlay")
            memory_meta = gr.Textbox(label="Metadata", lines=8, interactive=False)
            with gr.Row():
                gr.Button("Preview Item").click(preview_memory, inputs=[selected_id], outputs=[memory_image, memory_overlay, memory_meta])
                gr.Button("Delete Item").click(delete_memory, inputs=[selected_id], outputs=[memory_box, memory_table_box, memory_gallery_box, memory_meta])
            clear_confirm = gr.Textbox(label="Type CLEAR To Remove All Memory Items")
            gr.Button("Clear All Memory").click(clear_memory, inputs=[clear_confirm], outputs=[memory_box, memory_table_box, memory_gallery_box, memory_meta])

    return demo


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the Memory-SAM Gradio UI.")
    parser.add_argument("--memory-dir", default="user_memory")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--sam-checkpoint", default="checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l")
    parser.add_argument("--dinov3-model", default="dinov3_vitl16")
    parser.add_argument("--dinov3-repo", default="third_party/dinov3")
    parser.add_argument("--dinov3-weights", default="assets/dinov3_weights")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
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
    build_app(predictor, args.results_dir).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
