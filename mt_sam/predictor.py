from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .features import DINOv3FeatureExtractor
from .memory import MemoryBank
from .paths import ensure_third_party_imports, resolve_asset_path
from .prompting import (
    NoPromptCandidatesError,
    build_point_prompt,
    compute_similarity_maps,
    separability_score,
)


@dataclass(frozen=True)
class MTSAMConfig:
    sam_checkpoint: str
    sam_config: str = "configs/sam2.1/sam2.1_hiera_l"
    dinov3_model: str = "dinov3_vitl16"
    dinov3_repo: str = "third_party/dinov3"
    dinov3_weights: str = "assets/dinov3_weights"
    memory_dir: str = "memory_m20"
    device: str = "cuda"
    square_resize: int = 512
    retrieval_top_k: int = 5
    tau_fg: float = 0.7
    tau_bg: float = 0.7
    k_fg: int = 3
    conflict_window_patches: int = 2
    postprocess_mask: bool = True
    postprocess_min_component_ratio: float = 0.01
    postprocess_open_kernel: int = 3
    postprocess_close_kernel: int = 5


def resize_square(image: np.ndarray, size: int = 512) -> np.ndarray:
    if image.shape[:2] == (size, size):
        return image
    interpolation = cv2.INTER_AREA if max(image.shape[:2]) > size else cv2.INTER_LINEAR
    return cv2.resize(image, (size, size), interpolation=interpolation)


def keep_largest_component(mask: np.ndarray, min_component_ratio: float = 0.01) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.sum() == 0:
        return binary.astype(bool)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    largest_area = int(areas[largest_label - 1])
    min_area = int(max(1, round(float(binary.size) * float(min_component_ratio))))
    if largest_area < min_area:
        return binary.astype(bool)
    return labels == largest_label


def fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    flood = padded.copy()
    h, w = flood.shape[:2]
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), dtype=np.uint8), (0, 0), 1)
    holes = (flood == 0).astype(np.uint8)
    filled = np.maximum(padded, holes)[1:-1, 1:-1]
    return filled.astype(bool)


def morphology(mask: np.ndarray, open_kernel: int = 3, close_kernel: int = 5) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if open_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(open_kernel), int(open_kernel)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    if close_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(close_kernel), int(close_kernel)))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary > 0


class MTSAMPredictor:
    def __init__(self, config: MTSAMConfig):
        self.config = config
        self.extractor = DINOv3FeatureExtractor(
            model_name=config.dinov3_model,
            repo_dir=config.dinov3_repo,
            weights_dir=config.dinov3_weights,
            image_size=config.square_resize,
            device=config.device,
        )
        self.memory = MemoryBank(config.memory_dir)
        self.sam_predictor = self._load_sam()

    def _load_sam(self):
        ensure_third_party_imports()
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        from sam2.build_sam import build_sam2

        cfg = self.config.sam_config
        checkpoint = resolve_asset_path(self.config.sam_checkpoint, "checkpoints/sam2.1_hiera_large.pt")
        if cfg.endswith(".yaml"):
            cfg = cfg[:-5]
        model = build_sam2(cfg, str(checkpoint), device=self.config.device)
        return SAM2ImagePredictor(model)

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        with Image.open(path) as image:
            image.load()
            return np.array(image.convert("RGB"))

    def _select_exemplar(self, query_patches: np.ndarray, query_grid, retrieved) -> Dict:
        scored = []
        for candidate in retrieved:
            item_id = int(candidate["item"]["id"])
            ref = self.memory.load_sparse(item_id)
            s_fg, s_bg = compute_similarity_maps(
                query_patch_features=query_patches,
                ref_patch_features=ref["patch_features"],
                ref_mask=ref["mask"],
                query_grid=query_grid,
                ref_grid=ref["grid_size"],
            )
            scored.append(
                {
                    "item_id": item_id,
                    "retrieval_similarity": float(candidate["similarity"]),
                    "s_fg": s_fg,
                    "s_bg": s_bg,
                    "separability": separability_score(s_fg, s_bg),
                }
            )
        if not scored:
            raise NoPromptCandidatesError("no valid retrieved exemplar")
        return max(scored, key=lambda row: float(row["separability"]))

    def _postprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        if not self.config.postprocess_mask:
            return np.asarray(mask) > 0
        cleaned = keep_largest_component(
            mask,
            min_component_ratio=self.config.postprocess_min_component_ratio,
        )
        cleaned = fill_holes(cleaned)
        cleaned = morphology(
            cleaned,
            open_kernel=self.config.postprocess_open_kernel,
            close_kernel=self.config.postprocess_close_kernel,
        )
        cleaned = keep_largest_component(
            cleaned,
            min_component_ratio=self.config.postprocess_min_component_ratio,
        )
        return cleaned > 0

    def predict_array(self, image: np.ndarray) -> Dict:
        started = time.perf_counter()
        image_512 = resize_square(np.asarray(image), self.config.square_resize)
        query_patches, query_grid = self.extractor.extract_patch_features(image_512)
        query_global = self.extractor.extract_global_features(image_512)
        retrieved = self.memory.search(query_global, top_k=self.config.retrieval_top_k)
        if not retrieved:
            raise NoPromptCandidatesError("memory retrieval returned no exemplars")

        picked = self._select_exemplar(query_patches, query_grid, retrieved)
        prompt = build_point_prompt(
            s_fg=picked["s_fg"],
            s_bg=picked["s_bg"],
            grid_size=query_grid,
            image_shape=image_512.shape[:2],
            tau_fg=self.config.tau_fg,
            tau_bg=self.config.tau_bg,
            k_fg=self.config.k_fg,
            conflict_window_patches=self.config.conflict_window_patches,
        )

        self.sam_predictor.set_image(image_512)
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=prompt["points"],
            point_labels=prompt["labels"],
            multimask_output=True,
        )
        scores = np.asarray(scores).reshape(-1)
        best_idx = int(np.argmax(scores)) if scores.size else 0
        masks = np.asarray(masks)
        mask = masks[best_idx] if masks.ndim == 3 else np.squeeze(masks)
        mask = self._postprocess_mask(mask)
        return {
            "image": image_512,
            "mask": mask,
            "score": float(scores[best_idx]) if scores.size else 0.0,
            "prompt": prompt,
            "picked_item_id": int(picked["item_id"]),
            "similarity_maps": {
                "fg": picked["s_fg"],
                "bg": picked["s_bg"],
                "contrast": picked["s_fg"] - picked["s_bg"],
                "grid_size": tuple(query_grid),
            },
            "retrieved": [
                {"item_id": int(row["item"]["id"]), "similarity": float(row["similarity"])}
                for row in retrieved
            ],
            "separability": float(picked["separability"]),
            "latency_ms": float((time.perf_counter() - started) * 1000.0),
        }

    def add_reference(self, image: np.ndarray, mask: np.ndarray) -> int:
        image_512 = resize_square(np.asarray(image), self.config.square_resize)
        mask_512 = resize_square(np.asarray(mask), self.config.square_resize)
        if mask_512.ndim == 3:
            mask_512 = cv2.cvtColor(mask_512.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        patch_features, grid = self.extractor.extract_patch_features(image_512)
        global_features = self.extractor.extract_global_features(image_512)
        return self.memory.add(
            image=image_512,
            mask=mask_512,
            global_features=global_features,
            patch_features=patch_features,
            grid_size=grid,
        )

    def generate_mask_from_points(
        self,
        image: np.ndarray,
        points: Sequence[Tuple[float, float]],
        labels: Sequence[int],
    ) -> Dict:
        image_512 = resize_square(np.asarray(image), self.config.square_resize)
        if len(points) == 0:
            raise ValueError("at least one point prompt is required")
        point_array = np.asarray(points, dtype=np.float32)
        label_array = np.asarray(labels, dtype=np.int32)
        self.sam_predictor.set_image(image_512)
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=point_array,
            point_labels=label_array,
            multimask_output=True,
        )
        scores = np.asarray(scores).reshape(-1)
        best_idx = int(np.argmax(scores)) if scores.size else 0
        masks = np.asarray(masks)
        mask = masks[best_idx] if masks.ndim == 3 else np.squeeze(masks)
        mask = self._postprocess_mask(mask)
        return {
            "image": image_512,
            "mask": mask,
            "score": float(scores[best_idx]) if scores.size else 0.0,
            "points": point_array,
            "labels": label_array,
        }

    def predict_file(self, image_path: str) -> Dict:
        return self.predict_array(self.load_image(image_path))

    @staticmethod
    def save_mask(path: str, mask: np.ndarray) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), np.asarray(mask).astype(np.uint8) * 255)
