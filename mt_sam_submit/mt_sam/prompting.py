from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


class NoPromptCandidatesError(RuntimeError):
    pass


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), eps)
    return arr / norms


def as_patch_matrix(patch_features: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(patch_features, dtype=np.float32)
    if arr.ndim == 2:
        return np.ascontiguousarray(arr, dtype=np.float32)
    if arr.ndim == 3:
        d, h, w = arr.shape
        expected = int(grid_size[0] * grid_size[1])
        out = arr.reshape(d, -1).T
        if out.shape[0] != expected:
            raise ValueError(f"patch count mismatch: got={out.shape[0]}, expected={expected}")
        return np.ascontiguousarray(out, dtype=np.float32)
    raise ValueError(f"unsupported patch feature shape: {arr.shape}")


def quantize_mask_to_grid(mask: np.ndarray, grid_size: Tuple[int, int]) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    arr = (arr > 0).astype(np.uint8) * 255
    resized = cv2.resize(arr, (int(grid_size[1]), int(grid_size[0])), interpolation=cv2.INTER_NEAREST)
    return resized > 127


def compute_similarity_maps(
    query_patch_features: np.ndarray,
    ref_patch_features: np.ndarray,
    ref_mask: np.ndarray,
    query_grid: Tuple[int, int],
    ref_grid: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    query = l2_normalize_rows(as_patch_matrix(query_patch_features, query_grid))
    ref = l2_normalize_rows(as_patch_matrix(ref_patch_features, ref_grid))
    mask_flat = quantize_mask_to_grid(ref_mask, ref_grid).reshape(-1)
    ref_fg = np.ascontiguousarray(ref[mask_flat], dtype=np.float32)
    ref_bg = np.ascontiguousarray(ref[~mask_flat], dtype=np.float32)
    if ref_fg.size == 0:
        raise NoPromptCandidatesError("reference mask has no foreground patches")
    if ref_bg.size == 0:
        raise NoPromptCandidatesError("reference mask has no background patches")
    s_fg = np.max(query @ ref_fg.T, axis=1).astype(np.float32, copy=False)
    s_bg = np.max(query @ ref_bg.T, axis=1).astype(np.float32, copy=False)
    return s_fg, s_bg


def separability_score(s_fg: np.ndarray, s_bg: np.ndarray) -> float:
    return float(np.max(s_fg) - np.max(s_bg))


def suppress_local_conflicts(
    fg_idx: np.ndarray,
    bg_idx: np.ndarray,
    grid_size: Tuple[int, int],
    window_patches: int = 2,
) -> np.ndarray:
    gh, gw = int(grid_size[0]), int(grid_size[1])
    total = gh * gw
    fg = np.unique(np.asarray(fg_idx, dtype=np.int64).reshape(-1))
    bg = np.unique(np.asarray(bg_idx, dtype=np.int64).reshape(-1))
    fg = fg[(fg >= 0) & (fg < total)]
    bg = bg[(bg >= 0) & (bg < total)]
    if fg.size == 0 or bg.size == 0:
        return fg

    win_h = min(gh, max(1, int(window_patches)))
    win_w = min(gw, max(1, int(window_patches)))
    fg_map = np.zeros((gh, gw), dtype=bool)
    bg_map = np.zeros((gh, gw), dtype=bool)
    fg_map[fg // gw, fg % gw] = True
    bg_map[bg // gw, bg % gw] = True
    remove = np.zeros((gh, gw), dtype=bool)
    for y0 in range(gh - win_h + 1):
        for x0 in range(gw - win_w + 1):
            y1 = y0 + win_h
            x1 = x0 + win_w
            if np.any(fg_map[y0:y1, x0:x1]) and np.any(bg_map[y0:y1, x0:x1]):
                remove[y0:y1, x0:x1] |= fg_map[y0:y1, x0:x1]
    return fg[~remove[fg // gw, fg % gw]]


def build_point_prompt(
    s_fg: np.ndarray,
    s_bg: np.ndarray,
    grid_size: Tuple[int, int],
    image_shape: Tuple[int, int],
    tau_fg: float = 0.7,
    tau_bg: float = 0.7,
    k_fg: int = 3,
    conflict_window_patches: int = 2,
) -> Dict:
    gh, gw = int(grid_size[0]), int(grid_size[1])
    h, w = int(image_shape[0]), int(image_shape[1])
    total = gh * gw
    s_fg = np.asarray(s_fg, dtype=np.float32).reshape(-1)
    s_bg = np.asarray(s_bg, dtype=np.float32).reshape(-1)
    if s_fg.size != total or s_bg.size != total:
        raise ValueError("similarity map size does not match feature grid")

    contrast = s_fg - s_bg
    selected = np.argsort(contrast)[::-1][: max(1, int(k_fg))].astype(np.int64)
    if selected.size == 0:
        raise NoPromptCandidatesError("empty contrast map")

    sx = float(w) / float(gw)
    sy = float(h) / float(gh)
    points = np.array(
        [
            [
                float(np.clip((idx % gw + 0.5) * sx, 0, w - 1)),
                float(np.clip((idx // gw + 0.5) * sy, 0, h - 1)),
            ]
            for idx in selected
        ],
        dtype=np.float32,
    )
    return {
        "points": points,
        "labels": np.ones((points.shape[0],), dtype=np.int32),
        "debug": {
            "fg_raw": np.where(s_fg > float(tau_fg))[0].astype(np.int64).tolist(),
            "bg_raw": np.where(s_bg > float(tau_bg))[0].astype(np.int64).tolist(),
            "fg_refined": selected.tolist(),
            "fg_selected": selected.tolist(),
            "selection": "contrast_top_k",
        },
    }
