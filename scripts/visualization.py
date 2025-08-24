import numpy as np
import cv2
from typing import Optional, Tuple, List

class Visualization:
    """Handles all visualization-related tasks."""

    def __init__(self, sparse_matcher):
        self.sparse_matcher = sparse_matcher

    def visualize_mask(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Visualizes a mask overlay on an image."""
        vis = image.copy()

        # First, ensure mask is a 2D array.
        if mask.ndim != 2:
            print(f"Error: visualize_mask received a mask with invalid dimensions {mask.shape}. Cannot apply overlay.")
            return vis # Return original image without overlay

        # Safety check for shape mismatch (transposed dimensions or different size)
        if image.shape[:2] != mask.shape[:2]:
            if image.shape[0] == mask.shape[1] and image.shape[1] == mask.shape[0]:
                print(f"Warning: Mask shape {mask.shape} seems transposed. Transposing mask.")
                mask = mask.T
            else:
                print(f"Warning: Resizing mask from {mask.shape} to match image shape {image.shape[:2]}.")
                mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        binary_mask = mask > 0
        color_mask = np.zeros_like(vis, dtype=np.uint8)
        color_mask[binary_mask] = [30, 144, 255]
        cv2.addWeighted(color_mask, alpha, vis, 1 - alpha, 0, vis)
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)
        return vis

    def visualize_sparse_matches(
        self, 
        image1: np.ndarray, image2: np.ndarray,
        features1: np.ndarray, features2: np.ndarray,
        grid_size1: Tuple[int, int], grid_size2: Tuple[int, int],
        mask1: Optional[np.ndarray] = None, mask2: Optional[np.ndarray] = None,
        match_background: bool = True, use_kmeans: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Visualizes sparse matches between two images."""
        if mask1 is not None and mask1.ndim == 3: mask1 = cv2.cvtColor(mask1, cv2.COLOR_RGB2GRAY)
        if mask2 is not None and mask2.ndim == 3: mask2 = cv2.cvtColor(mask2, cv2.COLOR_RGB2GRAY)

        resized_mask1 = cv2.resize(mask1.astype(np.uint8), (grid_size1[1], grid_size1[0])).astype(bool) if mask1 is not None else np.ones(grid_size1, dtype=bool)
        resized_mask2 = cv2.resize(mask2.astype(np.uint8), (grid_size2[1], grid_size2[0])).astype(bool) if mask2 is not None else np.ones(grid_size2, dtype=bool)

        fg_coords1, bg_coords1 = np.where(resized_mask1), np.where(~resized_mask1)
        fg_coords2, bg_coords2 = np.where(resized_mask2), np.where(~resized_mask2)

        fg_features1 = features1[:, fg_coords1[0], fg_coords1[1]].T
        fg_features2 = features2[:, fg_coords2[0], fg_coords2[1]].T
        bg_features1 = features1[:, bg_coords1[0], bg_coords1[1]].T
        bg_features2 = features2[:, bg_coords2[0], bg_coords2[1]].T

        fg_match_coords1, fg_match_coords2, fg_similarities = self.sparse_matcher._match_features_with_coords(
            fg_features1, fg_features2, fg_coords1, fg_coords2, grid_size1, grid_size2, image1.shape, image2.shape, 0.8
        )
        bg_match_coords1, bg_match_coords2, bg_similarities = self.sparse_matcher._match_features_with_coords(
            bg_features1, bg_features2, bg_coords1, bg_coords2, grid_size1, grid_size2, image1.shape, image2.shape, 0.7
        )

        print(f"Initial matches found -> FG: {len(fg_match_coords1)}, BG: {len(bg_match_coords1)}")

        if use_kmeans:
            print(f"Applying K-Means clustering... (FG clusters: {self.sparse_matcher.kmeans_fg_clusters}, BG clusters: 5)")
            fg_match_coords1, fg_match_coords2, _ = self.sparse_matcher._cluster_feature_points(fg_match_coords1, fg_match_coords2, fg_similarities, self.sparse_matcher.kmeans_fg_clusters)
            
            if match_background:
                bg_match_coords1, bg_match_coords2, _ = self.sparse_matcher._cluster_feature_points(bg_match_coords1, bg_match_coords2, bg_similarities, 5)
            
            print(f"Points after clustering -> FG: {len(fg_match_coords1)}, BG: {len(bg_match_coords1) if match_background else 0}")
        else:
            print("Skipping clustering (showing all raw matches).")


        coords1 = fg_match_coords1 + (bg_match_coords1 if match_background else [])
        coords2 = fg_match_coords2 + (bg_match_coords2 if match_background else [])

        h1, w1 = image1.shape[:2]; h2, w2 = image2.shape[:2]
        vis_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis_img[:h1, :w1] = image1; vis_img[:h2, w1:w1+w2] = image2

        for (x1, y1), (x2, y2) in zip(coords1, coords2):
            color = (0, 255, 0)
            cv2.circle(vis_img, (x1, y1), 5, color, -1)
            cv2.circle(vis_img, (x2 + w1, y2), 5, color, -1)
            cv2.line(vis_img, (x1, y1), (x2 + w1, y2), color, 1)

        img1_points = self._draw_points_on_image_internal(image1, fg_match_coords1, bg_match_coords1, match_background)
        img2_points = self._draw_points_on_image_internal(image2, fg_match_coords2, bg_match_coords2, match_background)

        return vis_img, img1_points, img2_points

    def _draw_points_on_image_internal(self, image, fg_points, bg_points, match_background):
        img_copy = image.copy()
        for x, y in fg_points: cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
        if match_background:
            for x, y in bg_points: cv2.circle(img_copy, (x, y), 5, (255, 0, 0), -1)
        return img_copy
