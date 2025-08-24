import numpy as np
import cv2
from typing import Tuple, Dict

class PromptGenerator:
    """Generates SAM prompts from masks."""

    def __init__(self, use_kmeans_fg: bool = True, kmeans_fg_clusters: int = 10):
        self.use_kmeans_fg = use_kmeans_fg
        self.kmeans_fg_clusters = kmeans_fg_clusters
        self.last_fg_prompt_points = []
        self.last_bg_prompt_points = []

    def _kmeans_sampling(self, points: np.ndarray, n_clusters: int) -> np.ndarray:
        """Samples points using K-means clustering."""
        if len(points) <= n_clusters:
            return points
        
        # If there are too many points, take a random sample to avoid performance issues
        if len(points) > 50000:
            sample_indices = np.random.choice(len(points), 50000, replace=False)
            points_to_cluster = points[sample_indices]
        else:
            points_to_cluster = points

        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(points_to_cluster)
            
            selected_points = []
            for center in kmeans.cluster_centers_:
                # Find the closest point in the *original* set, not the sample
                distances = np.sum((points - center) ** 2, axis=1)
                closest_idx = np.argmin(distances)
                selected_points.append(points[closest_idx])
            return np.array(selected_points)
        except ImportError:
            print("scikit-learn not installed, using random sampling.")
            indices = np.random.choice(len(points), n_clusters, replace=False)
            return points[indices]

    def generate_prompt(self, mask: np.ndarray, 
                      original_size: Tuple[int, int], 
                      match_background: bool = True) -> Dict:
        """Generates a point prompt from a binary mask, resizing to match original_size."""
        
        # Resize mask to match the current image size before generating points
        if mask.shape[:2] != original_size:
            mask_resized = cv2.resize(mask.astype(np.uint8), (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask

        if np.sum(mask_resized) == 0:
            print("Warning: Empty mask passed to generate_prompt.")
            h, w = mask_resized.shape[:2]
            return {"points": np.array([[w // 2, h // 2]]), "labels": np.array([1])}

        foreground_points = np.argwhere(mask_resized > 0)
        
        points = []
        labels = []

        # Foreground points
        if len(foreground_points) > 0:
            k = max(1, self.kmeans_fg_clusters)
            if self.use_kmeans_fg:
                print(f" -> Generating {k} foreground points using K-Means.")
                selected_fg_points = self._kmeans_sampling(foreground_points, k)
            else:
                print(f" -> Generating {k} foreground points using random sampling.")
                if len(foreground_points) > k:
                    indices = np.random.choice(len(foreground_points), k, replace=False)
                    selected_fg_points = foreground_points[indices]
                else:
                    selected_fg_points = foreground_points
            
            self.last_fg_prompt_points = []
            for pt in selected_fg_points:
                points.append([pt[1], pt[0]]) # x, y order
                labels.append(1)
                self.last_fg_prompt_points.append([int(pt[1]), int(pt[0])])

        # Background points
        if match_background:
            background_points = np.argwhere(mask_resized == 0)
            if len(background_points) > 0:
                num_bg_points = min(5, len(background_points))
                selected_bg_points = self._kmeans_sampling(background_points, num_bg_points)
                
                self.last_bg_prompt_points = []
                for pt in selected_bg_points:
                    points.append([pt[1], pt[0]])
                    labels.append(0)
                    self.last_bg_prompt_points.append([int(pt[1]), int(pt[0])])

        if not points:
            h, w = mask_resized.shape[:2]
            return {"points": np.array([[w // 2, h // 2]]), "labels": np.array([1])}

        return {"points": np.array(points), "labels": np.array(labels)}
