import torch
import numpy as np
from PIL import Image

class FeatureExtractor:
    """Handles DINOv3 model loading and feature extraction."""

    def __init__(self, model_name="dinov3_vitl16", device="cuda"):
        self.device = device
        self.model_name = model_name
        self.dinov3_matcher = self._load_dinov3_matcher()

    def _load_dinov3_matcher(self):
        try:
            from scripts.dinov3_matcher import Dinov3Matcher
            print(f"Loading DINOv3 matcher/model: {self.model_name}")
            matcher = Dinov3Matcher(model_name=self.model_name, device=str(self.device))
            print("DINOv3 initialized successfully")
            return matcher
        except Exception as e:
            print(f"Failed to initialize DINOv3: {e}")
            raise

    def extract_global_features(self, image: np.ndarray) -> np.ndarray:
        """Extracts global features (CLS + patch mean, L2 normalized)."""
        features = self.dinov3_matcher.extract_global_features(image)
        print(f"Extracted DINOv3 global feature shape: {features.shape}, norm: {np.linalg.norm(features):.6f}")
        return features

    def extract_patch_features(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], float]:
        """Extracts patch features from an image."""
        image_tensor, grid_size, resize_scale = self.dinov3_matcher.prepare_image(image)
        patch_features = self.dinov3_matcher.extract_features(image_tensor)
        
        # Normalize patch features (row-wise)
        normalized_patch_features = np.zeros_like(patch_features)
        for i in range(patch_features.shape[0]):
            norm = np.linalg.norm(patch_features[i])
            if norm > 0:
                normalized_patch_features[i] = patch_features[i] / norm
        
        feature_norms = np.linalg.norm(normalized_patch_features, axis=1)
        mean_norm = np.mean(feature_norms)
        std_norm = np.std(feature_norms)
        
        print(f"Normalized patch feature shape: {normalized_patch_features.shape}")
        print(f"Patch feature norm statistics - mean: {mean_norm:.6f}, std: {std_norm:.6f}")
        
        return normalized_patch_features, grid_size, resize_scale
