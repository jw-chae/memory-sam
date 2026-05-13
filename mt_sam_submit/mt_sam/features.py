from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from .paths import resolve_asset_path


class DINOv3FeatureExtractor:
    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        repo_dir: str = "dinov3",
        weights_dir: str = "dinov3_weights",
        image_size: int = 512,
        patch_size: int = 16,
        device: str = "cuda",
    ):
        self.model_name = model_name
        self.repo_dir = resolve_asset_path(repo_dir, "third_party/dinov3")
        self.weights_dir = resolve_asset_path(weights_dir, "assets/dinov3_weights")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.device = device
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.model_to_layers = {
            "dinov3_vits16": 12,
            "dinov3_vits16plus": 12,
            "dinov3_vitb16": 12,
            "dinov3_vitl16": 24,
            "dinov3_vit7b16": 40,
        }
        self.n_layers = self.model_to_layers.get(model_name, 24)
        self.model = self._load_model()
        self.model.to(self.device).eval()

    def _weight_path(self) -> Path:
        weights = {
            "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
            "dinov3_vit7b16": "dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth",
        }
        return self.weights_dir / weights[self.model_name]

    def _load_model(self):
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"DINOv3 repo not found: {self.repo_dir}")
        weight_path = self._weight_path()
        if not weight_path.exists():
            raise FileNotFoundError(f"DINOv3 weight not found: {weight_path}")
        model = torch.hub.load(str(self.repo_dir), self.model_name, source="local", pretrained=False)
        state = torch.load(str(weight_path), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)
        return model

    def prepare_image(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        pil = Image.fromarray(image)
        w, h = pil.size
        h_patches = int(self.image_size / self.patch_size)
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        target_h = h_patches * self.patch_size
        target_w = w_patches * self.patch_size
        tensor = TF.to_tensor(TF.resize(pil, (target_h, target_w)))
        tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        return tensor, (h_patches, w_patches)

    def _layers_for_dense_features(self) -> List[int]:
        if self.n_layers >= 24:
            return [11, 17, 23]
        return [max(0, self.n_layers - 3), max(0, self.n_layers - 2), max(0, self.n_layers - 1)]

    def extract_patch_features(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        tensor, grid = self.prepare_image(image)
        with torch.no_grad():
            batch = tensor.unsqueeze(0).to(self.device)
            feats = self.model.get_intermediate_layers(
                batch,
                n=self._layers_for_dense_features(),
                reshape=True,
                norm=True,
            )
            fused = torch.cat(feats, dim=1).squeeze(0).float().cpu()
        d, _, _ = fused.shape
        patches = fused.reshape(d, -1).T.numpy().astype(np.float32)
        patches /= np.maximum(np.linalg.norm(patches, axis=1, keepdims=True), 1e-8)
        return patches, grid

    def extract_global_features(self, image: np.ndarray) -> np.ndarray:
        tensor, _ = self.prepare_image(image)
        with torch.no_grad():
            batch = tensor.unsqueeze(0).to(self.device)
            feats = self.model.get_intermediate_layers(
                batch,
                n=[max(0, self.n_layers - 1)],
                reshape=False,
                norm=True,
            )
            cls_token = feats[0][0, 0]
            desc = F.normalize(cls_token, p=2, dim=0)
        return desc.cpu().numpy().astype(np.float32)
