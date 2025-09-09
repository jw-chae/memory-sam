"""
DINOv3 기반 이미지 매칭 - 노트북과 정확히 동일한 구현
"""
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from typing import Tuple, List, Optional


class Dinov3Matcher:
    """DINOv3 노트북의 sparse matching 로직을 정확히 구현"""
    
    def __init__(self,
                 model_name: str = "dinov3_vitl16",
                 image_size: int = 768,
                 patch_size: int = 16,
                 device: str = "cuda"):
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.device = device
        
        # ImageNet 정규화 상수
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
        # 모델별 레이어 수
        self.model_to_layers = {
            "dinov3_vits16": 12,
            "dinov3_vits16plus": 12,
            "dinov3_vitb16": 12,
            "dinov3_vitl16": 24,
            "dinov3_vith16plus": 32,
            "dinov3_vit7b16": 40,
        }
        self.n_layers = self.model_to_layers.get(model_name, 24)
        
        # DINOv3 모델 로드
        print(f"Loading DINOv3 model: {model_name}")
        
        # 로컬 가중치 및 로컬 리포지토리 우선 사용
        import os
        import shutil
        WEIGHTS_DIR = "/home/joongwon00/memory-sam/dinov3_weights"
        LOCAL_REPO_DIR = "/home/joongwon00/memory-sam/vendor/dinov3"
        HUB_CACHE_REPO_DIR = "/home/joongwon00/.cache/torch/hub/facebookresearch_dinov3_main"
        CHECKPOINTS_DIR = os.path.join(os.path.expanduser("~"), ".cache", "torch", "hub", "checkpoints")
        os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
        
        # 모델명 -> 가중치 파일 매핑
        weights_map = {
            "dinov3_vitl16": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
            "dinov3_vits16": "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
            "dinov3_vits16plus": "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
            # vit7b16은 여러 변형이 있으므로 가벼운 linear head 버전으로 기본 설정
            "dinov3_vit7b16": "dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth",
        }
        
        local_weight_path = os.path.join(WEIGHTS_DIR, weights_map.get(model_name, "")) if model_name in weights_map else None
        if local_weight_path and os.path.exists(local_weight_path):
            dest_weight_path = os.path.join(CHECKPOINTS_DIR, os.path.basename(local_weight_path))
            if not os.path.exists(dest_weight_path):
                shutil.copyfile(local_weight_path, dest_weight_path)
                print(f"Copied weights to torch hub cache: {dest_weight_path}")
        else:
            if model_name in weights_map:
                print(f"Warning: Local weight not found for {model_name} at {local_weight_path}")
            else:
                print(f"Warning: No weight mapping for model {model_name}. Proceeding without local weight copy.")
        
        # 1) 로컬 리포지토리가 있으면 우선 사용 (네트워크 접근 방지)
        try:
            if os.path.isdir(LOCAL_REPO_DIR):
                print(f"Loading DINOv3 from local repo: {LOCAL_REPO_DIR}")
                self.model = torch.hub.load(
                    repo_or_dir=LOCAL_REPO_DIR,
                    model=model_name,
                    source="local"
                )
            elif os.path.isdir(HUB_CACHE_REPO_DIR):
                print(f"Loading DINOv3 from hub cache repo: {HUB_CACHE_REPO_DIR}")
                self.model = torch.hub.load(
                    repo_or_dir=HUB_CACHE_REPO_DIR,
                    model=model_name,
                    source="local"
                )
            else:
                raise FileNotFoundError(f"Local repos not found: {LOCAL_REPO_DIR} or {HUB_CACHE_REPO_DIR}")
        except Exception as e_local:
            print(f"Local load failed ({e_local}). Falling back to GitHub.")
            # 2) 로컬 리포 없거나 실패 시 GitHub 사용 (허브 캐시의 가중치 활용)
            self.model = torch.hub.load(
                repo_or_dir="facebookresearch/dinov3",
                model=model_name,
                source="github"
            )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 마스크 양자화 필터 (16x16 box blur)
        self.patch_quant_filter = torch.nn.Conv2d(1, 1, patch_size, stride=patch_size, bias=False)
        self.patch_quant_filter.weight.data.fill_(1.0 / (patch_size * patch_size))
        self.patch_quant_filter = self.patch_quant_filter.to(self.device)
    
    def resize_transform(self, image: Image.Image) -> torch.Tensor:
        """노트북과 동일한 리사이즈: 세로 패치 수 고정, 가로는 비율로 계산"""
        w, h = image.size
        h_patches = int(self.image_size / self.patch_size)
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        target_h = h_patches * self.patch_size
        target_w = w_patches * self.patch_size
        return TF.to_tensor(TF.resize(image, (target_h, target_w)))
    
    def prepare_image(self, rgb_image_numpy: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], float]:
        """이미지 준비: 리사이즈 + 정규화"""
        image = Image.fromarray(rgb_image_numpy)
        w, h = image.size
        
        # 리사이즈
        image_tensor = self.resize_transform(image)
        
        # 정규화
        image_tensor = TF.normalize(image_tensor, mean=self.mean, std=self.std)
        
        # 그리드 크기 계산
        h_patches = int(self.image_size / self.patch_size)
        w_patches = int((w * self.image_size) / (h * self.patch_size))
        grid_size = (h_patches, w_patches)
        
        # 리사이즈 스케일 (원본으로 되돌리기 위한)
        resize_scale = h / self.image_size
        
        return image_tensor, grid_size, resize_scale
    
    def prepare_mask(self, mask_numpy: np.ndarray, grid_size: Tuple[int, int]) -> torch.Tensor:
        """마스크를 패치 그리드로 양자화"""
        # 마스크를 단일 채널로 변환
        if mask_numpy.ndim > 2:
            mask_numpy = mask_numpy[:, :, 0]
        
        # 0-255 범위로 변환
        if mask_numpy.max() <= 1:
            mask_numpy = (mask_numpy > 0).astype(np.uint8) * 255
        else:
            mask_numpy = mask_numpy.astype(np.uint8)
        
        # PIL Image로 변환
        mask_pil = Image.fromarray(mask_numpy, mode='L')
        
        # 타겟 크기로 리사이즈
        h_patches, w_patches = grid_size
        target_h = h_patches * self.patch_size
        target_w = w_patches * self.patch_size
        
        # 텐서로 변환하고 리사이즈
        mask_tensor = TF.to_tensor(TF.resize(mask_pil, (target_h, target_w)))
        mask_tensor = mask_tensor.to(self.device)
        
        # 양자화 필터 적용
        with torch.no_grad():
            mask_quantized = self.patch_quant_filter(mask_tensor.unsqueeze(0))
        
        return mask_quantized.squeeze().cpu()
    
    def extract_features(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """노트북과 동일한 특징 추출"""
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            
            # get_intermediate_layers 호출
            feats = self.model.get_intermediate_layers(
                image_batch, 
                n=range(self.n_layers), 
                reshape=True, 
                norm=True
            )
            
            # 마지막 레이어 사용
            features = feats[-1].squeeze().cpu()  # [D, H, W]
            
        return features
    
    def extract_global_features(self, image: np.ndarray) -> np.ndarray:
        """전역 특징 추출 (CLS 토큰만 사용, L2 정규화)"""
        # 이미지 준비
        image_tensor, _, _ = self.prepare_image(image)
        
        with torch.no_grad():
            image_batch = image_tensor.unsqueeze(0).to(self.device)
            
            # 모델에서 특징 추출
            feats = self.model.get_intermediate_layers(
                image_batch, 
                n=range(self.n_layers), 
                reshape=False,  # CLS 토큰을 위해 reshape=False
                norm=True
            )
            
            # 마지막 레이어 사용
            features = feats[-1].squeeze()  # [num_patches + 1, D]
            
            # CLS 토큰 (첫 번째 토큰)만 사용
            cls_token = features[0]  # [D] = [768]
            
            # L2 정규화
            global_features = F.normalize(cls_token, p=2, dim=0)
            
        return global_features.cpu().numpy()
    
    def match_images(self, 
                    image1: np.ndarray, 
                    image2: np.ndarray,
                    mask1: Optional[np.ndarray] = None,
                    mask2: Optional[np.ndarray] = None,
                    stratify_threshold: float = 100.0) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """노트북과 동일한 sparse matching 수행"""
        
        # 이미지 준비
        tensor1, grid_size1, scale1 = self.prepare_image(image1)
        tensor2, grid_size2, scale2 = self.prepare_image(image2)
        
        # 특징 추출
        features1 = self.extract_features(tensor1)  # [D, H1, W1]
        features2 = self.extract_features(tensor2)  # [D, H2, W2]
        
        # 특징 정규화 (중요: dim=0으로 정규화)
        features1 = F.normalize(features1, p=2, dim=0)
        features2 = F.normalize(features2, p=2, dim=0)
        
        # 마스크 처리
        if mask1 is not None:
            mask1_quantized = self.prepare_mask(mask1, grid_size1)
        else:
            mask1_quantized = torch.ones(grid_size1)
            
        if mask2 is not None:
            mask2_quantized = self.prepare_mask(mask2, grid_size2)
        else:
            mask2_quantized = torch.ones(grid_size2)
        
        # 차원 정보
        dim = features1.shape[0]
        n_patches1 = features1.shape[1] * features1.shape[2]
        n_patches2 = features2.shape[1] * features2.shape[2]
        
        # 코사인 유사도 히트맵 계산
        # 노트북의 einsum 연산과 정확히 동일
        heatmaps = torch.einsum(
            "d n, d h w -> n h w",
            features1.view(dim, -1),
            features2
        )
        
        # 왼쪽 이미지의 2D 패치 위치 계산
        patch_indices_left = torch.arange(n_patches1)
        locs_2d_left = (
            torch.stack([
                patch_indices_left // features1.shape[2],  # row
                patch_indices_left % features1.shape[2]    # col
            ], dim=-1) + 0.5
        ) * self.patch_size
        
        # 오른쪽 이미지의 대응 패치 찾기 (argmax)
        patch_indices_right = torch.flatten(heatmaps, start_dim=-2).argmax(dim=-1)
        locs_2d_right = (
            torch.stack([
                patch_indices_right // features2.shape[2],  # row
                patch_indices_right % features2.shape[2]    # col
            ], dim=-1) + 0.5
        ) * self.patch_size
        
        # 전경 마스크 필터링
        MASK_FG_THRESHOLD = 0.5
        patches_left_fg = mask1_quantized.view(-1) > MASK_FG_THRESHOLD
        patches_right_fg = mask2_quantized.view(-1)[patch_indices_right] > MASK_FG_THRESHOLD
        patches_fg_selection = patches_left_fg & patches_right_fg
        
        # 전경 포인트만 선택
        locs_2d_left_fg = locs_2d_left[patches_fg_selection]
        locs_2d_right_fg = locs_2d_right[patches_fg_selection]
        
        if len(locs_2d_left_fg) == 0:
            return [], []
        
        # Stratify points (공간적 분산)
        indices_to_keep = self._stratify_points(
            locs_2d_left_fg * scale1,
            stratify_threshold ** 2
        )
        
        # 최종 포인트 선택
        sparse_points_left = locs_2d_left_fg[indices_to_keep] * scale1
        sparse_points_right = locs_2d_right_fg[indices_to_keep] * scale2
        
        # (x, y) 형식으로 변환
        coords1 = [(int(p[1]), int(p[0])) for p in sparse_points_left.numpy()]
        coords2 = [(int(p[1]), int(p[0])) for p in sparse_points_right.numpy()]
        
        print(f"매칭 완료: 전경 패치 {len(locs_2d_left_fg)}개 중 {len(coords1)}개 선택")
        
        return coords1, coords2
    
    def _stratify_points(self, pts_2d: torch.Tensor, threshold: float) -> torch.Tensor:
        """노트북의 stratify_points 함수와 동일한 구현"""
        n = len(pts_2d)
        if n == 0:
            return torch.tensor([], dtype=torch.long)
        
        max_value = threshold + 1
        
        # L2 거리 계산
        pts_2d_sq_norms = torch.linalg.vector_norm(pts_2d, dim=1)
        pts_2d_sq_norms.square_()
        
        distances = self._compute_distances_l2(pts_2d, pts_2d, pts_2d_sq_norms, pts_2d_sq_norms)
        distances.fill_diagonal_(max_value)
        
        # 거리 마스크
        distances_mask = torch.empty((n, n), dtype=pts_2d.dtype, device=pts_2d.device)
        torch.le(distances, threshold, out=distances_mask)
        
        ones_vec = torch.ones(n, device=pts_2d.device, dtype=pts_2d.dtype)
        counts_vec = torch.mv(distances_mask, ones_vec)
        
        indices_mask = np.ones(n)
        
        # 노트북과 동일한 greedy 알고리즘
        while torch.any(counts_vec).item():
            index_max = torch.argmax(counts_vec).item()
            indices_mask[index_max] = 0
            distances[index_max, :] = max_value
            distances[:, index_max] = max_value
            torch.le(distances, threshold, out=distances_mask)
            torch.mv(distances_mask, ones_vec, out=counts_vec)
        
        indices_to_keep = np.nonzero(indices_mask > 0)[0]
        return torch.tensor(indices_to_keep, dtype=torch.long)
    
    @staticmethod
    def _compute_distances_l2(X, Y, X_squared_norm, Y_squared_norm):
        """L2 거리 계산"""
        distances = -2 * X @ Y.T
        distances.add_(X_squared_norm[:, None]).add_(Y_squared_norm[None, :])
        return distances
