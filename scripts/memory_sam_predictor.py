import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import NearestNeighbors

# Import DINOv3 for feature extraction
# Note: DINOv3 uses torch.hub instead of transformers
import torch.hub

# SAM2 module import - trying various paths
possible_paths = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Parent directory of the current script
    '/home/joongwon00/sam2',  # sam2 original repository path
    '/home/joongwon00/Memory_SAM',  # Memory_SAM project root
]

for path in possible_paths:
    if path not in sys.path:
        sys.path.append(path)

# Set main SAM2 package path as well
sam2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sam2")
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)

# Set PYTHONPATH environment variable
os.environ["PYTHONPATH"] = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}:/home/joongwon00/sam2"

print(f"Python path: {sys.path}")

# Check Hydra module initialization
try:
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_module
    
    # Initialize Hydra if not already initialized
    if not GlobalHydra.instance().is_initialized():
        print("Initializing Hydra...")
        initialize_config_module("sam2", version_base="1.2")
        print("Hydra initialization complete")
except Exception as e:
    print(f"Error checking Hydra initialization: {e}")

# Import SAM2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# Import memory system and DINOv3 matcher
from scripts.memory_system import MemorySystem
from scripts.dinov3_matcher import Dinov3Matcher
from scripts.image_preprocessor import ImagePreprocessor
from scripts.matching_engine import MatchingEngine
from scripts.config import AppConfig
from scripts.logger import get_logger

class MemorySAMPredictor:
    """SAM2 with DINOv2 and a memory system for intelligent segmentation"""
    
    def __init__(self, 
                model_type: str = "hiera_l", 
                checkpoint_path: str = None,
                dinov3_model: str = "dinov3_vitb16",
                dinov3_matching_repo: str = "facebookresearch/dinov3",
                dinov3_matching_model: str = "dinov3_vitb16",
                memory_dir: str = "memory",
                results_dir: str = "results",
                device: str = "cuda",
                use_sparse_matching: bool = True):
        """
        Initialize Memory SAM predictor
        
        Args:
            model_type: SAM2 model type to use ("hiera_b+", "hiera_l", "hiera_s", "hiera_t")
            checkpoint_path: Path to SAM2 checkpoint
            dinov3_model: DINOv3 model name (torch.hub)
            dinov3_matching_repo: DINOv3 repository for sparse matching
            dinov3_matching_model: DINOv3 model for sparse matching
            memory_dir: Memory system directory
            results_dir: Directory to save results
            device: Device to use ("cuda" or "cpu")
            use_sparse_matching: Whether to use sparse matching
        """
        # Resizing options
        self.resize_images = False
        self.resize_scale = 1.0
        
        # Clustering hyperparameters
        self.similarity_threshold = 0.8
        self.background_weight = 0.3
        self.skip_clustering = False
        self.hybrid_clustering = False  # Hybrid clustering (foreground and background together)
        
        # 최대 포인트 수 설정
        self.max_positive_points = 10  # 최대 전경 포인트 수
        self.max_negative_points = 5   # 최대 배경 포인트 수
        # 전경 KMeans 중심 선택 옵션
        self.use_positive_kmeans = False
        self.positive_kmeans_clusters = 5
        
        # Set seeds for deterministic behavior
        try:
            import random
            np.random.seed(42)
            random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception as _e:
            print(f"Seed setup failed: {_e}")

        # Set device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            # Enable TF32 (Ampere GPU and newer)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = torch.device("cpu")
        
        print(f"Using device: {self.device}")
        
        # Find checkpoint path
        if checkpoint_path is None:
            # Mapping based on model type
            model_name_map = {
                "hiera_b+": ["base_plus", "b+"],
                "hiera_l": ["large", "l"],
                "hiera_s": ["small", "s"],
                "hiera_t": ["tiny", "t"]
            }
            
            # List of checkpoint file name patterns
            checkpoint_patterns = [
                f"sam2_{model_type}.pt",
                f"sam2.1_{model_type}.pt"
            ]
            
            # Generate additional patterns
            if model_type in model_name_map: # Check if model_type is a valid key
                for variant in model_name_map[model_type]:
                    checkpoint_patterns.extend([
                        f"sam2_hiera_{variant}.pt",
                        f"sam2.1_hiera_{variant}.pt"
                    ])
            
            # Checkpoint directory
            checkpoint_dir = os.path.join('/home/joongwon00/sam2', 'checkpoints')
            
            # Find checkpoint file matching pattern
            for pattern in checkpoint_patterns:
                path = os.path.join(checkpoint_dir, pattern)
                if os.path.exists(path):
                    checkpoint_path = path
                    break
            
            if checkpoint_path is None:
                raise FileNotFoundError(f"Checkpoint file not found. Attempted patterns: {checkpoint_patterns}")
        
        print(f"Checkpoint path: {checkpoint_path}")
        
        # SAM2 project root path
        sam2_root = '/home/joongwon00/sam2'
        
        # Possible Hydra config file names
        possible_config_names = [
            # SAM2.1 config
            f"configs/sam2.1/sam2.1_hiera_{model_type.replace('hiera_', '')}",
            # SAM2 original config
            f"configs/sam2/sam2_hiera_{model_type.replace('hiera_', '')}",
            # Absolute path
            f"/home/joongwon00/sam2/configs/sam2.1/sam2.1_hiera_{model_type.replace('hiera_', '')}.yaml",
            f"/home/joongwon00/sam2/configs/sam2/sam2_hiera_{model_type.replace('hiera_', '')}.yaml"
        ]
        
        # Try each name
        config_file = possible_config_names[0]  # Default value
        
        print(f"Attempting Hydra config files: {possible_config_names}")
        
        # Initialize SAM2
        all_failed = True
        for i, cfg in enumerate(possible_config_names):
            try:
                print(f"[{i+1}/{len(possible_config_names)}] Trying {cfg}...")
                self.sam_model = build_sam2(cfg, checkpoint_path, device=self.device)
                config_file = cfg  # Record successful config
                print(f"Success: {cfg}")
                all_failed = False
                break
            except Exception as e:
                print(f"Failed: {cfg} - {e}")
        
        if all_failed:
            print(f"All config file attempts failed: {possible_config_names}")
            print(f"Could be a Hydra config file issue. Trying alternative methods...")
            
            # Find absolute path directly and inform Hydra via environment variable
            config_paths = [
                # configs/sam2.1 path in SAM2 directory
                os.path.join('/home/joongwon00/sam2/configs/sam2.1', f'sam2.1_hiera_{model_type.replace("hiera_", "")}.yaml'),
                # configs/sam2 path in SAM2 directory
                os.path.join('/home/joongwon00/sam2/configs/sam2', f'sam2_hiera_{model_type.replace("hiera_", "")}.yaml'),
                # sam2/configs/sam2.1 path in SAM2 directory
                os.path.join('/home/joongwon00/sam2/sam2/configs/sam2.1', f'sam2.1_hiera_{model_type.replace("hiera_", "")}.yaml'),
                # sam2/configs/sam2 path in SAM2 directory
                os.path.join('/home/joongwon00/sam2/sam2/configs/sam2', f'sam2_hiera_{model_type.replace("hiera_", "")}.yaml')
            ]
            
            # Find existing config file
            config_path = None
            for path in config_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                raise FileNotFoundError(f"Config file not found. Attempted paths: {config_paths}")
            
            print(f"Config file path: {config_path}")
            
            # Call build_sam2 with config_path directly (this might not work)
            # If file already exists, copy it to Hydra's default path
            try:
                # 1. Read config file content
                with open(config_path, 'r') as f:
                    config_content = f.read()
                
                # 2. Create configs/sam2_hiera_l.yaml file (where Hydra expects it)
                hydra_config_path = os.path.join('/home/joongwon00/sam2/configs', f'sam2_hiera_{model_type}.yaml')
                os.makedirs(os.path.dirname(hydra_config_path), exist_ok=True)
                
                with open(hydra_config_path, 'w') as f:
                    f.write(config_content)
                
                print(f"Copied config file to Hydra default path: {hydra_config_path}")
                
                # 3. Try using default Hydra path without specifying config file
                try:
                    self.sam_model = build_sam2(f"configs/sam2_hiera_{model_type}", checkpoint_path, device=self.device)
                    print("Successfully loaded config from Hydra default path")
                except Exception as e3:
                    print(f"Failed to load from Hydra default path: {e3}")
                    # 4. Try specifying direct path
                    try:
                        self.sam_model = build_sam2(config_path, checkpoint_path, device=self.device)
                        print(f"Successfully loaded config from absolute path: {config_path}")
                    except Exception as e4:
                        print(f"All attempts failed: {e4}")
                        print("Contact system administrator or check config file path structure.")
                        raise e4
            except Exception as e2:
                print(f"Config file processing failed: {e2}")
                print("Contact system administrator or check config file path structure.")
                raise
        self.predictor = SAM2ImagePredictor(self.sam_model)
        
        # Initialize DINOv3 for global feature extraction - 사전 훈련된 가중치 사용
        try:
            # Force largest available DINOv3 config from local weights (ViT-L/16)
            dinov3_model = "dinov3_vitl16"
            print(f"Loading DINOv3 model: {dinov3_model}")
            # DINOv3 모듈 직접 import
            import sys
            dinov3_path = "/home/joongwon00/dinov3"
            if dinov3_path not in sys.path:
                sys.path.append(dinov3_path)
            
            from dinov3.models.vision_transformer import DinoVisionTransformer
            
            # DINOv3 ViT-L/16 구성으로 모델 생성 (큰 모델)
            self.dinov3_model = DinoVisionTransformer(
                img_size=518,
                patch_size=16,
                embed_dim=1024,
                depth=24,
                num_heads=16,
                mlp_ratio=4,
                block_chunks=0
            )
            
            # 사전 훈련된 가중치 로드
            weights_path = "/home/joongwon00/memory-sam/dinov3_weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
            if os.path.exists(weights_path):
                print(f"Loading pretrained weights from: {weights_path}")
                state_dict = torch.load(weights_path, map_location="cpu")
                self.dinov3_model.load_state_dict(state_dict, strict=False)
                print("Pretrained weights loaded successfully")
            else:
                print("Warning: Pretrained weights not found, using random initialization")
            
            self.dinov3_model = self.dinov3_model.to(self.device)
            self.dinov3_model.eval()
            print("DINOv3 model initialized successfully")
        except Exception as e:
            print(f"Failed to load DINOv3 model: {e}")
            raise
        
        # Initialize DINOv3 matcher for sparse matching
        self.use_sparse_matching = use_sparse_matching
        if use_sparse_matching:
            try:
                print(f"Initializing DINOv3 Matcher: {dinov3_matching_repo}/{dinov3_matching_model}")
                self.dinov3_matcher = Dinov3Matcher(
                    repo_name=dinov3_matching_repo,
                    model_name=dinov3_matching_model,
                    device=device,
                    half_precision=True
                )
                print("DINOv3 Matcher initialized successfully")
            except Exception as e:
                print(f"Failed to initialize DINOv3 Matcher: {e}")
                self.use_sparse_matching = False
                print("Sparse matching disabled. Using global feature matching only.")
        
        # Initialize memory system
        self.memory = MemorySystem(memory_dir)
        
        # Create results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # State variables
        self.current_image = None
        self.current_image_path = None
        self.current_mask = None
        self.current_features = None
        self.current_patch_features = None
        self.current_grid_size = None
        self.current_resize_scale = None

        # Utilities
        self.image_preprocessor = ImagePreprocessor()
        self.matching_engine = MatchingEngine(self)

        # App config and logger
        self.config = AppConfig()
        # sync legacy flags with config
        self.resize_images = self.config.resize.enabled
        self.resize_scale = self.config.resize.scale
        self.similarity_threshold = self.config.matching.similarity_threshold
        self.background_weight = self.config.matching.background_weight
        self.skip_clustering = self.config.matching.skip_clustering
        self.hybrid_clustering = self.config.matching.hybrid_clustering
        self.max_positive_points = self.config.matching.max_positive_points
        self.max_negative_points = self.config.matching.max_negative_points
        self.use_positive_kmeans = self.config.matching.use_positive_kmeans
        self.positive_kmeans_clusters = self.config.matching.positive_kmeans_clusters
        self.log = get_logger("MemorySAMPredictor")
    
    def _resize_image_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        필요에 따라 이미지 크기를 조정합니다.

        Args:
            image: 원본 이미지 (numpy 배열)

        Returns:
            (조정된 이미지, 실제 적용된 리사이즈 비율) 튜플
        """
        return self.image_preprocessor.resize_image(image, self.resize_images, self.resize_scale)
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract global features from image using DINOv3
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            Extracted feature vector
        """
        # Convert to PIL image if numpy array
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Process image for DINOv3
        # DINOv3 expects tensor input, so convert PIL to tensor
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(image_pil).unsqueeze(0).to(self.device)
        
        # Extract features using DINOv3
        with torch.no_grad():
            try:
                features = self.dinov3_model.forward_features(input_tensor)
            except RuntimeError as e:
                if "CUDA out of memory" in str(e) and self.device.type == "cuda":
                    torch.cuda.empty_cache()
                    print("Retrying DINOv3 forward_features on CPU due to OOM...")
                    features = self.dinov3_model.cpu().forward_features(input_tensor.cpu())
                    self.dinov3_model = self.dinov3_model.to(self.device)
                else:
                    raise
            # Get CLS token (first token)
            cls_features = features['x_norm_clstoken'].cpu().numpy()
        
        combined_features = cls_features
        
        # L2 normalization
        norm = np.linalg.norm(combined_features[0])
        if norm > 0:
            normalized_features = combined_features[0] / norm
        else:
            normalized_features = combined_features[0]
            
        print(f"Extracted feature vector shape: {normalized_features.shape}, norm: {np.linalg.norm(normalized_features):.6f}")
        
        return normalized_features

    def extract_patch_features(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """
        Extract patch features from image using DINOv3 Matcher
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            (patch_features, grid_size, resize_scale) tuple
        """
        if not self.use_sparse_matching:
            raise ValueError("Sparse matching is disabled")
        
        # Prepare image for DINOv3 format
        image_tensor, grid_size, resize_scale = self.dinov3_matcher.prepare_image(image)
        
        # Extract patch features
        patch_features = self.dinov3_matcher.extract_features(image_tensor)
        
        # Normalize patch features (row-wise)
        normalized_patch_features = np.zeros_like(patch_features)
        for i in range(patch_features.shape[0]):
            norm = np.linalg.norm(patch_features[i])
            if norm > 0:
                normalized_patch_features[i] = patch_features[i] / norm
            else:
                normalized_patch_features[i] = patch_features[i]
        
        # Check feature quality
        feature_norms = np.linalg.norm(normalized_patch_features, axis=1)
        mean_norm = np.mean(feature_norms)
        std_norm = np.std(feature_norms)
        
        print(f"Normalized patch feature shape: {normalized_patch_features.shape}")
        print(f"Patch feature norm statistics - mean: {mean_norm:.6f}, std: {std_norm:.6f}")
        
        return normalized_patch_features, grid_size, resize_scale
    
    def generate_prompt(self, mask: np.ndarray, prompt_type: str = "points", original_size: Tuple[int, int] = None) -> Dict:
        """
        Generate prompt from mask (points or box)
        
        Args:
            mask: Input mask (H, W)
            prompt_type: Prompt type ("points" or "box")
            original_size: Original image size (H, W) - used if resized
            
        Returns:
            Prompt dictionary (e.g., {"points": ..., "labels": ...} or {"box": ...})
        """
        if mask is None:
            return {}

        # If mask is float type, convert to bool
        if mask.dtype == np.float32 or mask.dtype == np.float64:
            mask = mask > 0.5 # Threshold can be adjusted
            
        # If mask has values other than 0 or 1, binarize
        if not ((mask == 0) | (mask == 1)).all():
            mask = (mask > np.min(mask)).astype(np.uint8) # Or appropriate threshold

        # Mask validation
        if np.sum(mask) == 0:
            print("Warning: Empty mask passed to generate_prompt.")
            return self._create_default_prompt(mask, prompt_type)

        # --- 마스크 리사이징 로직 강화 ---
        # `original_size`는 현재 처리할 대상 이미지의 크기 (예: 512x288)
        # `mask`는 메모리에서 온 마스크 (예: 1920x1080)
        if original_size is not None and mask.shape != original_size:
            target_h, target_w = original_size
            mask_h, mask_w = mask.shape

            # 종횡비를 유지하면서 리사이즈
            scale = min(target_w / mask_w, target_h / mask_h) if mask_w > 0 and mask_h > 0 else 0
            new_w, new_h = int(mask_w * scale), int(mask_h * scale)
            
            if new_w > 0 and new_h > 0:
                resized_mask_inter = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                resized_mask_inter = np.zeros((new_h, new_w), dtype=np.uint8)

            # 타겟 크기에 맞게 패딩 추가
            mask_resized = np.zeros(original_size, dtype=np.uint8)
            pad_x = (target_w - new_w) // 2
            pad_y = (target_h - new_h) // 2
            mask_resized[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_mask_inter
            
        else:
            mask_resized = mask
            
        # If mask has values other than 0 or 1, binarize (check again after resizing)
        if not ((mask_resized == 0) | (mask_resized == 1)).all():
            mask_resized = (mask_resized > np.min(mask_resized)).astype(np.uint8)

        if prompt_type == "points":
            # Sample foreground/background points
            foreground_points = np.argwhere(mask_resized > 0)
            background_points = np.argwhere(mask_resized == 0)
            
            points = []
            labels = []
            
            # If too many foreground points, determine sampling approach based on skip_clustering
            if len(foreground_points) > 0:
                # Determine max number of points based on skip_clustering setting
                if hasattr(self, 'skip_clustering') and self.skip_clustering:
                    # 클러스터링을 건너뛸 때도 사용자가 설정한 값을 사용하되 최솟값은 보장
                    max_fg_points = max(self.max_positive_points if hasattr(self, 'max_positive_points') else 5, min(20, len(foreground_points)))
                    print(f"Using {max_fg_points} foreground points (skip_clustering=True, user-defined max: {self.max_positive_points if hasattr(self, 'max_positive_points') else 5})")
                else:
                    # Use user-defined max_positive_points with clustering
                    max_fg_points = self.max_positive_points if hasattr(self, 'max_positive_points') else 5
                    print(f"Using {max_fg_points} foreground points (with clustering)")
                    
                if len(foreground_points) > max_fg_points:
                    # 결정적 균등간격 샘플링
                    idx = np.linspace(0, len(foreground_points)-1, num=max_fg_points, dtype=int)
                    selected_fg_points = foreground_points[idx]
                else:
                    selected_fg_points = foreground_points
                
                # 선택된 전경 포인트를 먼저 추가해 후보 목록 생성
                for pt in selected_fg_points:
                    points.append([pt[1], pt[0]]) # x, y order
                    labels.append(1) # Foreground

                # 전경 KMeans 중심 선택: 특징 공간 기반 n개로 축소
                try:
                    if getattr(self, 'use_positive_kmeans', False) and getattr(self, 'current_patch_features', None) is not None:
                        # 현재 이미지의 패치 특징과 그리드 사용
                        grid_h, grid_w = self.current_grid_size
                        # 선택된 전경 포인트의 패치 인덱스 계산
                        pts_xy = np.array(points[-len(selected_fg_points):])  # 마지막에 추가된 전경들
                        # 이미지 좌표 -> 그리드 인덱스
                        gx = np.clip((pts_xy[:,0] * grid_w / mask_resized.shape[1]).astype(int), 0, grid_w-1)
                        gy = np.clip((pts_xy[:,1] * grid_h / mask_resized.shape[0]).astype(int), 0, grid_h-1)
                        feat_idx = gy * grid_w + gx
                        feat_idx = np.clip(feat_idx, 0, self.current_patch_features.shape[0]-1)
                        feats = self.current_patch_features[feat_idx]
                        # KMeans in feature space
                        from sklearn.cluster import KMeans
                        n_clusters = int(getattr(self, 'positive_kmeans_clusters', 5))
                        n_clusters = max(1, min(n_clusters, len(selected_fg_points)))
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        labels_k = kmeans.fit_predict(feats)
                        centers = kmeans.cluster_centers_
                        keep_indices = []
                        for k in range(n_clusters):
                            idxs = np.where(labels_k == k)[0]
                            if idxs.size == 0:
                                continue
                            d = np.linalg.norm(feats[idxs] - centers[k], axis=1)
                            keep_indices.append(int(idxs[np.argmin(d)]))
                        # 전경 포인트를 중심 포인트로 재구성
                        kept_pts = pts_xy[keep_indices]
                        # 기존에 추가했던 전경 포인트를 제거하고 중심만 다시 추가
                        # points/labels의 뒤에서 전경 개수만 제거
                        for _ in range(len(selected_fg_points)):
                            points.pop()
                            labels.pop()
                        for pt in kept_pts:
                            points.append([int(pt[0]), int(pt[1])])
                            labels.append(1)
                except Exception as e:
                    print(f"Positive feature KMeans selection failed: {e}")
            
            # Process background points similarly
            if len(background_points) > 0:
                # Also adjust background points based on skip_clustering
                if hasattr(self, 'skip_clustering') and self.skip_clustering:
                    # 클러스터링을 건너뛸 때도 사용자가 설정한 값을 사용하되 최솟값은 보장
                    max_bg_points = max(self.max_negative_points if hasattr(self, 'max_negative_points') else 3, min(10, len(background_points)))
                    print(f"Using {max_bg_points} background points (skip_clustering=True, user-defined max: {self.max_negative_points if hasattr(self, 'max_negative_points') else 3})")
                else:
                    max_bg_points = self.max_negative_points if hasattr(self, 'max_negative_points') else 3  # Use user-defined max_negative_points
                    print(f"Using {max_bg_points} background points (with clustering)")
                
                if len(background_points) > max_bg_points:
                    idx = np.linspace(0, len(background_points)-1, num=max_bg_points, dtype=int)
                    selected_bg_points = background_points[idx]
                else:
                    selected_bg_points = background_points
                
                for pt in selected_bg_points:
                    points.append([pt[1], pt[0]]) # x, y order
                    labels.append(0) # Background

            if not points: # Mask exists but no sampled points (e.g., too small mask)
                return self._create_default_prompt(mask_resized, prompt_type)
                
            # 저장: 마지막 프롬프트 포인트(특히 전경 포인트)를 시각화를 위해 보관
            try:
                pts_arr = np.array(points)
                lbl_arr = np.array(labels)
                self.last_prompt_points = pts_arr
                self.last_prompt_labels = lbl_arr
                self.last_fg_prompt_points = pts_arr[lbl_arr == 1] if len(pts_arr) > 0 else None
            except Exception as e:
                print(f"Storing last prompt points failed: {e}")
            
            return {"points": np.array(points), "labels": np.array(labels)}
        
        else:
            # If unsupported prompt type, return default or empty prompt
            print(f"Unsupported prompt type: {prompt_type}. Using default prompt.")
            return self._create_default_prompt(mask_resized, "points") # Default to points

    def _create_default_prompt(self, mask, prompt_type):
        """Create default prompt when mask is empty or prompt generation is difficult"""
        h, w = mask.shape[:2]
        if prompt_type == "points":
            # One foreground point in image center
            center_x, center_y = w // 2, h // 2
            # Check if point is inside mask (optional)
            # if mask[center_y, center_x] > 0:
            #     points = np.array([[center_x, center_y]])
            #     labels = np.array([1])
            # else: # If center is background, find another point or just use center
            points = np.array([[center_x, center_y]])
            labels = np.array([1]) # Assume foreground by default
            return {"points": points, "labels": labels}
        # Removed "box" related default prompt logic
        # elif prompt_type == "box":
        #     # Box covering entire image
        #     return {"box": np.array([0, 0, w, h])}
        return {} # Empty prompt

    def segment_with_sam(self, 
                        image: np.ndarray, 
                        prompt: Dict,
                        multimask_output: bool = True) -> Tuple[np.ndarray, float]:
        """
        Perform image segmentation using SAM model
        
        Args:
            image: Input image (H, W, C)
            prompt: Prompt dictionary (can include points, box, etc.)
            multimask_output: Whether to output multiple masks
            
        Returns:
            (Optimal mask, highest score) tuple
        """
        self.predictor.set_image(image)
        
        points = prompt.get("points")
        labels = prompt.get("labels")
        # box = prompt.get("box") # Removed box prompt usage

        # if box is not None:
        #     box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device).unsqueeze(0)
        # else:
        #     box_torch = None
        
        if points is not None and labels is not None:
            points_torch = torch.as_tensor(points, dtype=torch.float, device=self.device).unsqueeze(0)
            labels_torch = torch.as_tensor(labels, dtype=torch.int, device=self.device).unsqueeze(0)
        else:
            points_torch = None
            labels_torch = None
            
        # SAM model prediction
        # Modified to call without box prompt
        masks, scores, logits = self.predictor.predict(
            point_coords=points_torch,
            point_labels=labels_torch,
            # box=box_torch, # Removed box argument
            multimask_output=multimask_output
        )
        
        if masks is None or scores is None or len(scores) == 0:
            print("SAM prediction failed: Did not return masks or scores")
            # Return empty mask and 0 score or raise exception
            return np.zeros(image.shape[:2], dtype=bool), 0.0
            
        # If scores is NumPy array, convert to PyTorch tensor
        if isinstance(scores, np.ndarray):
            scores_tensor = torch.from_numpy(scores).to(self.device)
        else:
            scores_tensor = scores # Already a tensor, use as is
            
        # Select best score mask
        best_mask_idx = torch.argmax(scores_tensor)
        
        # Handle based on masks type
        if isinstance(masks, torch.Tensor):
            # If masks is PyTorch tensor
            best_mask = masks[0, best_mask_idx].cpu().numpy() 
        elif isinstance(masks, np.ndarray):
            # If masks is NumPy array (error point)
            # Indexing adjustment needed based on return shape of self.predictor.predict
            # If (num_masks, H, W) shape, masks[best_mask_idx] might be correct
            # Currently assuming (1, num_masks, H, W) or similar, using masks[0, best_mask_idx]
            if masks.ndim == 4: # Assume (B, N, H, W) NumPy array (B=1)
                 best_mask = masks[0, best_mask_idx]
            elif masks.ndim == 3: # Assume (N, H, W) NumPy array
                 best_mask = masks[best_mask_idx]
            else:
                # Specify error possibility or default handling for unexpected shape
                print(f"Warning: masks has unexpected shape: {masks.shape}")
                best_mask = masks[best_mask_idx] # Try by default
        else:
            # Exception handling: Unsupported type
            raise TypeError(f"Unsupported type for masks: {type(masks)}")
            
        best_score = scores_tensor[best_mask_idx].item()
        
        return best_mask, best_score
    
    def process_image(self, 
                     image_path: str, 
                     reference_path: str = None,
                     prompt_type: str = "points",
                     use_sparse_matching: bool = None,
                     match_background: bool = True,
                     skip_clustering: bool = False,
                     auto_add_to_memory: bool = False,
                     save_results: bool = True) -> Dict:
        """
        Process image using Memory SAM system
        
        Args:
            image_path: Input image path (file or folder)
            reference_path: Reference image path (optional)
            prompt_type: Type of prompt to generate ('points' or 'box')
            use_sparse_matching: Whether to use sparse matching (None uses initialization setting)
            match_background: Whether to match background area
            skip_clustering: Whether to skip clustering (True shows all matches)
            auto_add_to_memory: Whether to automatically add to memory after processing
            
        Returns:
            Dictionary containing processing results
        """
        # Update internal state with skip_clustering value passed at function call
        self.skip_clustering = skip_clustering
        print(f"[process_image] Updated self.skip_clustering to: {self.skip_clustering}")

        # Validate prompt_type and set default
        if prompt_type not in ["points"]:
            print(f"Warning: Invalid prompt_type '{prompt_type}'. Setting to default 'points'.")
            prompt_type = "points"
            
        # Determine whether to use sparse matching
        if use_sparse_matching is None:
            use_sparse_matching = self.use_sparse_matching
        
        # Load image - handle file or folder path
        image_path = Path(image_path)
        
        # Process folder or single image
        if image_path.is_dir():
            # Process a folder of images
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = sorted([f for f in image_path.glob('*') if f.suffix.lower() in valid_extensions])
            
            if not image_files:
                raise ValueError(f"No valid image files found in the folder {image_path}.")

            # Store all image paths for other modules to potentially use
            self.folder_image_paths = [str(f) for f in image_files]
            self.is_folder_processing = True

            all_results = []
            for i, file_path in enumerate(image_files):
                print(f"Processing image {i+1}/{len(image_files)}: {file_path.name}")
                try:
                    # Process each image individually
                    single_result = self.process_image(
                        image_path=str(file_path),
                        reference_path=reference_path,
                        prompt_type=prompt_type,
                        use_sparse_matching=use_sparse_matching,
                        match_background=match_background,
                        skip_clustering=skip_clustering,
                        auto_add_to_memory=auto_add_to_memory,
                        save_results=save_results
                    )
                    all_results.append(single_result)
                except Exception as e:
                    print(f"Failed to process {file_path.name}: {e}")
            
            # Return a dictionary indicating folder processing is complete, along with the list of results
            return {
                "is_folder": True,
                "results_list": all_results,
                "folder_path": str(image_path),
                "image_count": len(all_results)
            }
        
        else:
            # Process a single image
            self.is_folder_processing = False
            self.folder_image_paths = []
            
            # Load the original image from the provided path
            image_original = np.array(Image.open(str(image_path)).convert("RGB"))
            
            # Resize the image according to UI settings before processing
            image_processed, actual_resize_scale = self._resize_image_if_needed(image_original)

            # Call the internal processing method for the single image
            results = self._process_single_image(
                image_processed,
                str(image_path),
                reference_path, 
                prompt_type, 
                use_sparse_matching,
                match_background,
                save_results=save_results
            )
            
            # Add additional metadata for the single image result
            results["is_folder"] = False
            results["original_image"] = image_original
            results["processed_image_shape"] = image_processed.shape
            results["actual_resize_scale"] = actual_resize_scale

        # Add to memory if requested
        if auto_add_to_memory and "image" in results and "mask" in results and "features" in results:
            memory_id = self.memory.add_memory(
                results["image"], 
                results["mask"], 
                results["features"],
                results.get("patch_features"),
                results.get("grid_size"),
                results.get("resize_scale"),
                {"original_path": str(image_path)}
            )
            
            results["added_to_memory"] = True
            results["memory_id"] = memory_id
        
        return results
    
    def _process_single_image(self, 
                           image: np.ndarray,
                           image_path: str, 
                           reference_path: str = None,
                           prompt_type: str = "points",
                           use_sparse_matching: bool = None,
                           match_background: bool = True,
                           save_results: bool = True) -> Dict:
        """
        Process a single image (helper method for process_image)
        
        Args:
            image: 처리할 이미지 (이미 리사이즈됨)
            image_path: 원본 이미지 파일 경로
            reference_path: Reference image path (optional)
            prompt_type: Type of prompt to generate
            use_sparse_matching: Whether to use sparse matching
            match_background: Whether to match background area
            
        Returns:
            Dictionary containing processing results
        """
        self.current_image = image
        self.current_image_path = str(image_path)
        
        # Extract global features with DINOv2
        features = self.extract_features(image)
        self.current_features = features
        
        # Extract patch features if sparse matching is enabled
        patch_features = None
        grid_size = None
        resize_scale = None
        
        if use_sparse_matching and self.use_sparse_matching:
            try:
                patch_features, grid_size, resize_scale = self.extract_patch_features(image)
                self.current_patch_features = patch_features
                self.current_grid_size = grid_size
                self.current_resize_scale = resize_scale
                print(f"Patch feature extraction complete: shape {patch_features.shape}, grid size {grid_size}")
            except Exception as e:
                print(f"Patch feature extraction failed: {e}")
                print("Continuing with global features only.")
                use_sparse_matching = False
        
        reference_mask = None
        
        # Use reference image if provided
        if reference_path:
            print(f"Using reference image: {reference_path}")
            try:
                reference_img = np.array(Image.open(reference_path).convert("RGB"))
                reference_mask_path = Path(reference_path).with_suffix('.png')
                
                if reference_mask_path.exists():
                    reference_mask = np.array(Image.open(reference_mask_path))
                    print(f"Reference mask loaded: {reference_mask_path}, shape: {reference_mask.shape}, type: {reference_mask.dtype}")
                    
                    # Check if mask is grayscale
                    if reference_mask.ndim == 2:
                        print("Grayscale mask detected")
                    elif reference_mask.ndim == 3:
                        print(f"Color mask detected, channels: {reference_mask.shape[2]}")
                        # Convert multi-channel to single channel (use first channel)
                        if reference_mask.shape[2] >= 3:
                            reference_mask = reference_mask[:, :, 0]
                            print(f"Converted to first channel, new shape: {reference_mask.shape}")
                else:
                    print(f"Warning: Reference mask {reference_mask_path} not found")
                    reference_mask = None
            except Exception as e:
                print(f"Error processing reference image: {e}")
                reference_mask = None
        
        # Find similar images in memory or use reference
        if reference_mask is not None:
            # Generate prompt using reference mask
            prompt = self.generate_prompt(reference_mask, prompt_type)
            similar_items = []
        else:
            similar_items = []
            
            # Use sparse matching if enabled
            if use_sparse_matching and patch_features is not None:
                print(f"Finding similar items using sparse matching... (background matching: {match_background})")
                # In initial stage, mask doesn't exist yet, so only use reference_mask
                similar_items = self.memory.get_most_similar_sparse(
                    patch_features, 
                    mask=reference_mask if match_background and reference_mask is not None else None,
                    grid_size=grid_size, 
                    top_k=3,
                    match_background=match_background
                )
                
                if similar_items:
                    print(f"Found {len(similar_items)} items with sparse matching")
                else:
                    print("No items found with sparse matching. Falling back to global features.")
            
            # Use global features if sparse matching fails or is disabled
            if not similar_items:
                print("Finding similar items using global features...")
                similar_items = self.memory.get_most_similar(features, top_k=3, method="global")
            
            if similar_items:
                try:
                    # Get most matching memory item
                    best_item = similar_items[0]["item"]
                    item_data = self.memory.load_item_data(best_item["id"])
                    
                    # Add logging for skip_clustering value and selected best_item ID
                    print(f"[process_image] skip_clustering: {self.skip_clustering}, selected best_item ID: {best_item['id']}")
                    
                    print(f"Memory mask loaded: ID {best_item['id']}")
                    if "mask" in item_data:
                        mask_data = item_data["mask"]
                        print(f"Memory mask shape: {mask_data.shape}, type: {mask_data.dtype}")
                        
                        # Process multi-channel mask
                        if mask_data.ndim > 2:
                            print(f"Multi-channel memory mask detected. Shape: {mask_data.shape}")
                            # Convert RGB or RGBA mask to single channel
                            if mask_data.shape[2] >= 3:
                                # Check if all channels are identical
                                if np.array_equal(mask_data[:,:,0], mask_data[:,:,1]) and np.array_equal(mask_data[:,:,1], mask_data[:,:,2]):
                                    # If identical, use first channel
                                    mask_data = mask_data[:,:,0]
                                else:
                                    # Convert to grayscale
                                    mask_data = cv2.cvtColor(mask_data, cv2.COLOR_RGB2GRAY)
                            else:
                                # Single channel 3D mask
                                mask_data = mask_data[:,:,0]
                        
                        # Normalize mask
                        if mask_data.dtype != np.bool_:
                            # Convert to binary mask (0 or 1)
                            mask_data = (mask_data > 0).astype(np.uint8)
                        
                        print(f"Processed memory mask shape: {mask_data.shape}, type: {mask_data.dtype}, pixel sum: {np.sum(mask_data)}")
                        
                        # Always generate prompt based on the size of the currently processing (representative) image
                        prompt = self.generate_prompt(mask_data, prompt_type, original_size=image.shape[:2])
                    else:
                        print("Mask not found in memory item. Using default prompt")
                        raise KeyError("mask")
                except Exception as e:
                    print(f"Error processing memory mask: {e}")
                    # Use default prompt on error
                    prompt = self._create_default_prompt(image, prompt_type)
            else:
                # No memory items found, use default prompt
                prompt = self._create_default_prompt(image, prompt_type)
        
        # Segment with SAM2
        mask, score = self.segment_with_sam(image, prompt)
        self.current_mask = mask
        
        # Save results (optional)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_base_path = self.results_dir / f"result_{timestamp}"
        if save_results:
            result_base_path.mkdir(exist_ok=True, parents=True)

        # Create subfolders by type
        input_folder = result_base_path / "inputs"
        mask_folder = result_base_path / "masks"
        overlay_folder = result_base_path / "overlays"
        segment_folder = result_base_path / "segments"

        if save_results:
            input_folder.mkdir(exist_ok=True)
            mask_folder.mkdir(exist_ok=True)
            overlay_folder.mkdir(exist_ok=True)
            segment_folder.mkdir(exist_ok=True)
        
        # Image filename stem
        image_stem = Path(self.current_image_path).stem

        # Save original image
        if save_results:
            Image.fromarray(image).save(str(input_folder / f"{image_stem}_input.png"))
        
        # Save mask
        mask_img = (mask * 255).astype(np.uint8)
        if save_results:
            Image.fromarray(mask_img).save(str(mask_folder / f"{image_stem}_mask.png"))

        # Save segmented color image (apply mask on original image)
        if save_results:
            try:
                seg_color = image.copy()
                if seg_color.ndim == 2:
                    seg_color = cv2.cvtColor(seg_color, cv2.COLOR_GRAY2RGB)
                seg_color[~mask.astype(bool)] = 0
                Image.fromarray(seg_color).save(str(segment_folder / f"{image_stem}_segment.png"))
            except Exception as e:
                print(f"Error saving segmented color image, fallback to mask: {e}")
                Image.fromarray(mask_img).save(str(segment_folder / f"{image_stem}_segment.png"))
        
        # Save visualization (overlay)
        vis_img = self.visualize_mask(image, mask)
        if save_results:
            Image.fromarray(vis_img).save(str(overlay_folder / f"{image_stem}_overlay.png"))
        
        # Sparse matching visualization (if similar items exist)
        sparse_match_vis = None
        img1_points = None
        img2_points = None
        
        if use_sparse_matching and self.use_sparse_matching and similar_items and patch_features is not None:
            try:
                best_item = similar_items[0]["item"]
                item_data = self.memory.load_item_data(best_item["id"])
                
                if "image" in item_data and "mask" in item_data:
                    memory_image = item_data["image"]
                    memory_mask = item_data["mask"]
                    
                    # Preprocess memory mask
                    if memory_mask.ndim > 2:
                        print(f"Preprocessing multi-channel memory mask. Shape: {memory_mask.shape}")
                        # Convert RGB or RGBA mask to single channel
                        if memory_mask.shape[2] >= 3:
                            memory_mask = memory_mask[:,:,0]
                    
                    # Visualize sparse matches
                    sparse_match_vis, img1_points, img2_points = self.visualize_sparse_matches(
                        image1=image, image2=memory_image, mask1=mask, mask2=memory_mask, 
                        skip_clustering=self.skip_clustering,
                        match_background=match_background,
                        save_path=str(result_base_path / f"sparse_matches_{image_stem}.png") if save_results else None
                    )
            except Exception as e:
                print(f"Error visualizing sparse matches: {e}")
        
        # Prepare results dictionary
        results = {
            "image": image,
            "mask": mask,
            "score": float(score),
            "features": features,
            "image_path": self.current_image_path,
            "result_path": str(result_base_path),
            "visualization": vis_img,
            "timestamp": timestamp,
            "similar_items": similar_items
        }
        
        if patch_features is not None:
            results["patch_features"] = patch_features
            results["grid_size"] = grid_size
            results["resize_scale"] = resize_scale
        
        if sparse_match_vis is not None:
            results["sparse_match_visualization"] = sparse_match_vis
            results["img1_points"] = img1_points
            results["img2_points"] = img2_points
        
        return results
    
    def visualize_mask(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Visualize mask
        
        Args:
            image: Original image (RGB)
            mask: Binary mask (0 or 1)
            alpha: Mask transparency
            
        Returns:
            Visualized image
        """
        # Save original image size
        original_image_shape = image.shape[:2]
        
        # Input validation
        if image is None or mask is None:
            print("visualize_mask error: Image or mask is None.")
            # Return empty image
            return np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Check and adjust mask size
        if mask.shape[:2] != original_image_shape:
            print(f"Mask and image size mismatch detected: mask {mask.shape[:2]}, image {original_image_shape}")
            print(f"Adjusting mask size to match image size.")
            mask = cv2.resize(mask.astype(np.uint8), (original_image_shape[1], original_image_shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
        
        # Check if image is RGB
        if len(image.shape) != 3 or image.shape[2] != 3:
            # Convert grayscale image to RGB
            print(f"Converting grayscale image to RGB: {image.shape}")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Check mask dimensions
        if mask.ndim > 2:
            print(f"Multi-channel mask detected: {mask.shape}")
            # Convert multi-channel mask to first channel or grayscale
            if mask.shape[2] == 1:
                mask = mask[:,:,0]
            else:
                # Check if all channels are identical
                if mask.shape[2] >= 3 and np.array_equal(mask[:,:,0], mask[:,:,1]) and np.array_equal(mask[:,:,1], mask[:,:,2]):
                    mask = mask[:,:,0]
                else:
                    # Convert to grayscale
                    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Convert mask to binary mask
        binary_mask = mask > 0
        
        # Copy image and mask
        vis = image.copy()
        
        # Set mask color (blue)
        color_mask = np.zeros_like(image, dtype=np.uint8)
        color_mask[binary_mask] = [30, 144, 255]  # Red, Green, Blue
        
        # Blend mask and image
        cv2.addWeighted(color_mask, alpha, vis, 1 - alpha, 0, vis)
        
        # Extract mask contours
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)
        
        return vis
    
    def visualize_sparse_matches(self, image1: np.ndarray, image2: np.ndarray, 
                               mask1: Optional[np.ndarray] = None, 
                               mask2: Optional[np.ndarray] = None,
                               max_matches: int = 50,
                               save_path: Optional[str] = None,
                               skip_clustering: bool = False,
                               hybrid_clustering: bool = False,
                               match_background: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Visualize sparse matches between two images
        
        Args:
            image1: First image
            image2: Second image
            mask1: Mask for first image (optional)
            mask2: Mask for second image (optional)
            max_matches: Maximum number of matches to display
            save_path: Path to save result (optional)
            skip_clustering: Whether to skip clustering (True shows all matches)
            hybrid_clustering: Whether to use hybrid clustering (cluster by displacement)
            
        Returns:
            Tuple of visualized matching image, image1 points, image2 points
        """
        # image1: 현재 처리중인 이미지(리사이즈됨), image2: 메모리 이미지(원본 크기)
        
        # Add logging for skip_clustering and hybrid_clustering values
        print(f"[visualize_sparse_matches] Received skip_clustering: {skip_clustering}, hybrid_clustering: {hybrid_clustering}, match_background: {match_background}")
        
        # Determine effective clustering parameters
        # - If skip_clustering is True, we don't cluster at all
        # - If skip_clustering is False but hybrid_clustering is True, we use hybrid clustering
        # - If both are False, we use standard clustering
        use_skip_clustering = skip_clustering
        use_hybrid_clustering = hybrid_clustering and not skip_clustering
        
        if use_skip_clustering:
            print(f"Using all matched points without clustering (skip_clustering=True)")
            # Increase max_matches when skipping clustering to show more points
            max_matches = max(max_matches, 100)
        elif use_hybrid_clustering:
            print(f"Using hybrid clustering mode (cluster by displacement)")
        
        try:
            # 원본 이미지 크기 저장 (시각화를 위해)
            # image1은 UI에서 리사이즈된 이미지일 수 있으므로, 원본 크기 정보가 필요함
            # self.current_image가 UI 리사이즈가 적용된 이미지이므로, 이를 원본으로 간주
            original_shape1 = image1.shape
            original_shape2 = image2.shape
            
            # Check if masks are same size as images
            if mask1 is not None and (mask1.shape[0] != image1.shape[0] or mask1.shape[1] != image1.shape[1]):
                print(f"Resizing mask1: {mask1.shape} -> {image1.shape[:2]}")
                mask1 = cv2.resize(mask1.astype(np.uint8), (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            
            if mask2 is not None and (mask2.shape[0] != image2.shape[0] or mask2.shape[1] != image2.shape[1]):
                print(f"Resizing mask2: {mask2.shape} -> {image2.shape[:2]}")
                mask2 = cv2.resize(mask2.astype(np.uint8), (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            
            # 특징 추출을 위해 DINOv3 Matcher가 요구하는 크기(512px)로 리사이즈
            # 시각화는 원본 비율의 이미지에 수행
            image1_for_feature, _, _ = self.dinov3_matcher.prepare_image(image1)
            image2_for_feature, _, _ = self.dinov3_matcher.prepare_image(image2)
            
            # Grayscale conversion (improves keypoint extraction)
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if len(image1.shape) == 3 else image1
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY) if len(image2.shape) == 3 else image2
            
            # Extract or reuse patch features (reuse current image features to avoid recomputation)
            reuse1 = hasattr(self, 'current_image') and self.current_image is not None and \
                     image1.shape == self.current_image.shape and np.array_equal(image1, self.current_image) and \
                     hasattr(self, 'current_patch_features') and self.current_patch_features is not None
            if reuse1:
                patch_features1 = self.current_patch_features
                grid_size1 = self.current_grid_size
            else:
                patch_features1, grid_size1, _ = self.extract_patch_features(image1)

            reuse2 = hasattr(self, 'current_image') and self.current_image is not None and \
                     image2.shape == self.current_image.shape and np.array_equal(image2, self.current_image) and \
                     hasattr(self, 'current_patch_features') and self.current_patch_features is not None
            if reuse2:
                patch_features2 = self.current_patch_features
                grid_size2 = self.current_grid_size
            else:
                patch_features2, grid_size2, _ = self.extract_patch_features(image2)
            
            # Resize and convert mask
            if mask1 is not None:
                # Convert mask if not boolean type
                if not np.issubdtype(mask1.dtype, np.bool_):
                    if len(mask1.shape) == 3:
                        print(f"Multi-channel mask1 detected. Shape: {mask1.shape}")
                        # Convert multi-channel mask to grayscale then binarize
                        if mask1.shape[2] == 3 or mask1.shape[2] == 4:
                            mask1_gray = cv2.cvtColor(mask1, cv2.COLOR_RGB2GRAY)
                            mask1 = mask1_gray > 0
                        else:
                            mask1 = mask1[:,:,0] > 0
                    else:
                        mask1 = mask1.astype(bool)
                
                # Check if image and mask sizes match
                if mask1.shape[0] != image1.shape[0] or mask1.shape[1] != image1.shape[1]:
                    print(f"Adjusting mask1 size to match image1 size: {mask1.shape} -> {image1.shape[:2]}")
                    mask1 = cv2.resize(mask1.astype(np.uint8), (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                
                # Check mask pixel sum (for debugging)
                print(f"Mask1 stats: shape {mask1.shape}, type: {mask1.dtype}, pixel sum: {np.sum(mask1)}")
                
                # Resize mask to grid size
                resized_mask1 = cv2.resize(
                    mask1.astype(np.uint8), 
                    (grid_size1[1], grid_size1[0]), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                
                # Check mask pixel count
                mask1_pixel_count = np.sum(resized_mask1)
                print(f"Mask1 pixel count: {mask1_pixel_count}/{resized_mask1.size} ({mask1_pixel_count/resized_mask1.size*100:.2f}%)")
            else:
                # If no mask, use entire image
                resized_mask1 = np.ones((grid_size1[0], grid_size1[1]), dtype=bool)
            
            if mask2 is not None:
                # Convert mask if not boolean type
                if not np.issubdtype(mask2.dtype, np.bool_):
                    if len(mask2.shape) == 3:
                        print(f"Multi-channel mask2 detected. Shape: {mask2.shape}")
                        # Convert multi-channel mask to grayscale then binarize
                        if mask2.shape[2] == 3 or mask2.shape[2] == 4:
                            mask2_gray = cv2.cvtColor(mask2, cv2.COLOR_RGB2GRAY)
                            mask2 = mask2_gray > 0
                        else:
                            mask2 = mask2[:,:,0] > 0
                    else:
                        mask2 = mask2.astype(bool)
                
                # Check if image and mask sizes match
                if mask2.shape[0] != image2.shape[0] or mask2.shape[1] != image2.shape[1]:
                    print(f"Adjusting mask2 size to match image2 size: {mask2.shape} -> {image2.shape[:2]}")
                    mask2 = cv2.resize(mask2.astype(np.uint8), (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                
                # Check mask pixel sum (for debugging)
                print(f"Mask2 stats: shape {mask2.shape}, type: {mask2.dtype}, pixel sum: {np.sum(mask2)}")
                
                # Resize mask to grid size
                resized_mask2 = cv2.resize(
                    mask2.astype(np.uint8), 
                    (grid_size2[1], grid_size2[0]), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                
                # Check mask pixel count
                mask2_pixel_count = np.sum(resized_mask2)
                print(f"Mask2 pixel count: {mask2_pixel_count}/{resized_mask2.size} ({mask2_pixel_count/resized_mask2.size*100:.2f}%)")
            else:
                # If no mask, use entire image
                resized_mask2 = np.ones((grid_size2[0], grid_size2[1]), dtype=bool)
            
            # Add debugging info
            print(f"Image 1 shape: {image1.shape}, grid size 1: {grid_size1}")
            print(f"Image 2 shape: {image2.shape}, grid size 2: {grid_size2}")
            print(f"Patch features 1 shape: {patch_features1.shape}, patch features 2 shape: {patch_features2.shape}")
            
            # Separate foreground and background areas
            # Keypoint indices for foreground area
            fg_features1_indices = np.where(resized_mask1.flatten())[0]
            fg_features2_indices = np.where(resized_mask2.flatten())[0]
            
            # Check and filter index range
            fg_features1_indices = fg_features1_indices[fg_features1_indices < patch_features1.shape[0]]
            fg_features2_indices = fg_features2_indices[fg_features2_indices < patch_features2.shape[0]]
            
            # Keypoint indices for background area
            bg_features1_indices = np.where(~resized_mask1.flatten())[0]
            bg_features2_indices = np.where(~resized_mask2.flatten())[0]
            
            # Check and filter index range
            bg_features1_indices = bg_features1_indices[bg_features1_indices < patch_features1.shape[0]]
            bg_features2_indices = bg_features2_indices[bg_features2_indices < patch_features2.shape[0]]
            
            # Give higher priority to foreground keypoints
            similarity_threshold_fg = 0.8  # Default similarity threshold
            similarity_threshold_bg = 0.7  # Background area similarity threshold
            
            # If too few keypoints, lower similarity threshold and continue using mask
            if len(fg_features1_indices) < 5 or len(fg_features2_indices) < 5:
                print(f"Too few keypoints in foreground area: {len(fg_features1_indices)}, {len(fg_features2_indices)}")
                print("Maintaining mask area and lowering similarity threshold.")
                similarity_threshold_fg = 0.65  # Use lower similarity threshold
            elif len(fg_features1_indices) > 30 and len(fg_features2_indices) > 30:
                # If enough foreground keypoints, use stricter similarity threshold
                print(f"Sufficient foreground keypoints: {len(fg_features1_indices)}, {len(fg_features2_indices)}")
                similarity_threshold_fg = 0.85  # Use higher similarity threshold
                # Reduce background keypoint ratio
                similarity_threshold_bg = 0.75  # Higher background similarity threshold
            
            # Process foreground keypoints (select top points based on variance)
            fg_features1_masked = patch_features1[fg_features1_indices]
            fg_features2_masked = patch_features2[fg_features2_indices]
            
            # Process background keypoints (if background area exists)
            bg_features1_masked = np.array([])
            bg_features2_masked = np.array([])
            
            # Limit number of background keypoints (proportional to foreground keypoints)
            bg_limit = min(len(fg_features1_indices) // 2, 30)  # Half of foreground or max 30
            
            if len(bg_features1_indices) > 0 and len(bg_features2_indices) > 0:
                bg_features1_masked = patch_features1[bg_features1_indices]
                bg_features2_masked = patch_features2[bg_features2_indices]
                
                # Evaluate background keypoint quality (variance-based)
                bg_features1_variance = np.var(bg_features1_masked, axis=1)
                bg_features2_variance = np.var(bg_features2_masked, axis=1)
                
                # Select top background keypoints with high variance (limited number)
                bg_top_k1 = min(len(bg_features1_variance), bg_limit)
                bg_top_k2 = min(len(bg_features2_variance), bg_limit)
                
                bg_top_indices1 = np.argsort(bg_features1_variance)[-bg_top_k1:]
                bg_top_indices2 = np.argsort(bg_features2_variance)[-bg_top_k2:]
                
                bg_features1_filtered = bg_features1_masked[bg_top_indices1]
                bg_features2_filtered = bg_features2_masked[bg_top_indices2]
                
                # Maintain original index mapping
                bg_original_indices1 = bg_features1_indices[bg_top_indices1]
                bg_original_indices2 = bg_features2_indices[bg_top_indices2]
            
            # Evaluate foreground keypoint quality (variance-based) - select more foreground keypoints
            if len(fg_features1_masked) > 0 and len(fg_features2_masked) > 0:
                fg_features1_variance = np.var(fg_features1_masked, axis=1)
                fg_features2_variance = np.var(fg_features2_masked, axis=1)
                
                # Select top foreground keypoints with high variance (more discriminative features)
                # Select more keypoints in foreground (to better represent mask area)
                fg_top_k1 = min(len(fg_features1_variance), 200)  # Increased number of foreground keypoints
                fg_top_k2 = min(len(fg_features2_variance), 200)
                
                fg_top_indices1 = np.argsort(fg_features1_variance)[-fg_top_k1:]
                fg_top_indices2 = np.argsort(fg_features2_variance)[-fg_top_k2:]
                
                fg_features1_filtered = fg_features1_masked[fg_top_indices1]
                fg_features2_filtered = fg_features2_masked[fg_top_indices2]
                
                # Maintain original index mapping
                fg_original_indices1 = fg_features1_indices[fg_top_indices1]
                fg_original_indices2 = fg_features2_indices[fg_top_indices2]
                
                # Set max_matches value for _match_features based on skip_clustering (for foreground)
                current_max_matches_param_fg = 100000 if use_skip_clustering else 100

                # Match foreground keypoints
                fg_coords1, fg_coords2, fg_match_similarities = self._match_features(
                    fg_features1_filtered, fg_features2_filtered,
                    fg_original_indices1, fg_original_indices2,
                    grid_size1, grid_size2, 
                    image1.shape, image2.shape, # 원본(리사이즈된) 이미지 shape 전달
                    similarity_threshold=similarity_threshold_fg,
                    max_matches=current_max_matches_param_fg 
                )
            else:
                print("Skipping matching due to no foreground keypoints.")
                fg_coords1, fg_coords2, fg_match_similarities = [], [], []
            
            # Match background keypoints (if enabled)
            bg_coords1, bg_coords2, bg_match_similarities = [], [], []
            # bg_limit variable should be defined earlier. Check its definition at top of visualize_sparse_matches function.
            # Definition: bg_limit = min(len(fg_features1_indices) // 2, 30)
            if match_background and len(bg_features1_indices) > 0 and len(bg_features2_indices) > 0 and 'bg_original_indices1' in locals() and 'bg_original_indices2' in locals(): # Check if bg_features1_filtered etc. are ready
                # Set max_matches value for _match_features based on skip_clustering (for background)
                current_max_matches_param_bg = 50000 if use_skip_clustering else bg_limit 

                bg_coords1, bg_coords2, bg_match_similarities = self._match_features(
                    bg_features1_filtered, bg_features2_filtered, # These variables might only be defined within above if block, so check
                    bg_original_indices1, 
                    bg_original_indices2,
                    grid_size1, grid_size2, 
                    image1.shape, image2.shape, # 원본(리사이즈된) 이미지 shape 전달
                    similarity_threshold=similarity_threshold_bg,
                    max_matches=current_max_matches_param_bg
                )
            
            # 하이브리드 모드: 전경/배경 근접 혼재(KNN 그룹)에 대한 겹침 제거 전처리
            if match_background and use_hybrid_clustering and len(fg_coords1) > 0 and len(bg_coords1) > 0:
                try:
                    (fg_coords1, fg_coords2, fg_match_similarities,
                     bg_coords1, bg_coords2, bg_match_similarities) = self._remove_fg_bg_overlap_knn_groups(
                        fg_coords1, fg_coords2, fg_match_similarities,
                        bg_coords1, bg_coords2, bg_match_similarities,
                        k_neighbors=4
                    )
                    print(f"[visualize_sparse_matches] Hybrid KNN-group overlap filtering applied. FG: {len(fg_coords1)}, BG: {len(bg_coords1)}")
                except Exception as e:
                    print(f"[visualize_sparse_matches] Error during hybrid KNN-group overlap filtering: {e}")

            # Determine whether to apply clustering
            if not use_skip_clustering:
                print(f"[visualize_sparse_matches] Applying clustering. Initial foreground matches: {len(fg_coords1)}, background matches: {len(bg_coords1)}")
                
                # Store original hybrid_clustering setting
                original_hybrid_clustering = getattr(self, 'hybrid_clustering', False)
                
                # Set hybrid_clustering flag to the value passed to this function
                self.hybrid_clustering = use_hybrid_clustering
                
                # Cluster foreground and background keypoints
                if len(fg_coords1) > 0:
                    fg_coords1, fg_coords2, fg_match_similarities = self._cluster_feature_points(
                        fg_coords1, fg_coords2, fg_match_similarities, n_clusters=2, is_foreground=True
                    )
                
                if len(bg_coords1) > 0:
                    bg_coords1, bg_coords2, bg_match_similarities = self._cluster_feature_points(
                        bg_coords1, bg_coords2, bg_match_similarities, n_clusters=1, is_foreground=False
                    )
                
                # Restore original hybrid_clustering setting
                self.hybrid_clustering = original_hybrid_clustering
                
                print(f"[visualize_sparse_matches] Clustering applied. Final foreground matches: {len(fg_coords1)}, background matches: {len(bg_coords1)}")
            else:
                print(f"[visualize_sparse_matches] Skipping clustering. Using all matches - FG: {len(fg_coords1)}, BG: {len(bg_coords1)}")
                # When skip_clustering is true, prevent too many points by similarity and user budgets
                # 1) Coarse limit by max_matches
                if len(fg_coords1) > max_matches * 3:
                    indices = np.argsort(fg_match_similarities)[::-1][:max_matches * 3]
                    fg_coords1 = [fg_coords1[i] for i in indices]
                    fg_coords2 = [fg_coords2[i] for i in indices]
                    fg_match_similarities = [fg_match_similarities[i] for i in indices]
                if len(bg_coords1) > max_matches // 2:
                    indices = np.argsort(bg_match_similarities)[::-1][:max_matches // 2]
                    bg_coords1 = [bg_coords1[i] for i in indices]
                    bg_coords2 = [bg_coords2[i] for i in indices]
                    bg_match_similarities = [bg_match_similarities[i] for i in indices]
                # 2) Enforce strict budgets from UI (max_positive_points/max_negative_points)
                fg_budget = int(getattr(self, 'max_positive_points', 50))
                bg_budget = int(getattr(self, 'max_negative_points', 25))
                if len(fg_coords1) > fg_budget:
                    indices = np.argsort(fg_match_similarities)[::-1][:fg_budget]
                    fg_coords1 = [fg_coords1[i] for i in indices]
                    fg_coords2 = [fg_coords2[i] for i in indices]
                    fg_match_similarities = [fg_match_similarities[i] for i in indices]
                if len(bg_coords1) > bg_budget:
                    indices = np.argsort(bg_match_similarities)[::-1][:bg_budget]
                    bg_coords1 = [bg_coords1[i] for i in indices]
                    bg_coords2 = [bg_coords2[i] for i in indices]
                    bg_match_similarities = [bg_match_similarities[i] for i in indices]
            
            # Combine foreground and background keypoints
            coords1 = fg_coords1 + bg_coords1
            coords2 = fg_coords2 + bg_coords2
            match_similarities = fg_match_similarities + bg_match_similarities

            # Geometric consistency filtering (RANSAC affine)
            try:
                if len(coords1) >= 4 and len(coords2) >= 4:
                    pts1 = np.array(coords1, dtype=np.float32)
                    pts2 = np.array(coords2, dtype=np.float32)
                    M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=2000, confidence=0.99)
                    if inliers is not None:
                        inliers = inliers.reshape(-1).astype(bool)
                        if inliers.sum() >= 3:
                            coords1 = [tuple(map(int, p)) for p in pts1[inliers]]
                            coords2 = [tuple(map(int, p)) for p in pts2[inliers]]
                            match_similarities = [match_similarities[i] for i, m in enumerate(inliers) if m]
                            print(f"[visualize_sparse_matches] RANSAC filtered: {inliers.sum()} / {len(inliers)} inliers")
            except Exception as e:
                print(f"[visualize_sparse_matches] RANSAC filtering error: {e}")
            
            # Print matching results with similarity stats
            def _stats(arr):
                if not arr:
                    return (0.0, 0.0)
                return (float(np.mean(arr)), float(np.std(arr)))
            m_mean, m_std = _stats(match_similarities)
            print(f"Foreground matches: {len(fg_coords1)}, background matches: {len(bg_coords1)}, total matches: {len(coords1)}; sim_mean={m_mean:.3f}, sim_std={m_std:.3f}")
            
            # Optional: crop black borders introduced by letterbox padding (for visualization only)
            def _crop_black_borders(img: np.ndarray):
                try:
                    if img.ndim == 3:
                        nonblack = np.any(img > 0, axis=2)
                    else:
                        nonblack = img > 0
                    rows = np.where(np.any(nonblack, axis=1))[0]
                    cols = np.where(np.any(nonblack, axis=0))[0]
                    if rows.size == 0 or cols.size == 0:
                        return img, (0, 0), img.shape[:2]
                    top, bottom = int(rows[0]), int(rows[-1] + 1)
                    left, right = int(cols[0]), int(cols[-1] + 1)
                    cropped = img[top:bottom, left:right]
                    return cropped, (left, top), (cropped.shape[0], cropped.shape[1])
                except Exception:
                    return img, (0, 0), img.shape[:2]

            img1_vis_src, (off1x, off1y), _ = _crop_black_borders(image1)
            img2_vis_src, (off2x, off2y), _ = _crop_black_borders(image2)

            # Adjust coordinates to cropped visualization space
            def _shift_coords(coords, offx, offy):
                out = []
                for (x, y) in coords:
                    out.append((max(0, x - offx), max(0, y - offy)))
                return out

            coords1 = _shift_coords(coords1, off1x, off1y)
            coords2 = _shift_coords(coords2, off2x, off2y)

            # Place images side-by-side (with cropped views)
            h1, w1 = img1_vis_src.shape[:2]
            h2, w2 = img2_vis_src.shape[:2]
            
            # Match height of both images
            max_h = max(h1, h2)
            image1_resized_vis = np.zeros((max_h, w1, 3), dtype=np.uint8)
            image2_resized_vis = np.zeros((max_h, w2, 3), dtype=np.uint8)
            
            image1_resized_vis[:h1, :w1] = img1_vis_src if len(img1_vis_src.shape) == 3 else cv2.cvtColor(img1_vis_src, cv2.COLOR_GRAY2RGB)
            image2_resized_vis[:h2, :w2] = img2_vis_src if len(img2_vis_src.shape) == 3 else cv2.cvtColor(img2_vis_src, cv2.COLOR_GRAY2RGB)
            
            # Combined image
            vis_img = np.hstack([image1_resized_vis, image2_resized_vis])
            
            # Visualize matches
            for i, ((x1, y1), (x2, y2), sim) in enumerate(zip(coords1, coords2, match_similarities)):
                # Color based on similarity (greener for higher, redder for lower)
                color = (
                    int(255 * (1 - sim)),  # B
                    int(255 * sim),        # G
                    0                      # R
                )
                
                # Point on first image
                cv2.circle(vis_img, (x1, y1), 5, color, -1)
                
                # Point on second image (add width of first image to x coordinate)
                cv2.circle(vis_img, (x2 + w1, y2), 5, color, -1)
                
                # Connecting line
                cv2.line(vis_img, (x1, y1), (x2 + w1, y2), color, 1)
            
            # Visualize keypoints on individual images
            img1_points = img1_vis_src.copy() if len(img1_vis_src.shape) == 3 else cv2.cvtColor(img1_vis_src, cv2.COLOR_GRAY2RGB)
            img2_points = img2_vis_src.copy() if len(img2_vis_src.shape) == 3 else cv2.cvtColor(img2_vis_src, cv2.COLOR_GRAY2RGB)
            
            # Visualize original mask (translucent overlay)
            if mask1 is not None:
                mask1_overlay = img1_points.copy()
                mask1_color = np.zeros_like(mask1_overlay)
                # Crop mask1 with same offsets if sizes match original
                try:
                    if mask1.shape[:2] == image1.shape[:2]:
                        mask1 = mask1[off1y:off1y+h1, off1x:off1x+w1]
                except Exception:
                    pass
                mask1_color[mask1 > 0] = [0, 200, 0]
                # Translucently overlay mask area
                cv2.addWeighted(mask1_color, 0.3, mask1_overlay, 0.7, 0, img1_points)
                
            if mask2 is not None:
                mask2_overlay = img2_points.copy()
                mask2_color = np.zeros_like(mask2_overlay)
                try:
                    if mask2.shape[:2] == image2.shape[:2]:
                        mask2 = mask2[off2y:off2y+h2, off2x:off2x+w2]
                except Exception:
                    pass
                mask2_color[mask2 > 0] = [0, 200, 0]
                # Translucently overlay mask area
                cv2.addWeighted(mask2_color, 0.3, mask2_overlay, 0.7, 0, img2_points)
            
            # Display foreground keypoints (blueish)
            for i, ((x1, y1), sim) in enumerate(zip(fg_coords1, fg_match_similarities)):
                # Change color intensity based on similarity
                intensity = int(255 * sim)
                color = (intensity, 0, 0)  # Display foreground in blue (BGR)
                cv2.circle(img1_points, (x1, y1), 5, color, -1)
                # Also display similarity value
                if i < 5:  # Display for top 5 only
                    cv2.putText(img1_points, f"{sim:.2f}", (x1+5, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            for i, ((x2, y2), sim) in enumerate(zip(fg_coords2, fg_match_similarities)):
                # Change color intensity based on similarity
                intensity = int(255 * sim)
                color = (intensity, 0, 0)  # Display foreground in blue (BGR)
                cv2.circle(img2_points, (x2, y2), 5, color, -1)
                # Also display similarity value
                if i < 5:  # Display for top 5 only
                    cv2.putText(img2_points, f"{sim:.2f}", (x2+5, y2-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Highlight KMeans centers over the selected foreground points only (not all points)
            try:
                from sklearn.cluster import KMeans
                if len(fg_coords1) >= 1:
                    pts1 = np.array(fg_coords1, dtype=np.float32)
                    pts2 = np.array(fg_coords2, dtype=np.float32) if len(fg_coords2) == len(fg_coords1) else None
                    k = int(getattr(self, 'positive_kmeans_clusters', 4))
                    k = max(1, min(k, len(pts1)))
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels_k = kmeans.fit_predict(pts1)
                    centers1 = kmeans.cluster_centers_.astype(int)
                    # centers2: 각 클러스터의 대응 포인트 평균 위치
                    if pts2 is not None:
                        centers2 = []
                        for c in range(k):
                            idxs = np.where(labels_k == c)[0]
                            if idxs.size == 0:
                                continue
                            cxy = np.mean(pts2[idxs], axis=0)
                            centers2.append(cxy.astype(int))
                    else:
                        centers2 = []
                    center_highlight_color = (0, 255, 255)
                    for (cx1, cy1) in centers1:
                        cv2.circle(img1_points, (int(cx1), int(cy1)), 9, center_highlight_color, 3)
                    for (cx2, cy2) in centers2:
                        cv2.circle(img2_points, (int(cx2), int(cy2)), 9, center_highlight_color, 3)
                    print(f"KMeans center highlight: {k} centers")
            except Exception as e:
                print(f"Center highlight failed: {e}")
            
            # Display background keypoints (reddish)
            for i, ((x1, y1), sim) in enumerate(zip(bg_coords1, bg_match_similarities)):
                # Change color intensity based on similarity
                intensity = int(255 * sim)
                color = (0, 0, intensity)  # Display background in red (BGR)
                cv2.circle(img1_points, (x1, y1), 5, color, -1)
            
            for i, ((x2, y2), sim) in enumerate(zip(bg_coords2, bg_match_similarities)):
                # Change color intensity based on similarity
                intensity = int(255 * sim)
                color = (0, 0, intensity)  # Display background in red (BGR)
                cv2.circle(img2_points, (x2, y2), 5, color, -1)
            
            # Add info text to image
            clustering_mode = "Skip clustering" if use_skip_clustering else "Hybrid clustering" if use_hybrid_clustering else "Standard clustering"
            cv2.putText(img1_points, f"{clustering_mode}: FG: {len(fg_coords1)}, BG: {len(bg_coords1)}", 
                     (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img2_points, f"{clustering_mode}: FG: {len(fg_coords2)}, BG: {len(bg_coords2)}", 
                     (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save result
            if save_path:
                cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                
                # Also save individual images
                save_dir = os.path.dirname(save_path)
                base_name = os.path.splitext(os.path.basename(save_path))[0]
                cv2.imwrite(os.path.join(save_dir, f"{base_name}_img1_points.png"), 
                           cv2.cvtColor(img1_points, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_dir, f"{base_name}_img2_points.png"), 
                           cv2.cvtColor(img2_points, cv2.COLOR_RGB2BGR))
            
            return vis_img, img1_points, img2_points
            
        except Exception as e:
            print(f"Error generating sparse matching visualization: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty image on error
            empty_img = np.ones((400, 800, 3), dtype=np.uint8) * 255
            cv2.putText(empty_img, f"Matching visualization error: {str(e)}", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return empty_img, empty_img.copy(), empty_img.copy()
    
    def visualize_raw_sparse_matches(self, image1: np.ndarray, image2: np.ndarray, 
                                 mask1: Optional[np.ndarray] = None, 
                                 mask2: Optional[np.ndarray] = None,
                                 max_matches: int = 50,
                                 save_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Visualize all sparse matches between two images without clustering
        
        Args:
            image1: First image
            image2: Second image
            mask1: Mask for first image (optional)
            mask2: Mask for second image (optional)
            max_matches: Maximum number of matches to display
            save_path: Path to save result (optional)
            
        Returns:
            Tuple of visualized matching image, image1 points, image2 points
        """
        print("Visualizing all sparse matches without clustering.")
        return self.visualize_sparse_matches(
            image1, image2, mask1, mask2, max_matches, save_path, skip_clustering=True
        )
    
    def _match_features(self, features1, features2, original_indices1, original_indices2, 
                      grid_size1, grid_size2, image1_shape, image2_shape, 
                      similarity_threshold=0.7, max_matches=100, ratio_test=0.8, mutual_check=True):
        """
        Helper function to perform matching between two feature sets
        
        Args:
            features1: First feature set
            features2: Second feature set
            original_indices1: Original indices of first features
            original_indices2: Original indices of second features
            grid_size1: Grid size of first image
            grid_size2: Grid size of second image
            image1_shape: Shape of first image
            image2_shape: Shape of second image
            similarity_threshold: Similarity threshold
            max_matches: Maximum number of matches
            
        Returns:
            (coords1, coords2, match_similarities) tuple
        """
        if len(features1) == 0 or len(features2) == 0:
            print("Skipping matching due to no keypoints.")
            return [], [], []
        
        # Normalize features
        features1_norm = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
        features2_norm = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
        
        # Calculate cosine similarity
        similarities = np.matmul(features1_norm, features2_norm.T)
        
        # Find top-2 matches per row for Lowe's ratio test
        # argsort descending per row
        top2_idx = np.argpartition(similarities, -2, axis=1)[:, -2:]
        # sort the two so that first is the best
        row_indices = np.arange(similarities.shape[0])
        a, b = top2_idx[:, 0], top2_idx[:, 1]
        sim_a = similarities[row_indices, a]
        sim_b = similarities[row_indices, b]
        # ensure first is max
        swap_mask = sim_b > sim_a
        best_idx = np.where(swap_mask, b, a)
        second_idx = np.where(swap_mask, a, b)
        best_sim = similarities[row_indices, best_idx]
        second_sim = similarities[row_indices, second_idx]

        # ratio test: best >= similarity_threshold and best/second >= ratio_test
        # if second_sim is 0, accept when best_sim >= threshold
        ratio = np.divide(best_sim, np.maximum(second_sim, 1e-6))
        pass_mask = (best_sim >= similarity_threshold) & (ratio >= ratio_test)

        # mutual nearest neighbor check (optional)
        if mutual_check and np.any(pass_mask):
            # for each column, compute its best row
            col_best_rows = np.argmax(similarities, axis=0)
            mutual_mask = col_best_rows[best_idx] == row_indices
            pass_mask = pass_mask & mutual_mask

        best_matches = [(int(i), int(j), float(best_sim[i])) for i, j in zip(row_indices[pass_mask], best_idx[pass_mask])]
        
        # Sort by similarity
        best_matches.sort(key=lambda x: x[2], reverse=True)
        
        # Limit maximum number of matches
        best_matches = best_matches[:max_matches]
        
        # Convert coordinates
        coords1 = []
        coords2 = []
        match_similarities = []
        
        for i, j, sim in best_matches:
            # Convert to original index
            feat1_idx = original_indices1[i]
            feat2_idx = original_indices2[j]
            
            # Convert to grid coordinates
            y1, x1 = np.unravel_index(feat1_idx, (grid_size1[0], grid_size1[1]))
            y2, x2 = np.unravel_index(feat2_idx, (grid_size2[0], grid_size2[1]))
            
            # 특징 그리드 좌표 -> 이미지 좌표로 변환
            # grid_size는 특징 추출 시 사용된 이미지의 그리드 크기
            # image_shape는 현재 시각화에 사용되는 (리사이즈된) 이미지의 크기
            img1_x = int((x1 + 0.5) * (image1_shape[1] / grid_size1[1]))
            img1_y = int((y1 + 0.5) * (image1_shape[0] / grid_size1[0]))
            img2_x = int((x2 + 0.5) * (image2_shape[1] / grid_size2[1]))
            img2_y = int((y2 + 0.5) * (image2_shape[0] / grid_size2[0]))
            
            coords1.append((img1_x, img1_y))
            coords2.append((img2_x, img2_y))
            match_similarities.append(sim)
        
        print(f"Found {len(coords1)} matches with similarity threshold {similarity_threshold}")
        return coords1, coords2, match_similarities
    
    def _cluster_feature_points(self, coords1, coords2, similarities, n_clusters=2, is_foreground=True):
        """
        Helper function to cluster feature points
        
        Args:
            coords1: List of coordinates for first image
            coords2: List of coordinates for second image
            similarities: List of similarities
            n_clusters: Number of clusters
            is_foreground: Whether this is foreground (True) or background (False) clustering
            
        Returns:
            (clustered_coords1, clustered_coords2, clustered_similarities) tuple
        """
        if len(coords1) <= n_clusters or n_clusters <= 0:
            # If too few matches, return as is
            print(f"Skipping clustering due to too few matches: {len(coords1)}")
            return coords1, coords2, similarities
        
        try:
            from sklearn.cluster import KMeans
            
            # Convert coordinates to numpy array
            coords1_array = np.array(coords1)
            coords2_array = np.array(coords2)
            similarities_array = np.array(similarities)
            
            # Dynamically adjust number of clusters based on match count
            # More clusters for more matches
            if len(coords1) > 30:
                adjusted_n_clusters = min(n_clusters * 2, len(coords1) // 5)
            else:
                adjusted_n_clusters = min(n_clusters, len(coords1) // 2)
                
            # Ensure minimum number of clusters
            adjusted_n_clusters = max(adjusted_n_clusters, 2)
            
            print(f"Clustering {len(coords1)} {'foreground' if is_foreground else 'background'} matches into {adjusted_n_clusters} clusters")
            
            # Apply K-means clustering
            # Determine features for clustering
            if hasattr(self, 'hybrid_clustering') and self.hybrid_clustering:
                # Hybrid clustering: Use only coordinate differences to cluster
                # This treats foreground and background points together
                print("Using hybrid clustering (foreground and background together)")
                features = np.column_stack((
                    coords1_array - coords2_array,  # Coordinate differences (dx, dy)
                    similarities_array.reshape(-1, 1) * 100  # Weight similarity
                ))
            else:
                # Standard clustering: Use all coordinates and similarity
                features = np.column_stack((
                    coords1_array,  # x1, y1 coordinates
                    coords2_array,  # x2, y2 coordinates
                    similarities_array.reshape(-1, 1) * 100  # Weight similarity
                ))
            
            kmeans = KMeans(n_clusters=adjusted_n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features)
            
            # Select keypoint with highest similarity from each cluster
            clustered_coords1 = []
            clustered_coords2 = []
            clustered_similarities = []
            
            # 전역 포인트 예산 (전경/배경 구분)
            max_points_limit = int(self.max_positive_points if is_foreground else self.max_negative_points)

            for i in range(adjusted_n_clusters):
                # 예산 소진 시 중단
                if len(clustered_coords1) >= max_points_limit:
                    break
                cluster_indices = np.where(cluster_labels == i)[0]
                if len(cluster_indices) > 0:
                    # Select keypoint with highest similarity within cluster
                    best_idx_in_cluster = cluster_indices[np.argmax(similarities_array[cluster_indices])]
                    if len(clustered_coords1) < max_points_limit:
                        clustered_coords1.append(coords1[best_idx_in_cluster])
                        clustered_coords2.append(coords2[best_idx_in_cluster])
                        clustered_similarities.append(similarities[best_idx_in_cluster])
                    
                    # 추가 포인트 선택 로직 - max_positive_points/max_negative_points 제한 고려
                    # 전달받은 is_foreground 파라미터를 사용
                    # 전역 예산(전체 포인트 수 한도) 소진 전까지 현재 클러스터에서 추가 포인트 선별
                    remaining_budget = max(0, max_points_limit - len(clustered_coords1))
                    if remaining_budget > 0 and len(cluster_indices) > 1:
                        sorted_indices = np.argsort(similarities_array[cluster_indices])[::-1]
                        added = 0
                        # 0번째는 이미 best로 사용했으므로 1부터 시작
                        for idx_pos in range(1, len(sorted_indices)):
                            if added >= remaining_budget:
                                break
                            idx = sorted_indices[idx_pos]
                            if similarities_array[cluster_indices[idx]] > 0.75:
                                clustered_coords1.append(coords1[cluster_indices[idx]])
                                clustered_coords2.append(coords2[cluster_indices[idx]])
                                clustered_similarities.append(similarities[cluster_indices[idx]])
                                added += 1
            
            print(f"Clustered {'foreground' if is_foreground else 'background'} keypoints into {len(clustered_coords1)}.")
            
            # Sort by similarity
            if clustered_coords1:
                sorted_indices = np.argsort(clustered_similarities)[::-1]
                clustered_coords1 = [clustered_coords1[i] for i in sorted_indices]
                clustered_coords2 = [clustered_coords2[i] for i in sorted_indices]
                clustered_similarities = [clustered_similarities[i] for i in sorted_indices]
                
            return clustered_coords1, clustered_coords2, clustered_similarities
            
        except Exception as e:
            print(f"Error during keypoint clustering: {e}")
            # If clustering fails, select top n_clusters only
            if len(coords1) > n_clusters:
                sorted_indices = np.argsort(similarities)[::-1][:n_clusters * 2]
                coords1 = [coords1[i] for i in sorted_indices]
                coords2 = [coords2[i] for i in sorted_indices]
                similarities = [similarities[i] for i in sorted_indices]
                print(f"Selected top {len(coords1)} keypoints instead of clustering.")
            return coords1, coords2, similarities
    
    
    def _remove_fg_bg_overlap_knn_groups(self,
                                         fg_coords1, fg_coords2, fg_similarities,
                                         bg_coords1, bg_coords2, bg_similarities,
                                         k_neighbors: int = 4,
                                         max_remove_ratio: float = 0.6):
        """
        전경/배경 포인트를 통합하여 KNN 기반 근접 그룹을 형성하고, 그룹 내에 전경과 배경이 함께 존재하면 그룹 내 포인트를 제거.
        - image1 공간과 image2 공간 각각에서 처리 후, 제거 인덱스를 합집합으로 결합
        - k_neighbors는 3~4 권장
        - 제거 비율이 너무 크면 롤백
        """
        try:
            import numpy as np
            from sklearn.neighbors import NearestNeighbors

            if len(fg_coords1) == 0 or len(bg_coords1) == 0:
                return fg_coords1, fg_coords2, fg_similarities, bg_coords1, bg_coords2, bg_similarities

            fg_c1 = np.array(fg_coords1, dtype=np.float32)
            fg_c2 = np.array(fg_coords2, dtype=np.float32)
            bg_c1 = np.array(bg_coords1, dtype=np.float32)
            bg_c2 = np.array(bg_coords2, dtype=np.float32)

            def collect_groups(points: np.ndarray, labels: np.ndarray, k: int):
                if len(points) == 0:
                    return []
                k_eff = min(k, len(points))
                nn = NearestNeighbors(n_neighbors=k_eff)
                nn.fit(points)
                neigh_idx = nn.kneighbors(points, return_distance=False)
                groups = []
                for i in range(len(points)):
                    grp = set(neigh_idx[i].tolist())
                    grp.add(i)  # 자기 자신 포함
                    grp_labels = set(labels[list(grp)])
                    if 0 in grp_labels and 1 in grp_labels:
                        groups.append(grp)
                return groups

            # image1 공간
            combined_c1 = np.vstack([fg_c1, bg_c1])
            labels_c1 = np.array([1] * len(fg_c1) + [0] * len(bg_c1))
            groups_c1 = collect_groups(combined_c1, labels_c1, k_neighbors)

            # image2 공간
            combined_c2 = np.vstack([fg_c2, bg_c2])
            labels_c2 = np.array([1] * len(fg_c2) + [0] * len(bg_c2))
            groups_c2 = collect_groups(combined_c2, labels_c2, k_neighbors)

            # 제거 인덱스 합집합
            remove_indices = set()
            for g in groups_c1:
                remove_indices |= g
            for g in groups_c2:
                remove_indices |= g

            if not remove_indices:
                return fg_coords1, fg_coords2, fg_similarities, bg_coords1, bg_coords2, bg_similarities

            # FG/BG로 분리된 로컬 인덱스 계산
            fg_len = len(fg_coords1)
            fg_remove_local = sorted([i for i in remove_indices if i < fg_len])
            bg_remove_local = sorted([i - fg_len for i in remove_indices if i >= fg_len])

            # 안전장치: 제거 비율 제한
            if len(fg_remove_local) > max_remove_ratio * max(1, fg_len) or \
               len(bg_remove_local) > max_remove_ratio * max(1, len(bg_coords1)):
                print("[hybrid_knn_groups] Removal ratio too high, rollback.")
                return fg_coords1, fg_coords2, fg_similarities, bg_coords1, bg_coords2, bg_similarities

            def filter_by_indices(lst, remove_set):
                rem = set(remove_set)
                return [v for i, v in enumerate(lst) if i not in rem]

            fg_coords1_f = filter_by_indices(fg_coords1, fg_remove_local)
            fg_coords2_f = filter_by_indices(fg_coords2, fg_remove_local)
            fg_sims_f   = filter_by_indices(fg_similarities, fg_remove_local)
            bg_coords1_f = filter_by_indices(bg_coords1, bg_remove_local)
            bg_coords2_f = filter_by_indices(bg_coords2, bg_remove_local)
            bg_sims_f    = filter_by_indices(bg_similarities, bg_remove_local)

            print(f"[hybrid_knn_groups] Removed FG: {len(fg_remove_local)}, BG: {len(bg_remove_local)} (k={k_neighbors})")
            return fg_coords1_f, fg_coords2_f, fg_sims_f, bg_coords1_f, bg_coords2_f, bg_sims_f
        except Exception as e:
            print(f"[hybrid_knn_groups] Error: {e}")
            return fg_coords1, fg_coords2, fg_similarities, bg_coords1, bg_coords2, bg_similarities

