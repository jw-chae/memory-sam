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

# (removed) transformers 기반 DINOv2 전역 특징 추출은 DINOv3로 대체됨

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

# Import memory system and DINOv2 matcher
from scripts.memory_system import MemorySystem
from scripts.dinov3_matcher import Dinov3Matcher
from scripts.feature_extractor import FeatureExtractor
from scripts.prompt_generator import PromptGenerator
from scripts.sparse_matcher import SparseMatcher
from scripts.visualization import Visualization


class MemorySAMPredictor:
    """Orchestrates the Memory SAM segmentation process using refactored components."""
    
    def __init__(self, 
                model_type: str = "hiera_l", 
                checkpoint_path: str = None,
                dinov3_model: str = "dinov3_vitl16",
                memory_dir: str = "memory",
                results_dir: str = "results",
                device: str = "cuda",
                process_at_dinov3_size: bool = True):
        
        self.device = self._get_device(device)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Performance optimization: process at DINOv3 size
        self.process_at_dinov3_size = process_at_dinov3_size
        self.dinov3_image_size = 768
        self.dinov3_patch_size = 16
        
        # UI-configurable settings
        self.use_kmeans_fg = True
        self.kmeans_fg_clusters = 10
        self.skip_clustering = False
        
        # Initialize components
        self.sam_predictor = self._load_sam_model(model_type, checkpoint_path)
        self.memory = MemorySystem(memory_dir)
        self.feature_extractor = FeatureExtractor(dinov3_model, self.device)
        self.prompt_generator = PromptGenerator()
        self.sparse_matcher = SparseMatcher()
        self.visualization = Visualization(self.sparse_matcher)

        # Limit SAM input size to avoid OOM; mask will be upscaled back
        self.sam_max_side = 1400  # you can adjust via attribute if needed

    def _get_device(self, device_str: str) -> torch.device:
        final_device = torch.device("cpu")
        if device_str == "cuda" and torch.cuda.is_available():
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            final_device = torch.device("cuda")
        
        print(f"Using device: {final_device}")
        return final_device
    
    def resize_to_dinov3_size(self, image: np.ndarray) -> np.ndarray:
        """
        DINOv3와 동일한 크기로 리사이즈 (종횡비 유지)
        작은 이미지는 업스케일하지 않음
        
        Args:
            image: 입력 이미지 (H, W, 3)
            
        Returns:
            리사이즈된 이미지
        """
        h, w = image.shape[:2]
        
        # DINOv3 리사이즈 로직
        h_patches = int(self.dinov3_image_size / self.dinov3_patch_size)
        w_patches = int((w * self.dinov3_image_size) / (h * self.dinov3_patch_size))
        target_h = h_patches * self.dinov3_patch_size
        target_w = w_patches * self.dinov3_patch_size
        
        # 이미지가 이미 작으면 리사이즈하지 않음 (업스케일 방지)
        if target_w * target_h >= w * h:
            print(f"[RESIZE] {w}x{h} - 이미 충분히 작음, 리사이즈 건너뜀")
            return image
        
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        print(f"[RESIZE] {w}x{h} -> {target_w}x{target_h} (DINOv3 size, {target_w*target_h/(w*h)*100:.1f}% of original)")
        
        return resized
        
    def _load_sam_model(self, model_type, checkpoint_path):
        if checkpoint_path is None:
            model_name_map = {
                "hiera_b+": ["base_plus", "b+"],
                "hiera_l": ["large", "l"],
                "hiera_s": ["small", "s"],
                "hiera_t": ["tiny", "t"],
            }
            checkpoint_patterns = [
                f"sam2_{model_type}.pt",
                f"sam2.1_{model_type}.pt",
            ]
            if model_type in model_name_map:
                for variant in model_name_map[model_type]:
                    checkpoint_patterns.extend([
                        f"sam2_hiera_{variant}.pt",
                        f"sam2.1_hiera_{variant}.pt",
                    ])
            
            checkpoint_dir = '/home/joongwon00/sam2/checkpoints'
            found_path = None
            for pattern in checkpoint_patterns:
                path = os.path.join(checkpoint_dir, pattern)
                if os.path.exists(path):
                    found_path = path
                    break
            
            if found_path is None:
                raise FileNotFoundError(f"SAM Checkpoint not found for model {model_type}. Searched patterns: {checkpoint_patterns}")
            checkpoint_path = found_path

        print(f"Found SAM Checkpoint at: {checkpoint_path}")

        if not GlobalHydra.instance().is_initialized():
            initialize_config_module("sam2", version_base="1.2")
        
        model_type_short = model_type.replace('hiera_', '')
        config_name = f"configs/sam2.1/sam2.1_hiera_{model_type_short}"
        
        try:
            sam_model = build_sam2(config_name, checkpoint_path, device=self.device)
            print(f"Successfully loaded SAM model with config: {config_name}")
        except Exception as e:
            print(f"Failed to load SAM model with config {config_name}, trying fallback. Error: {e}")
            config_name = f"configs/sam2/sam2_hiera_{model_type_short}"
            try:
                sam_model = build_sam2(config_name, checkpoint_path, device=self.device)
                print(f"Successfully loaded SAM model with fallback config: {config_name}")
            except Exception as e2:
                print("All attempts to load SAM model failed.")
                raise e2
                
        return SAM2ImagePredictor(sam_model)

    def process_image(self, image_path: str, match_background: bool = True) -> Dict:
        print("\n" + "="*50)
        print(f"START: Processing image '{os.path.basename(image_path)}'")
        print(f"Params: match_background={match_background}, skip_clustering={self.skip_clustering}, kmeans_clusters={self.kmeans_fg_clusters}")
        print("="*50)

        # Load image with error handling for corrupted files
        try:
            pil_image = Image.open(image_path)
            pil_image.load()  # Force load to detect truncated files
            image = np.array(pil_image.convert("RGB"))
        except (OSError, IOError) as e:
            error_msg = f"이미지 파일 손상 또는 읽기 실패: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "error": error_msg,
                "image_path": image_path,
                "mask": None,
                "score": 0.0,
            }
        except Exception as e:
            error_msg = f"이미지 로드 중 예상치 못한 오류: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                "error": error_msg,
                "image_path": image_path,
                "mask": None,
                "score": 0.0,
            }
        
        # Resize to DINOv3 size for performance optimization
        if self.process_at_dinov3_size:
            image = self.resize_to_dinov3_size(image)
        
        # 1. Feature Extraction
        print("\n[Step 1] Extracting patch features from input image...")
        patch_features, grid_size, _ = self.feature_extractor.extract_patch_features(image)
        print(" -> Done.")

        # 2. Memory Search (Top 5)
        print("\n[Step 2] Searching for top 5 similar items in memory...")
        similar_items = self.memory.get_most_similar_sparse(
            patch_features, grid_size=grid_size, top_k=5, match_background=match_background
        )
        if not similar_items:
            print(" -> No similar items found in memory. Aborting.")
            return {"error": "No similar items found."}
        print(f" -> Found {len(similar_items)} similar items.")

        # 3. Prompt Generation from DINO-matched points (centroid on target image)
        print("\n[Step 3] Generating SAM prompt from DINO-matched points (centroid)...")
        best_item_data = self.memory.load_item_data(similar_items[0]["item"]["id"])
        memory_mask = best_item_data.get("mask")
        memory_patch_features = best_item_data.get("patch_features")
        memory_grid_size = best_item_data.get("grid_size")

        # Update settings for downstream modules
        self.prompt_generator.use_kmeans_fg = not self.skip_clustering
        self.prompt_generator.kmeans_fg_clusters = self.kmeans_fg_clusters
        self.sparse_matcher.kmeans_fg_clusters = self.kmeans_fg_clusters

        prompt = None
        try:
            if memory_patch_features is not None and memory_grid_size is not None:
                # Prepare coordinates on memory grid (prefer foreground region via mask)
                mem_h, mem_w = memory_grid_size
                if memory_mask is not None:
                    mem_mask_resized = cv2.resize(memory_mask.astype(np.uint8), (mem_w, mem_h), interpolation=cv2.INTER_NEAREST) > 0
                    mem_y, mem_x = np.where(mem_mask_resized)
                else:
                    mem_y, mem_x = np.where(np.ones((mem_h, mem_w), dtype=bool))

                # Prepare coordinates on query grid (use all locations)
                qry_h, qry_w = grid_size
                qry_y, qry_x = np.where(np.ones((qry_h, qry_w), dtype=bool))

                # Flatten patch features to rows
                # Shapes: (C, H, W) -> (H*W, C)
                mem_flat = memory_patch_features.reshape(memory_patch_features.shape[0], -1).T
                qry_flat = patch_features.reshape(patch_features.shape[0], -1).T

                # Index by grid coords to align features with coords arrays
                mem_indices = mem_y * mem_w + mem_x
                qry_indices = qry_y * qry_w + qry_x
                mem_feats_sel = mem_flat[mem_indices]
                qry_feats_sel = qry_flat[qry_indices]

                # Match features and get coordinates on both images
                match_coords1, match_coords2, match_sims = self._match_features_with_coords(
                    mem_feats_sel, qry_feats_sel,
                    (mem_y, mem_x), (qry_y, qry_x),
                    memory_grid_size, grid_size,
                    best_item_data["image"].shape, image.shape,
                    similarity_threshold=0.7, max_matches=200
                )

                if len(match_coords2) > 0:
                    # Compute centroid on the target image
                    xs = np.array([p[0] for p in match_coords2], dtype=np.float32)
                    ys = np.array([p[1] for p in match_coords2], dtype=np.float32)
                    cx = float(np.mean(xs))
                    cy = float(np.mean(ys))
                    point = np.array([[cx, cy]], dtype=np.float32)
                    labels = np.array([1], dtype=np.int32)
                    prompt = {"points": point, "labels": labels}
                    print(f" -> Generated centroid prompt from {len(match_coords2)} matches at ({cx:.1f}, {cy:.1f}).")
                else:
                    print(" -> No reliable matches found. Falling back to mask-based prompt.")
            else:
                print(" -> Memory item lacks patch features or grid size. Falling back to mask-based prompt.")
        except Exception as e:
            print(f" -> Matching-based prompt generation failed: {e}. Falling back to mask-based prompt.")

        # Fallback: use mask-based KMeans prompt
        if prompt is None:
            prompt = self.prompt_generator.generate_prompt(
                memory_mask if memory_mask is not None else np.zeros(image.shape[:2], dtype=np.uint8), 
                original_size=image.shape[:2], 
                match_background=match_background
            )
        print(f" -> Generated prompt with {len(prompt['points'])} point(s) ({np.sum(prompt['labels']==1)} FG, {np.sum(prompt['labels']==0)} BG).")

        # 4. SAM Segmentation
        print("\n[Step 4] Performing segmentation with SAM...")
        mask, score = self.segment_with_sam(image, prompt)
        print(f" -> Segmentation complete. Mask score: {score:.4f}")

        # 5. Visualization
        print("\n[Step 5] Generating visualizations...")
        print(f"[DEBUG] 배경 매칭 상태: {match_background}")
        vis_img = self.visualization.visualize_mask(image, mask)
        
        # Also generate sparse match visualization
        memory_image = best_item_data["image"]
        memory_patch_features = best_item_data.get("patch_features")
        memory_grid_size = best_item_data.get("grid_size")
        
        print(f"[DEBUG] 메모리 패치 특징: {memory_patch_features.shape if memory_patch_features is not None else 'None'}")
        print(f"[DEBUG] 메모리 그리드 크기: {memory_grid_size}")
        
        sparse_match_vis, img1_points, img2_points = (None, None, None)
        if memory_patch_features is not None and memory_grid_size is not None:
            print(" -> Generating sparse match visualization...")
            try:
                sparse_match_vis, img1_points, img2_points = self.visualization.visualize_sparse_matches(
                    memory_image, image,
                    memory_patch_features, patch_features,
                    memory_grid_size, grid_size,
                    memory_mask, mask,
                    match_background=match_background,
                    use_kmeans=(not self.skip_clustering) # Pass use_kmeans instead
                )
                print(f"[DEBUG] 스파스 매칭 시각화 완료: {sparse_match_vis.shape if sparse_match_vis is not None else 'None'}")
                print(f"[DEBUG] 이미지1 포인트: {img1_points.shape if img1_points is not None else 'None'}")
                print(f"[DEBUG] 이미지2 포인트: {img2_points.shape if img2_points is not None else 'None'}")
            except Exception as e:
                print(f"[ERROR] 스파스 매칭 시각화 생성 실패: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(" -> Skipping sparse match visualization (memory item lacks patch features).")
        print(" -> Done.")


        # 6. Prepare gallery items for UI
        print("\n[Step 6] Preparing final results package...")
        gallery_items = []
        for item in similar_items:
            item_id = item["item"]["id"]
            # Avoid loading full image arrays into memory; fetch only metadata
            item_meta = self.memory.get_item(item_id)
            if item_meta and "image_path" in item_meta:
                image_path_full = str(self.memory.memory_dir / item_meta["image_path"]) 
                gallery_items.append({
                    "id": item_id,
                    "similarity": item["similarity"],
                    "image_path": image_path_full
                })

        return {
            "image": image, "mask": mask, "score": float(score),
            "visualization": vis_img, "image_path": image_path,
            "gallery_items": gallery_items,
            "sparse_match_visualization": sparse_match_vis,
            "img1_points": img1_points,
            "img2_points": img2_points,
        }

    def segment_with_sam(self, image: np.ndarray, prompt: Dict) -> Tuple[np.ndarray, float]:
        # Optionally downscale image to reduce memory consumption during SAM
        orig_h, orig_w = image.shape[:2]
        max_side = max(orig_h, orig_w)
        scale = 1.0
        proc_image = image
        points_np = np.array(prompt["points"], dtype=np.float32)
        labels_np = np.array(prompt["labels"], dtype=np.int32)
        if max_side > getattr(self, 'sam_max_side', max_side):
            scale = self.sam_max_side / float(max_side)
            new_size = (int(round(orig_w * scale)), int(round(orig_h * scale)))
            proc_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
            points_np = points_np * scale
            print(f"[SAM] Downscaled image for SAM: {orig_w}x{orig_h} -> {new_size[0]}x{new_size[1]} (scale={scale:.4f})")

        self.sam_predictor.set_image(proc_image)
        
        # Convert to tensors
        points = torch.as_tensor([points_np], dtype=torch.float, device=self.device)
        labels = torch.as_tensor([labels_np], dtype=torch.int, device=self.device)
        
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=points, 
            point_labels=labels, 
            multimask_output=True
        )
        
        if masks is None or scores is None or scores.size == 0:
            return np.zeros(image.shape[:2], dtype=bool), 0.0
            
        # scores is a numpy array, handle different shapes and convert to tensor
        scores_for_image = scores[0] if scores.ndim > 1 else scores
        scores_tensor = torch.from_numpy(np.atleast_1d(scores_for_image)).to(self.device)
        best_mask_idx = torch.argmax(scores_tensor)
        
        # Handle different possible shapes of the masks array based on backup logic
        if masks.ndim == 4:  # Batch, Num_Masks, H, W
            best_mask = masks[0, best_mask_idx]
        elif masks.ndim == 3:  # Num_Masks, H, W
            best_mask = masks[best_mask_idx]
        else:
            print(f"Warning: Unexpected mask shape: {masks.shape}. Attempting to select best mask.")
            # This case might be risky, but it's a fallback based on old code
            if masks.ndim == 2 and len(scores_for_image) > 1: # multiple masks returned as a stack of 2D arrays
                best_mask = masks # This is likely wrong, let's assume it's one mask
            elif masks.ndim == 2:
                best_mask = masks
            else: # 1D or other
                best_mask = masks[best_mask_idx]

            
        best_score = scores_tensor[best_mask_idx].item()
        
        # Final safety check for mask dimension
        if best_mask.ndim != 2:
            print(f"Error: Final mask has invalid dimension {best_mask.ndim}. Returning empty mask.")
            return np.zeros(image.shape[:2], dtype=bool), 0.0
        
        # If we downscaled for SAM, upscale mask back to original image size
        final_mask = best_mask
        if scale != 1.0:
            final_mask_np = best_mask.astype(np.uint8)
            final_mask_np = cv2.resize(final_mask_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            final_mask = final_mask_np.astype(bool)
            print(f"[SAM] Upscaled mask back to original size: {orig_w}x{orig_h}")

        # Attempt to free intermediate CUDA memory (if using GPU)
        try:
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
        except Exception:
            pass

        return final_mask, best_score
    
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
        
        # Blend only inside the mask to avoid darkening outside the mask
        if np.any(binary_mask):
            vis_float = vis.astype(np.float32)
            blend_color = np.array([30, 144, 255], dtype=np.float32)
            vis_float[binary_mask] = vis_float[binary_mask] * (1.0 - float(alpha)) + blend_color * float(alpha)
            vis = vis_float.astype(np.uint8)
        
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
        print(f"[DEBUG] visualize_sparse_matches 시작 - 이미지1: {image1.shape}, 이미지2: {image2.shape}")
        print(f"[DEBUG] 마스크1: {mask1.shape if mask1 is not None else 'None'}, 마스크2: {mask2.shape if mask2 is not None else 'None'}")
        print(f"[DEBUG] 매개변수: max_matches={max_matches}, skip_clustering={skip_clustering}, hybrid_clustering={hybrid_clustering}, match_background={match_background}")
        
        # Ensure masks are single-channel before processing
        if mask1 is not None and mask1.ndim == 3:
            print(f"[DEBUG] 마스크1을 그레이스케일로 변환: {mask1.shape}")
            mask1 = cv2.cvtColor(mask1, cv2.COLOR_RGB2GRAY)
        if mask2 is not None and mask2.ndim == 3:
            print(f"[DEBUG] 마스크2를 그레이스케일로 변환: {mask2.shape}")
            mask2 = cv2.cvtColor(mask2, cv2.COLOR_RGB2GRAY)

        # Add logging for skip_clustering and hybrid_clustering values
        print(f"[visualize_sparse_matches] Received skip_clustering: {skip_clustering}, hybrid_clustering: {hybrid_clustering}")
        
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
            # Save original image sizes
            original_shape1 = image1.shape
            original_shape2 = image2.shape
            
            # Check if masks are same size as images
            if mask1 is not None and (mask1.shape[0] != image1.shape[0] or mask1.shape[1] != image1.shape[1]):
                print(f"Resizing mask1: {mask1.shape} -> {image1.shape[:2]}")
                mask1 = cv2.resize(mask1.astype(np.uint8), (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            
            if mask2 is not None and (mask2.shape[0] != image2.shape[0] or mask2.shape[1] != image2.shape[1]):
                print(f"Resizing mask2: {mask2.shape} -> {image2.shape[:2]}")
                mask2 = cv2.resize(mask2.astype(np.uint8), (image2.shape[1], image2.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            
            # Check and adjust image size (save scale info)
            scale1_x, scale1_y = 1.0, 1.0
            scale2_x, scale2_y = 1.0, 1.0
            
            if image1.shape[0] > 1000 or image1.shape[1] > 1000:
                scale = min(1000 / image1.shape[0], 1000 / image1.shape[1])
                new_size = (int(image1.shape[1] * scale), int(image1.shape[0] * scale))
                # Save scale info (original size / adjusted size)
                scale1_x = original_shape1[1] / new_size[0]
                scale1_y = original_shape1[0] / new_size[1]
                print(f"Resizing image1: {image1.shape[:2]} -> {(new_size[1], new_size[0])}")
                print(f"Image1 scale info: x={scale1_x:.3f}, y={scale1_y:.3f}")
                image1 = cv2.resize(image1, new_size)
                if mask1 is not None:
                    mask1 = cv2.resize(mask1.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST) > 0
            
            if image2.shape[0] > 1000 or image2.shape[1] > 1000:
                scale = min(1000 / image2.shape[0], 1000 / image2.shape[1])
                new_size = (int(image2.shape[1] * scale), int(image2.shape[0] * scale))
                # Save scale info (original size / adjusted size)
                scale2_x = original_shape2[1] / new_size[0]
                scale2_y = original_shape2[0] / new_size[1]
                print(f"Resizing image2: {image2.shape[:2]} -> {(new_size[1], new_size[0])}")
                print(f"Image2 scale info: x={scale2_x:.3f}, y={scale2_y:.3f}")
                image2 = cv2.resize(image2, new_size)
                if mask2 is not None:
                    mask2 = cv2.resize(mask2.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST) > 0
            
            # Save scale info as global variable (for use in other functions)
            self.current_scale_image1 = (scale1_x, scale1_y)
            self.current_scale_image2 = (scale2_x, scale2_y)
            self.current_original_shape1 = original_shape1[:2]
            self.current_original_shape2 = original_shape2[:2]
            
            # Grayscale conversion (improves keypoint extraction)
            gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY) if len(image1.shape) == 3 else image1
            gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY) if len(image2.shape) == 3 else image2
            
            # Extract patch features
            print(f"[DEBUG] 이미지1에서 패치 특징 추출 시작...")
            patch_features1, grid_size1, resize_scale1 = self.feature_extractor.extract_patch_features(image1)
            print(f"[DEBUG] 이미지1 특징 추출 완료: {patch_features1.shape}, 그리드 크기: {grid_size1}")
            
            print(f"[DEBUG] 이미지2에서 패치 특징 추출 시작...")
            patch_features2, grid_size2, resize_scale2 = self.feature_extractor.extract_patch_features(image2)
            print(f"[DEBUG] 이미지2 특징 추출 완료: {patch_features2.shape}, 그리드 크기: {grid_size2}")
            
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
            
            # Separate foreground and background areas using grid coordinates
            # Get grid coordinates for foreground area
            fg_coords1 = np.where(resized_mask1)
            fg_coords2 = np.where(resized_mask2)
            
            # Get grid coordinates for background area
            bg_coords1 = np.where(~resized_mask1)
            bg_coords2 = np.where(~resized_mask2)
            
            # Extract features using grid coordinates
            fg_features1_masked = patch_features1[:, fg_coords1[0], fg_coords1[1]].T  # Shape: (num_fg_points, 1024)
            fg_features2_masked = patch_features2[:, fg_coords2[0], fg_coords2[1]].T  # Shape: (num_fg_points, 1024)
            
            bg_features1_masked = patch_features1[:, bg_coords1[0], bg_coords1[1]].T  # Shape: (num_bg_points, 1024)
            bg_features2_masked = patch_features2[:, bg_coords2[0], bg_coords2[1]].T  # Shape: (num_bg_points, 1024)
            
            # Give higher priority to foreground keypoints
            similarity_threshold_fg = 0.8  # Default similarity threshold
            similarity_threshold_bg = 0.7  # Background area similarity threshold
            
            # If too few keypoints, lower similarity threshold and continue using mask
            if len(fg_coords1[0]) < 5 or len(fg_coords2[0]) < 5:
                print(f"Too few keypoints in foreground area: {len(fg_coords1[0])}, {len(fg_coords2[0])}")
                print("Maintaining mask area and lowering similarity threshold.")
                similarity_threshold_fg = 0.65  # Use lower similarity threshold
            elif len(fg_coords1[0]) > 30 and len(fg_coords2[0]) > 30:
                # If enough foreground keypoints, use stricter similarity threshold
                print(f"Sufficient foreground keypoints: {len(fg_coords1[0])}, {len(fg_coords2[0])}")
                similarity_threshold_fg = 0.8  # Use higher similarity threshold (Adjusted from 0.85)
                # Reduce background keypoint ratio
                similarity_threshold_bg = 0.7  # Higher background similarity threshold (Adjusted from 0.75)
            
            # Limit number of background keypoints (proportional to foreground keypoints)
            bg_limit = min(len(fg_coords1[0]) // 2, 30)  # Half of foreground or max 30
            
            if len(bg_coords1[0]) > 0 and len(bg_coords2[0]) > 0:
                print(f"[DEBUG] 배경 좌표 필터링 시작 - 원본 배경 좌표: {len(bg_coords1[0])}, {len(bg_coords2[0])}")
                
                # Evaluate background keypoint quality (variance-based)
                bg_features1_variance = np.var(bg_features1_masked, axis=1)
                bg_features2_variance = np.var(bg_features2_masked, axis=1)
                
                # Select top background keypoints with high variance (limited number)
                bg_top_k1 = min(len(bg_features1_variance), bg_limit)
                bg_top_k2 = min(len(bg_features2_variance), bg_limit)
                print(f"[DEBUG] 배경 특징 필터링: {bg_top_k1}, {bg_top_k2}개 선택 (제한: {bg_limit})")
                
                bg_top_indices1 = np.argsort(bg_features1_variance)[-bg_top_k1:]
                bg_top_indices2 = np.argsort(bg_features2_variance)[-bg_top_k2:]
                
                bg_features1_filtered = bg_features1_masked[bg_top_indices1]
                bg_features2_filtered = bg_features2_masked[bg_top_indices2]
                
                # Maintain original coordinate mapping
                bg_coords1_filtered = (bg_coords1[0][bg_top_indices1], bg_coords1[1][bg_top_indices1])
                bg_coords2_filtered = (bg_coords2[0][bg_top_indices2], bg_coords2[1][bg_top_indices2])
                print(f"[DEBUG] 배경 좌표 필터링 완료: {len(bg_coords1_filtered[0])}, {len(bg_coords2_filtered[0])}개")
            else:
                print(f"[DEBUG] 배경 좌표가 없어서 필터링 건너뜀")
            
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
                
                # Maintain original coordinate mapping
                fg_coords1_filtered = (fg_coords1[0][fg_top_indices1], fg_coords1[1][fg_top_indices1])
                fg_coords2_filtered = (fg_coords2[0][fg_top_indices2], fg_coords2[1][fg_top_indices2])
                
                # Set max_matches value for _match_features based on skip_clustering (for foreground)
                current_max_matches_param_fg = 100000 if use_skip_clustering else 100

                # Match foreground keypoints
                fg_match_coords1, fg_match_coords2, fg_match_similarities = self.sparse_matcher.match_features_with_coords(
                    fg_features1_filtered, fg_features2_filtered,
                    fg_coords1_filtered, fg_coords2_filtered,
                    grid_size1, grid_size2, image1.shape, image2.shape,
                    similarity_threshold=similarity_threshold_fg,
                    max_matches=current_max_matches_param_fg 
                )
            else:
                print("Skipping matching due to no foreground keypoints.")
                fg_match_coords1, fg_match_coords2, fg_match_similarities = [], [], []
            
            # Match background keypoints (if background area exists) - apply stricter threshold
            bg_match_coords1, bg_match_coords2, bg_match_similarities = [], [], []
            print(f"[DEBUG] 배경 매칭 조건 확인:")
            print(f"[DEBUG] - match_background: {match_background}")
            print(f"[DEBUG] - bg_coords1 길이: {len(bg_coords1[0]) if len(bg_coords1) > 0 else 0}")
            print(f"[DEBUG] - bg_coords2 길이: {len(bg_coords2[0]) if len(bg_coords2) > 0 else 0}")
            print(f"[DEBUG] - bg_coords1_filtered 존재: {'bg_coords1_filtered' in locals()}")
            print(f"[DEBUG] - bg_coords2_filtered 존재: {'bg_coords2_filtered' in locals()}")
            
            if match_background and len(bg_coords1[0]) > 0 and len(bg_coords2[0]) > 0 and 'bg_coords1_filtered' in locals() and 'bg_coords2_filtered' in locals():
                print(f"[DEBUG] 배경 매칭 시작 - 필터링된 배경 특징: {bg_features1_filtered.shape}, {bg_features2_filtered.shape}")
                # Set max_matches value for _match_features based on skip_clustering (for background)
                current_max_matches_param_bg = 50000 if use_skip_clustering else bg_limit 
                print(f"[DEBUG] 배경 매칭 max_matches: {current_max_matches_param_bg}")

                bg_match_coords1, bg_match_coords2, bg_match_similarities = self.sparse_matcher.match_features_with_coords(
                    bg_features1_filtered, bg_features2_filtered,
                    bg_coords1_filtered, bg_coords2_filtered,
                    grid_size1, grid_size2, image1.shape, image2.shape,
                    similarity_threshold=similarity_threshold_bg,
                    max_matches=current_max_matches_param_bg
                )
                print(f"[DEBUG] 배경 매칭 완료: {len(bg_match_coords1)}개 매칭")
            else:
                print(f"[DEBUG] 배경 매칭 건너뜀 - 조건 불충족")
            
            # Determine whether to apply clustering
            if not use_skip_clustering:
                print(f"[visualize_sparse_matches] Applying clustering. Initial foreground matches: {len(fg_match_coords1)}, background matches: {len(bg_match_coords1)}")
                
                # Store original hybrid_clustering setting
                original_hybrid_clustering = getattr(self, 'hybrid_clustering', False)
                
                # Set hybrid_clustering flag to the value passed to this function
                self.hybrid_clustering = use_hybrid_clustering
                
                # Cluster foreground and background keypoints
                if len(fg_match_coords1) > 0:
                    fg_match_coords1, fg_match_coords2, fg_match_similarities = self.sparse_matcher.cluster_feature_points(
                        fg_match_coords1, fg_match_coords2, fg_match_similarities, 
                        n_clusters=self.kmeans_fg_clusters, is_foreground=True
                    )
                
                if match_background and len(bg_match_coords1) > 0:
                    bg_match_coords1, bg_match_coords2, bg_match_similarities = self.sparse_matcher.cluster_feature_points(
                        bg_match_coords1, bg_match_coords2, bg_match_similarities, 
                        n_clusters=5, is_foreground=False # Use a fixed number for background
                    )
                
                # Restore original hybrid_clustering setting
                self.hybrid_clustering = original_hybrid_clustering
                
                print(f"[visualize_sparse_matches] Clustering applied. Final foreground matches: {len(fg_match_coords1)}, background matches: {len(bg_match_coords1)}")
            else:
                print(f"[visualize_sparse_matches] Skipping clustering. Using all matches - FG: {len(fg_match_coords1)}, BG: {len(bg_match_coords1)}")
                # When skip_clustering is true, limit matches to prevent too many points
                if len(fg_match_coords1) > max_matches * 3:
                    # Sort by similarity and take top matches for foreground
                    indices = np.argsort(fg_match_similarities)[::-1][:max_matches * 3]
                    fg_match_coords1 = [fg_match_coords1[i] for i in indices]
                    fg_match_coords2 = [fg_match_coords2[i] for i in indices]
                    fg_match_similarities = [fg_match_similarities[i] for i in indices]
                    print(f"Limited foreground matches to {len(fg_match_coords1)} (top similarity)")
                
                if match_background and len(bg_match_coords1) > max_matches // 2:
                    # Sort by similarity and take top matches for background
                    indices = np.argsort(bg_match_similarities)[::-1][:max_matches // 2]
                    bg_match_coords1 = [bg_match_coords1[i] for i in indices]
                    bg_match_coords2 = [bg_match_coords2[i] for i in indices]
                    bg_match_similarities = [bg_match_similarities[i] for i in indices]
                    print(f"Limited background matches to {len(bg_match_coords1)} (top similarity)")
            
            # Combine foreground and background keypoints
            if match_background:
                coords1 = fg_match_coords1 + bg_match_coords1
                coords2 = fg_match_coords2 + bg_match_coords2
                match_similarities = fg_match_similarities + bg_match_similarities
                point_types = ['foreground'] * len(fg_match_coords1) + ['background'] * len(bg_match_coords1)
            else:
                coords1, coords2, match_similarities = fg_match_coords1, fg_match_coords2, fg_match_similarities
                point_types = ['foreground'] * len(fg_match_coords1)

            # K-means point filtering if requested (guard undefined attribute)
            if getattr(self, 'use_kmeans_fg', False) and getattr(self, 'show_only_kmeans_points', False) and len(fg_match_coords1) > self.kmeans_fg_clusters:
                print(f"Filtering visualization to show only {self.kmeans_fg_clusters} K-means foreground points.")
                
                # Select only foreground points for k-means
                fg_indices = [i for i, ptype in enumerate(point_types) if ptype == 'foreground']
                fg_coords1_to_cluster = np.array([coords1[i] for i in fg_indices])
                
                if len(fg_coords1_to_cluster) > self.kmeans_fg_clusters:
                    # Use k-means to find representative points
                    kmeans_points = self.sparse_matcher._kmeans_sampling(fg_coords1_to_cluster, n_clusters=self.kmeans_fg_clusters)
                    
                    # Find the original points that are closest to the k-means centers
                    final_indices = []
                    for kmeans_point in kmeans_points:
                        distances = np.linalg.norm(fg_coords1_to_cluster - kmeans_point, axis=1)
                        closest_original_idx = fg_indices[np.argmin(distances)]
                        final_indices.append(closest_original_idx)
                    
                    # Also include all background points
                    bg_indices = [i for i, ptype in enumerate(point_types) if ptype == 'background']
                    final_indices.extend(bg_indices)
                    
                    # Filter all lists based on final indices
                    coords1 = [coords1[i] for i in final_indices]
                    coords2 = [coords2[i] for i in final_indices]
                    match_similarities = [match_similarities[i] for i in final_indices]
            
            # Print matching results
            print(f"[DEBUG] 매칭 결과 요약:")
            print(f"[DEBUG] - 전경 매칭: {len(fg_match_coords1)}개")
            print(f"[DEBUG] - 배경 매칭: {len(bg_match_coords1) if match_background else 0}개")
            print(f"[DEBUG] - 총 시각화용 매칭: {len(coords1)}개")
            print(f"[DEBUG] - 전경 좌표 샘플: {fg_match_coords1[:3] if fg_match_coords1 else 'None'}")
            print(f"[DEBUG] - 배경 좌표 샘플: {bg_match_coords1[:3] if bg_match_coords1 else 'None'}")
            
            # Place images side-by-side
            h1, w1 = image1.shape[:2]
            h2, w2 = image2.shape[:2]
            
            # Match height of both images
            max_h = max(h1, h2)
            image1_resized = np.zeros((max_h, w1, 3), dtype=np.uint8)
            image2_resized = np.zeros((max_h, w2, 3), dtype=np.uint8)
            
            image1_resized[:h1, :w1] = image1 if len(image1.shape) == 3 else cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
            image2_resized[:h2, :w2] = image2 if len(image2.shape) == 3 else cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
            
            # Combined image
            print(f"[DEBUG] 결합 이미지 생성: {image1_resized.shape} + {image2_resized.shape}")
            vis_img = np.hstack([image1_resized, image2_resized])
            print(f"[DEBUG] 최종 시각화 이미지 크기: {vis_img.shape}")
            
            # Visualize matches
            print(f"[DEBUG] 매칭 포인트 시각화 시작...")
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
            img1_points = image1.copy() if len(image1.shape) == 3 else cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
            img2_points = image2.copy() if len(image2.shape) == 3 else cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
            
            # Visualize original mask (translucent overlay)
            if mask1 is not None:
                mask1_overlay = img1_points.copy()
                mask1_color = np.zeros_like(mask1_overlay)
                mask1_color[mask1 > 0] = [0, 200, 0]  # Display mask in green
                # Translucently overlay mask area
                cv2.addWeighted(mask1_color, 0.3, mask1_overlay, 0.7, 0, img1_points)
                
            if mask2 is not None:
                mask2_overlay = img2_points.copy()
                mask2_color = np.zeros_like(mask2_overlay)
                mask2_color[mask2 > 0] = [0, 200, 0]  # Display mask in green
                # Translucently overlay mask area
                cv2.addWeighted(mask2_color, 0.3, mask2_overlay, 0.7, 0, img2_points)
            
            # Display foreground keypoints (blueish)
            for i in range(len(fg_match_coords1)):
                x1, y1 = fg_match_coords1[i]
                sim = fg_match_similarities[i] if i < len(fg_match_similarities) else 0.0
                # Change color intensity based on similarity
                intensity = int(255 * sim)
                color = (intensity, 0, 0)  # Display foreground in blue (BGR)
                cv2.circle(img1_points, (x1, y1), 5, color, -1)
                # Also display similarity value
                if i < 5:  # Display for top 5 only
                    cv2.putText(img1_points, f"{sim:.2f}", (x1+5, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            for i in range(len(fg_match_coords2)):
                x2, y2 = fg_match_coords2[i]
                sim = fg_match_similarities[i] if i < len(fg_match_similarities) else 0.0
                # Change color intensity based on similarity
                intensity = int(255 * sim)
                color = (intensity, 0, 0)  # Display foreground in blue (BGR)
                cv2.circle(img2_points, (x2, y2), 5, color, -1)
                # Also display similarity value
                if i < 5:  # Display for top 5 only
                    cv2.putText(img2_points, f"{sim:.2f}", (x2+5, y2-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Display background keypoints (reddish)
            for i in range(len(bg_match_coords1)):
                x1, y1 = bg_match_coords1[i]
                sim = bg_match_similarities[i] if i < len(bg_match_similarities) else 0.0
                # Change color intensity based on similarity
                intensity = int(255 * sim)
                color = (0, 0, intensity)  # Display background in red (BGR)
                cv2.circle(img1_points, (x1, y1), 5, color, -1)
            
            for i in range(len(bg_match_coords2)):
                x2, y2 = bg_match_coords2[i]
                sim = bg_match_similarities[i] if i < len(bg_match_similarities) else 0.0
                # Change color intensity based on similarity
                intensity = int(255 * sim)
                color = (0, 0, intensity)  # Display background in red (BGR)
                cv2.circle(img2_points, (x2, y2), 5, color, -1)
            
            # Add info text to image
            clustering_mode = "Skip clustering" if use_skip_clustering else "Hybrid clustering" if use_hybrid_clustering else "Standard clustering"
            cv2.putText(img1_points, f"{clustering_mode}: FG: {len(fg_match_coords1)}, BG: {len(bg_match_coords1)}", 
                     (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img2_points, f"{clustering_mode}: FG: {len(fg_match_coords2)}, BG: {len(bg_match_coords2)}", 
                     (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Save result
            if save_path:
                print(f"[DEBUG] 결과를 {save_path}에 저장 중...")
                cv2.imwrite(save_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                
                # Also save individual images
                save_dir = os.path.dirname(save_path)
                base_name = os.path.splitext(os.path.basename(save_path))[0]
                cv2.imwrite(os.path.join(save_dir, f"{base_name}_img1_points.png"), 
                           cv2.cvtColor(img1_points, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(save_dir, f"{base_name}_img2_points.png"), 
                           cv2.cvtColor(img2_points, cv2.COLOR_RGB2BGR))
                print(f"[DEBUG] 개별 이미지도 저장 완료")
            
            print(f"[DEBUG] visualize_sparse_matches 완료 - 반환 이미지 크기: {vis_img.shape}, {img1_points.shape}, {img2_points.shape}")
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
                      similarity_threshold=0.7, max_matches=100):
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
        
        # Find best match for each feature
        best_matches = []
        
        for i in range(len(features1_norm)):
            best_idx = np.argmax(similarities[i])
            best_sim = similarities[i][best_idx]
            
            if best_sim >= similarity_threshold:
                best_matches.append((i, best_idx, best_sim))
        
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
            
            # Convert to original image coordinates
            img1_x = int(x1 * (image1_shape[1] / grid_size1[1]))
            img1_y = int(y1 * (image1_shape[0] / grid_size1[0]))
            img2_x = int(x2 * (image2_shape[1] / grid_size2[1]))
            img2_y = int(y2 * (image2_shape[0] / grid_size2[0]))
            
            coords1.append((img1_x, img1_y))
            coords2.append((img2_x, img2_y))
            match_similarities.append(sim)
        
        print(f"Found {len(coords1)} matches with similarity threshold {similarity_threshold}")
        return coords1, coords2, match_similarities
    
    def _match_features_with_coords(self, features1, features2, coords1, coords2, 
                                  grid_size1, grid_size2, image1_shape, image2_shape, 
                                  similarity_threshold=0.7, max_matches=100):
        """
        Helper function to perform matching between two feature sets using coordinates
        
        Args:
            features1: First feature set
            features2: Second feature set
            coords1: Coordinates of first features (y, x)
            coords2: Coordinates of second features (y, x)
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
        
        # Find best match for each feature
        best_matches = []
        
        for i in range(len(features1_norm)):
            best_idx = np.argmax(similarities[i])
            best_sim = similarities[i][best_idx]
            
            if best_sim >= similarity_threshold:
                best_matches.append((i, best_idx, best_sim))
        
        # Sort by similarity
        best_matches.sort(key=lambda x: x[2], reverse=True)
        
        # Limit maximum number of matches
        best_matches = best_matches[:max_matches]
        
        # Convert coordinates
        match_coords1 = []
        match_coords2 = []
        match_similarities = []
        
        for i, j, sim in best_matches:
            # Get grid coordinates
            y1, x1 = coords1[0][i], coords1[1][i]
            y2, x2 = coords2[0][j], coords2[1][j]
            
            # Convert to original image coordinates
            img1_x = int(x1 * (image1_shape[1] / grid_size1[1]))
            img1_y = int(y1 * (image1_shape[0] / grid_size1[0]))
            img2_x = int(x2 * (image2_shape[1] / grid_size2[1]))
            img2_y = int(y2 * (image2_shape[0] / grid_size2[0]))
            
            match_coords1.append((img1_x, img1_y))
            match_coords2.append((img2_x, img2_y))
            match_similarities.append(sim)
        
        print(f"Found {len(match_coords1)} matches with similarity threshold {similarity_threshold}")
        return match_coords1, match_coords2, match_similarities
    
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
            
            # The number of clusters is now passed directly and not adjusted.
            adjusted_n_clusters = min(n_clusters, len(coords1))
            print(f"Clustering {len(coords1)} {'foreground' if is_foreground else 'background'} matches into {adjusted_n_clusters} clusters")
            
            # Apply K-means clustering
            # Determine features for clustering
            if hasattr(self, 'hybrid_clustering') and self.hybrid_clustering:
                # Hybrid clustering: Use only coordinate differences to cluster
                # This treats foreground and background points together
                print("Using hybrid clustering (foreground and background together)")
                
                # Add a feature to distinguish between foreground and background
                group_identifier = 1 if is_foreground else 0
                group_feature = np.full((len(coords1_array), 1), group_identifier * 1000) # High weight

                features = np.column_stack((
                    coords1_array - coords2_array,  # Coordinate differences (dx, dy)
                    similarities_array.reshape(-1, 1) * 100,  # Weight similarity
                    group_feature # Add group identifier to prevent mixing
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
            
            for i in range(adjusted_n_clusters):
                cluster_indices = np.where(cluster_labels == i)[0]
                if len(cluster_indices) > 0:
                    # Select keypoint with highest similarity within cluster
                    best_idx_in_cluster = cluster_indices[np.argmax(similarities_array[cluster_indices])]
                    clustered_coords1.append(coords1[best_idx_in_cluster])
                    clustered_coords2.append(coords2[best_idx_in_cluster])
                    clustered_similarities.append(similarities[best_idx_in_cluster])
            
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
