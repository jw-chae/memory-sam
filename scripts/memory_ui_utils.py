import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union
import tempfile
import shutil
import matplotlib.pyplot as plt

def visualize_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Visualize mask overlay on image
    
    Args:
        image: Original image
        mask: Segmentation mask
        alpha: Opacity
        
    Returns:
        Visualized image
    """
    vis = image.copy()
    
    # Create color overlay for mask
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = [30, 144, 255]  # Blue color for mask
    
    # Blend image and mask
    vis = cv2.addWeighted(vis, 1, color_mask, alpha, 0)
    
    # Draw contours
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
    
    return vis

def draw_points_on_image(image: np.ndarray, points: List[List[int]], labels: List[int], mask: Optional[np.ndarray] = None, point_size: int = 5) -> np.ndarray:
    """
    Visualize points and mask on image
    
    Args:
        image: Input image
        points: List of point coordinates [[x1, y1], [x2, y2], ...]
        labels: List of point labels (1: foreground, 0: background)
        mask: Segmentation mask (optional)
        point_size: Size of the points
        
    Returns:
        Visualized image
    """
    result_img = image.copy()
    
    # If mask is provided, visualize it
    if mask is not None:
        # Convert mask to boolean array
        if not np.issubdtype(mask.dtype, np.bool_):
            mask = mask > 0
            
        # Mask overlay
        mask_color = np.zeros_like(result_img)
        mask_color[mask] = [0, 100, 200]  # Orange color series
        
        # Semi-transparent overlay
        cv2.addWeighted(mask_color, 0.5, result_img, 1, 0, result_img)
        
        # Extract contours and draw
        if mask.dtype != np.uint8:
            mask_uint8 = mask.astype(np.uint8) * 255
        else:
            mask_uint8 = mask
            
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, contours, -1, (255, 255, 255), 2)
    
    # Draw points
    for i, (point, label) in enumerate(zip(points, labels)):
        x, y = point
        
        # Point color (foreground: blue, background: red)
        color = (255, 0, 0) if label == 1 else (0, 0, 255)
        
        # Draw point (circle and center point)
        cv2.circle(result_img, (x, y), point_size, color, -1)
        cv2.circle(result_img, (x, y), point_size + 2, (255, 255, 255), 1)
        
        # Point number display
        cv2.putText(result_img, str(i+1), (x+5, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result_img

class SparseMatchVisualizer:
    """스파스 매칭 시각화를 위한 유틸리티 클래스"""
    
    def __init__(self, memory_sam_predictor):
        """초기화"""
        self.memory_sam = memory_sam_predictor
        self.last_visualization = None
        self.last_settings = {
            'skip_clustering': False,
            'hybrid_clustering': False
        }
    
    def visualize_matches(self, image1, image2, mask1=None, mask2=None, 
                           skip_clustering=False, hybrid_clustering=False,
                           save_path=None, max_matches=50):
        """두 이미지 간의 스파스 매칭을 시각화"""
        try:
            # 클러스터링 설정을 현재 상태로 저장 (원본 설정 백업)
            original_skip = getattr(self.memory_sam, 'skip_clustering', False)
            original_hybrid = getattr(self.memory_sam, 'hybrid_clustering', False)
            
            # 임시로 입력 설정 적용
            self.memory_sam.skip_clustering = skip_clustering
            self.memory_sam.hybrid_clustering = hybrid_clustering
            
            # 시각화 생성
            sparse_vis, img1_vis, img2_vis = self.memory_sam.visualize_sparse_matches(
                image1, image2, mask1, mask2,
                skip_clustering=skip_clustering,  # 명시적 전달
                hybrid_clustering=hybrid_clustering,  # 명시적 전달
                max_matches=max_matches,
                save_path=save_path
            )
            
            # 원래 설정 복원
            self.memory_sam.skip_clustering = original_skip
            self.memory_sam.hybrid_clustering = original_hybrid
            
            # 현재 시각화와 설정 저장
            self.last_visualization = {
                'sparse_vis': sparse_vis,
                'img1_vis': img1_vis,
                'img2_vis': img2_vis
            }
            
            self.last_settings = {
                'skip_clustering': skip_clustering,
                'hybrid_clustering': hybrid_clustering
            }
            
            return sparse_vis, img1_vis, img2_vis
            
        except Exception as e:
            import traceback
            print(f"스파스 매칭 시각화 중 오류: {e}")
            traceback.print_exc()
            
            # 이전에 생성된 시각화가 있으면 반환
            if self.last_visualization:
                return (self.last_visualization['sparse_vis'], 
                        self.last_visualization['img1_vis'],
                        self.last_visualization['img2_vis'])
            
            # 오류 시 빈 이미지 반환
            empty = np.ones((400, 800, 3), dtype=np.uint8) * 255
            cv2.putText(empty, f"시각화 오류: {str(e)}", (50, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return empty, empty.copy(), empty.copy()
    
    def get_last_visualization(self):
        """마지막으로 생성된 시각화 반환"""
        if self.last_visualization:
            return (self.last_visualization['sparse_vis'], 
                    self.last_visualization['img1_vis'],
                    self.last_visualization['img2_vis'])
        return None, None, None

def browse_directory() -> str:
    """
    Browse for directory using system file browser
    
    Returns:
        Selected directory path or empty string
    """
    try:
        import subprocess
        result = subprocess.run(['zenity', '--file-selection', '--directory'], 
                                capture_output=True, text=True)
        if result.returncode == 0:
            folder_path = result.stdout.strip()
            print(f"Selected folder: {folder_path}")
            return folder_path
        return ""
    except Exception as e:
        print(f"Error browsing for folder: {e}")
        return ""

def prepare_input_data(files: Union[List[Any], str, Path], folder_path: str = "") -> Union[str, Any]:
    """
    Prepare input data (file or folder)
    
    Args:
        files: File upload or file path
        folder_path: Folder path input
        
    Returns:
        Input data to process
    """
    # Use folder path if specified
    if folder_path and os.path.isdir(folder_path):
        print(f"Using direct folder path: {folder_path}")
        return folder_path
    
    # Files as list (folder upload case)
    if isinstance(files, list):
        if len(files) == 0:
            return None
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp()
        try:
            # Copy all files to temporary directory
            for f in files:
                shutil.copy(f.name, temp_dir)
            
            # Process temporary directory path as input
            print(f"Folder processing mode: copied {len(files)} files to temporary directory {temp_dir}")
            return temp_dir
        except Exception as e:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            print(f"Error copying files: {e}")
            raise e
    elif isinstance(files, (str, Path)) and os.path.isdir(str(files)):
        # Direct folder path provided
        return str(files)
    else:
        # Single file case
        if files is not None:
            return files.name
        return None
    