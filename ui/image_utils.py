import cv2
import numpy as np
import PIL
from PIL import Image
from typing import Tuple, Union, List, Dict, Any, Optional

class ImageResizer:
    """이미지 리사이징을 처리하는 유틸리티 클래스"""
    
    RESIZE_OPTIONS = {
        "원본 이미지": 1.0,
        "512x512 고정": "512x512"
    }
    
    @staticmethod
    def resize_image(image: Union[np.ndarray, PIL.Image.Image], scale) -> np.ndarray:
        """
        이미지를 지정된 설정에 따라 리사이징합니다.
        
        Args:
            image: 원본 이미지 (NumPy 배열 또는 PIL 이미지)
            scale: 리사이징 설정 (1.0 또는 "512x512")
            
        Returns:
            리사이징된 이미지 (NumPy 배열)
        """
        # PIL 이미지를 NumPy 배열로 변환
        if isinstance(image, PIL.Image.Image):
            image = np.array(image)
            
        # 원본 크기 유지
        if scale == 1.0:
            return image
        
        # 512x512 크기로 고정 리사이징
        if scale == "512x512":
            resized = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
            return resized
            
        # 이 부분은 더 이상 사용되지 않지만 호환성을 위해 유지
        if isinstance(scale, float) and scale < 1.0:
            # 이미지 크기 계산
            h, w = image.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            
            # 리사이징
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized
            
        return image
    
    @staticmethod
    def resize_mask(mask: np.ndarray, original_size: Tuple[int, int], scale: float) -> np.ndarray:
        """
        마스크를 원본 이미지 크기로 다시 리사이징합니다.
        
        Args:
            mask: 마스크 (NumPy 배열)
            original_size: 원본 이미지 크기 (높이, 너비)
            scale: 현재 스케일 (리사이징 메타데이터에 있음)
            
        Returns:
            원본 크기로 리사이징된 마스크
        """
        h, w = original_size
        resized_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return resized_mask