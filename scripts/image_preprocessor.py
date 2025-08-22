import numpy as np
import cv2
from typing import Tuple


class ImagePreprocessor:
    """이미지 전처리 유틸리티.

    - "512x512" 모드: 종횡비를 유지한 리사이즈 후 중앙 패딩으로 정확히 512x512 생성
    - 배율(float<1.0) 모드: 단순 축소
    - 그 외: 원본 유지
    """

    def resize_image(self, image: np.ndarray, resize_enabled: bool, resize_scale) -> Tuple[np.ndarray, float]:
        """리사이즈 적용.

        Args:
            image: 입력 이미지 (H, W, C)
            resize_enabled: 리사이즈 사용 여부
            resize_scale: "512x512" 또는 0<scale<=1.0 실수

        Returns:
            (리사이즈된 이미지, 적용 배율)
        """
        if not resize_enabled:
            return image, 1.0

        # 512x512 정사각형 고정: 종횡비 유지 리사이즈 + 패딩
        if resize_scale == "512x512":
            target_size = 512
            original_height, original_width = image.shape[:2]

            if max(original_width, original_height) == 0:
                return np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype), 0.0

            scale = target_size / max(original_width, original_height)
            new_width = max(1, int(round(original_width * scale)))
            new_height = max(1, int(round(original_height * scale)))

            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

            pad_w = target_size - new_width
            pad_h = target_size - new_height
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top

            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            return padded, scale

        # 숫자 배율 축소
        if isinstance(resize_scale, (int, float)) and 0 < float(resize_scale) < 1.0:
            new_width = max(1, int(round(image.shape[1] * float(resize_scale))))
            new_height = max(1, int(round(image.shape[0] * float(resize_scale))))
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            return resized_image, float(resize_scale)

        # 원본 유지
        return image, 1.0


