from typing import Optional, Tuple, List
import numpy as np


class MatchingEngine:
    """스파스 매칭 파사드.

    내부적으로 predictor의 DINOv3 matcher와 시각화 함수를 호출하여
    통일된 인터페이스를 제공합니다.
    """

    def __init__(self, memory_sam_predictor):
        self.pred = memory_sam_predictor

    def extract_patch_features(self, image: np.ndarray):
        return self.pred.extract_patch_features(image)

    def visualize(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        mask1: Optional[np.ndarray] = None,
        mask2: Optional[np.ndarray] = None,
        skip_clustering: bool = False,
        hybrid_clustering: bool = False,
        match_background: bool = True,
        max_matches: int = 50,
        save_path: Optional[str] = None,
    ):
        return self.pred.visualize_sparse_matches(
            image1=image1,
            image2=image2,
            mask1=mask1,
            mask2=mask2,
            skip_clustering=skip_clustering,
            hybrid_clustering=hybrid_clustering,
            match_background=match_background,
            max_matches=max_matches,
            save_path=save_path,
        )


