import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import Tuple, Optional
import faiss

class Dinov2Matcher:
    """DINOv2 기반 이미지 매칭을 위한 클래스"""

    def __init__(self, 
                repo_name="facebookresearch/dinov2", 
                model_name="dinov2_vitb14", 
                smaller_edge_size=448, 
                half_precision=False, 
                device="cuda"):
        """
        DINOv2 Matcher 초기화
        
        Args:
            repo_name: DINOv2 모델이 있는 리포지토리 이름
            model_name: 사용할 DINOv2 모델 이름
            smaller_edge_size: 입력 이미지의 작은 쪽 크기 조정
            half_precision: 절반 정밀도 사용 여부 (메모리 절약)
            device: 사용할 장치 ('cuda' 또는 'cpu')
        """
        self.repo_name = repo_name
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.device = device

        # 모델 로드
        if self.half_precision:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
            self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

        self.model.eval()

        # 이미지 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # ImageNet 기본값
        ])

    def prepare_image(self, rgb_image_numpy: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], float]:
        """
        이미지를 DINOv2 처리용으로 준비
        
        Args:
            rgb_image_numpy: RGB 이미지 (numpy 배열)
            
        Returns:
            (image_tensor, grid_size, resize_scale) 튜플
        """
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]

        # 패치 크기의 배수가 되도록 이미지 크기 조정
        height, width = image_tensor.shape[1:]  # C x H x W
        cropped_width = width - width % self.model.patch_size
        cropped_height = height - height % self.model.patch_size
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size, resize_scale
    
    def prepare_mask(self, mask_image_numpy: np.ndarray, grid_size: Tuple[int, int], resize_scale: float) -> np.ndarray:
        """
        마스크 이미지를 DINOv2 그리드 크기로 조정
        
        Args:
            mask_image_numpy: 마스크 이미지
            grid_size: 그리드 크기
            resize_scale: 크기 조정 비율
            
        Returns:
            조정된 마스크 (1D 배열)
        """
        cropped_mask_image_numpy = mask_image_numpy[
            :int(grid_size[0]*self.model.patch_size*resize_scale), 
            :int(grid_size[1]*self.model.patch_size*resize_scale)
        ]
        image = Image.fromarray(cropped_mask_image_numpy)
        resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
        resized_mask = np.asarray(resized_mask).flatten()
        return resized_mask
    
    def extract_features(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        이미지 텐서에서 DINOv2 특징 추출
        
        Args:
            image_tensor: 처리된 이미지 텐서
            
        Returns:
            특징 배열 (numpy)
        """
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        return tokens.cpu().numpy()
    
    def idx_to_source_position(self, idx: int, grid_size: Tuple[int, int], resize_scale: float) -> Tuple[float, float]:
        """
        특징 인덱스를 원본 이미지의 위치로 변환
        
        Args:
            idx: 특징 인덱스
            grid_size: 그리드 크기
            resize_scale: 크기 조정 비율
            
        Returns:
            (row, col) 원본 이미지에서의 위치
        """
        row = (idx // grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
        col = (idx % grid_size[1])*self.model.patch_size*resize_scale + self.model.patch_size / 2
        return row, col
    
    def get_embedding_visualization(self, tokens: np.ndarray, grid_size: Tuple[int, int], 
                                   resized_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        특징 임베딩 시각화 생성
        
        Args:
            tokens: 특징 토큰
            grid_size: 그리드 크기
            resized_mask: 선택적 마스크
            
        Returns:
            시각화 이미지 (numpy 배열)
        """
        pca = PCA(n_components=3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        normalized_tokens = (reduced_tokens-np.min(reduced_tokens))/(np.max(reduced_tokens)-np.min(reduced_tokens))
        return normalized_tokens
    
    def find_matches(self, features1: np.ndarray, features2: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        두 특징 집합 간의 일치 항목 찾기
        
        Args:
            features1: 첫 번째 특징 집합
            features2: 두 번째 특징 집합
            k: 각 쿼리에 대해 찾을 이웃 수
            
        Returns:
            (distances, matches) 튜플
            - distances: 각 일치 항목의 거리
            - matches: features1의 인덱스에 대응하는 features2의 인덱스
        """
        # 특징 차원 확인
        d = features1.shape[1]
        
        # FAISS 인덱스 생성
        index = faiss.IndexFlatL2(d)
        
        # 특징 정규화 및 추가
        features1 = features1.astype(np.float32)
        faiss.normalize_L2(features1)
        index.add(features1)
        
        # 쿼리 특징 정규화
        features2 = features2.astype(np.float32)
        faiss.normalize_L2(features2)
        
        # FAISS 검색 수행
        distances, matches = index.search(features2, k)
        
        return distances, matches