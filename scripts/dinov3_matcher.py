import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from typing import Tuple, Optional

# FAISS가 없는 환경에서도 동작하도록 안전한 폴백 추가
try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None  # type: ignore
    _FAISS_AVAILABLE = False

class Dinov3Matcher:
    """DINOv3 기반 이미지 매칭을 위한 클래스"""

    def __init__(self, 
                repo_name="facebookresearch/dinov3", 
                model_name="dinov3_vitl16", 
                smaller_edge_size=448, 
                half_precision=False, 
                device="cuda"):
        """
        DINOv3 Matcher 초기화
        
        Args:
            repo_name: DINOv3 모델이 있는 리포지토리 이름
            model_name: 사용할 DINOv3 모델 이름
            smaller_edge_size: 입력 이미지의 작은 쪽 크기 조정
            half_precision: 절반 정밀도 사용 여부 (메모리 절약)
            device: 사용할 장치 ('cuda' 또는 'cpu')
        """
        self.repo_name = repo_name
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.device = device

        # 모델 로드 - DINOv3 사전 훈련된 가중치 사용
        import sys
        import os
        dinov3_path = "/home/joongwon00/dinov3"
        if dinov3_path not in sys.path:
            sys.path.append(dinov3_path)
        
        from dinov3.models.vision_transformer import DinoVisionTransformer
        
        # DINOv3 ViT-L/16 모델 생성 (대형 모델)
        self.model = DinoVisionTransformer(
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
            self.model.load_state_dict(state_dict, strict=False)
            print("Pretrained weights loaded successfully")
        else:
            print("Warning: Pretrained weights not found, using random initialization")
        
        if self.half_precision:
            self.model = self.model.half().to(self.device)
        else:
            self.model = self.model.to(self.device)

        self.model.eval()

        # 이미지 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def prepare_image(self, rgb_image_numpy: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int], float]:
        """
        이미지를 DINOv3 처리를 위해 준비합니다. 종횡비를 유지하며 리사이즈하고 패딩을 추가합니다.
        """
        image = Image.fromarray(rgb_image_numpy)
        original_w, original_h = image.size

        # 종횡비를 유지하며 smaller_edge_size에 맞게 크기 계산
        scale = self.smaller_edge_size / min(original_w, original_h)
        resized_w, resized_h = int(original_w * scale), int(original_h * scale)
        image_resized = image.resize((resized_w, resized_h), Image.Resampling.BICUBIC)
        
        # 패딩을 추가하여 정사각형(smaller_edge_size x smaller_edge_size)으로 만듭니다.
        # 그러나 DINOv3는 패치 크기의 배수를 선호하므로, 먼저 패딩 크기를 계산합니다.
        # 여기서는 ToTensor 이후에 처리하는 것이 더 정확합니다.
        
        image_tensor = self.transform(image_resized)
        
        # 패치 크기의 배수가 되도록 패딩 추가
        c, h, w = image_tensor.shape
        pad_w = (self.model.patch_size - w % self.model.patch_size) % self.model.patch_size
        pad_h = (self.model.patch_size - h % self.model.patch_size) % self.model.patch_size

        padded_tensor = torch.nn.functional.pad(image_tensor, (0, pad_w, 0, pad_h))

        grid_size = (padded_tensor.shape[1] // self.model.patch_size, padded_tensor.shape[2] // self.model.patch_size)
        
        # resize_scale은 원본 대비 최종 텐서의 비율을 나타내야 하지만,
        # 패딩 때문에 복잡해지므로, 여기서는 원본 대비 리사이즈 비율을 사용합니다.
        resize_scale = original_w / resized_w

        return padded_tensor, grid_size, resize_scale
    
    def prepare_mask(self, mask_image_numpy: np.ndarray, grid_size: Tuple[int, int], resize_scale: float) -> np.ndarray:
        """
        마스크 이미지를 DINOv3 그리드 크기로 조정
        
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
        이미지 텐서에서 DINOv3 특징 추출
        
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

            # DINOv3는 get_intermediate_layers 대신 forward_features 사용
            features = self.model.forward_features(image_batch)
            # 패치 토큰만 추출 (CLS 토큰 제외)
            if hasattr(features, 'x_norm_patchtokens'):
                tokens = features.x_norm_patchtokens.squeeze()
            else:
                # 백업: 전체 특징에서 CLS 토큰 제외
                if isinstance(features, dict) and 'x_norm_patchtokens' in features:
                    tokens = features['x_norm_patchtokens'].squeeze()
                elif isinstance(features, dict) and 'x_prenorm' in features:
                    # CLS 토큰 제외하고 패치 토큰만 사용
                    tokens = features['x_prenorm'][:, 1:].squeeze()
                else:
                    # 마지막 백업: features가 텐서인 경우
                    if hasattr(features, 'shape') and len(features.shape) >= 2:
                        tokens = features[:, 1:].squeeze()  # [0]은 CLS 토큰
                    else:
                        raise ValueError(f"Unexpected features format: {type(features)}")
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
        
        # 입력 타입/정규화
        features1 = features1.astype(np.float32)
        features2 = features2.astype(np.float32)

        if _FAISS_AVAILABLE:
            # FAISS 경로 (L2)
            index = faiss.IndexFlatL2(d)
            faiss.normalize_L2(features1)
            index.add(features1)
            faiss.normalize_L2(features2)
            distances, matches = index.search(features2, k)
        else:
            # 폴백: sklearn NearestNeighbors (cosine 거리를 유사도로 대체)
            # cosine 거리(작을수록 유사) => 유사도는 1 - 거리
            nn = NearestNeighbors(n_neighbors=k, metric="cosine")
            nn.fit(features1)
            distances_cos, indices = nn.kneighbors(features2, n_neighbors=k, return_distance=True)
            # FAISS 인터페이스와 동일한 형태로 반환값 정규화
            matches = indices
            # cosine 거리 -> L2와 스케일이 다르므로, 여기서는 단순히 거리 배열을 그대로 반환
            distances = distances_cos
        
        return distances, matches
    
    def match_images(self, image1: np.ndarray, image2: np.ndarray, 
                    mask1: Optional[np.ndarray] = None, mask2: Optional[np.ndarray] = None,
                    similarity_threshold: float = 0.7, max_matches: int = 50) -> Tuple[list, list, list]:
        """
        두 이미지 간의 스파스 매칭 수행
        
        Args:
            image1: 첫 번째 이미지 (numpy 배열)
            image2: 두 번째 이미지 (numpy 배열)
            mask1: 첫 번째 이미지의 마스크 (선택사항)
            mask2: 두 번째 이미지의 마스크 (선택사항)
            similarity_threshold: 유사도 임계값
            max_matches: 최대 매칭 수
            
        Returns:
            (coords1, coords2, similarities) 튜플
            - coords1: 첫 번째 이미지의 매칭 좌표
            - coords2: 두 번째 이미지의 매칭 좌표
            - similarities: 매칭 유사도
        """
        try:
            # 이미지 전처리
            tensor1, grid_size1, resize_scale1 = self.prepare_image(image1)
            tensor2, grid_size2, resize_scale2 = self.prepare_image(image2)
            
            # 특징 추출
            features1 = self.extract_features(tensor1)
            features2 = self.extract_features(tensor2)
            
            # 마스크 처리 (선택사항)
            if mask1 is not None:
                mask1_resized = self.prepare_mask(mask1, grid_size1, resize_scale1)
                features1 = features1[mask1_resized > 0]
                valid_indices1 = np.where(mask1_resized > 0)[0]
            else:
                valid_indices1 = np.arange(len(features1))
            
            if mask2 is not None:
                mask2_resized = self.prepare_mask(mask2, grid_size2, resize_scale2)
                features2 = features2[mask2_resized > 0]
                valid_indices2 = np.where(mask2_resized > 0)[0]
            else:
                valid_indices2 = np.arange(len(features2))
            
            if len(features1) == 0 or len(features2) == 0:
                print("특징이 없어 매칭을 수행할 수 없습니다.")
                return [], [], []
            
            # 특징 정규화
            features1_norm = features1 / np.linalg.norm(features1, axis=1, keepdims=True)
            features2_norm = features2 / np.linalg.norm(features2, axis=1, keepdims=True)
            
            # 코사인 유사도 계산
            similarities_matrix = np.dot(features1_norm, features2_norm.T)
            
            # 최적 매칭 찾기
            coords1 = []
            coords2 = []
            similarities = []
            
            for i in range(len(features1)):
                best_match_idx = np.argmax(similarities_matrix[i])
                best_similarity = similarities_matrix[i][best_match_idx]
                
                if best_similarity >= similarity_threshold:
                    # 좌표 변환
                    idx1 = valid_indices1[i]
                    idx2 = valid_indices2[best_match_idx]
                    
                    row1, col1 = self.idx_to_source_position(idx1, grid_size1, resize_scale1)
                    row2, col2 = self.idx_to_source_position(idx2, grid_size2, resize_scale2)
                    
                    coords1.append((int(col1), int(row1)))
                    coords2.append((int(col2), int(row2)))
                    similarities.append(best_similarity)
            
            # 유사도 순으로 정렬
            if coords1:
                sorted_indices = np.argsort(similarities)[::-1]
                coords1 = [coords1[i] for i in sorted_indices[:max_matches]]
                coords2 = [coords2[i] for i in sorted_indices[:max_matches]]
                similarities = [similarities[i] for i in sorted_indices[:max_matches]]
            
            print(f"매칭 완료: {len(coords1)}개의 매칭 포인트 발견")
            return coords1, coords2, similarities
            
        except Exception as e:
            print(f"이미지 매칭 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return [], [], []