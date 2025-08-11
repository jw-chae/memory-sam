import json
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.neighbors import NearestNeighbors
import faiss

class MemorySystem:
    """Memory system to store and retrieve image-mask pairs based on feature similarity"""
    
    def __init__(self, memory_dir: str = "memory"):
        """
        Initialize memory system
        
        Args:
            memory_dir (str): Directory to store memory items
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True, parents=True)
        
        # Index file
        self.index_path = self.memory_dir / "index.json"
        
        # Load or initialize index
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                "items": [],
                "next_id": 0
            }
            self._save_index()
        
        # Initialize FAISS index
        self.faiss_index_path = self.memory_dir / "faiss_index.bin"
        self.feature_dim = None
        self.faiss_index = None
        self.id_to_index_map = {}  # Map memory ID to FAISS index
        
        # Build FAISS index from existing items
        self._build_faiss_index()
    
    def _build_faiss_index(self):
        """Build FAISS index from existing memory items"""
        if not self.index["items"]:
            return
        
        # Determine feature dimension from first item
        first_item = self.index["items"][0]
        first_features_path = self.memory_dir / first_item["features_path"]
        if first_features_path.exists():
            first_features = np.load(str(first_features_path))
            self.feature_dim = first_features.shape[0]
            
            # Create FAISS index
            self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
            
            # Add all items
            for idx, item in enumerate(self.index["items"]):
                try:
                    features_path = self.memory_dir / item["features_path"]
                    features = np.load(str(features_path))
                    features = self._normalize_features(features)
                    
                    # L2 normalization (important for FAISS search)
                    features = features.reshape(1, -1).astype(np.float32)
                    faiss.normalize_L2(features)
                    
                    # Add to index
                    self.faiss_index.add(features)
                    self.id_to_index_map[item["id"]] = idx
                except Exception as e:
                    print(f"Error building FAISS index for item ID {item['id']}: {e}")
            
            # Save index
            if self.faiss_index.ntotal > 0:
                faiss.write_index(self.faiss_index, str(self.faiss_index_path))
                print(f"FAISS index built with {self.faiss_index.ntotal} items")
    
    def _save_index(self):
        """Save index to disk"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def add_memory(self, 
                  image: np.ndarray, 
                  mask: np.ndarray, 
                  features: np.ndarray,
                  patch_features: Optional[np.ndarray] = None,
                  grid_size: Optional[Tuple[int, int]] = None,
                  resize_scale: Optional[float] = None,
                  metadata: Dict = None) -> int:
        """
        Add new image-mask pair to memory
        
        Args:
            image: Original image (numpy array)
            mask: Segmentation mask (numpy array)
            features: DINOv2 global features (numpy array)
            patch_features: DINOv2 patch features (optional)
            grid_size: Feature grid size (optional)
            resize_scale: Resize scale (optional)
            metadata: Optional metadata
            
        Returns:
            ID of the saved memory item
        """
        memory_id = self.index["next_id"]
        self.index["next_id"] += 1
        
        # 항목 디렉토리 생성
        item_dir = self.memory_dir / f"item_{memory_id}"
        item_dir.mkdir(exist_ok=True)
        
        # 이미지, 마스크, 특징 저장
        image_path = item_dir / "image.png"
        mask_path = item_dir / "mask.png"
        features_path = item_dir / "features.npy"
        
        # 이미지 저장
        image_pil = Image.fromarray(image)
        image_pil.save(str(image_path))
        
        # 마스크 저장
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask)
        mask_pil.save(str(mask_path))
        
        # 특징 저장
        np.save(str(features_path), features)
        
        # 패치 특징 저장 (있는 경우)
        patch_features_path = None
        if patch_features is not None:
            patch_features_path = item_dir / "patch_features.npy"
            np.save(str(patch_features_path), patch_features)
            
            # 그리드 크기와 크기 조정 비율 저장
            if grid_size is not None and resize_scale is not None:
                with open(item_dir / "patch_info.json", 'w') as f:
                    json.dump({
                        "grid_size": grid_size,
                        "resize_scale": resize_scale
                    }, f)
        
        # 메타데이터 저장 (있는 경우)
        if metadata is not None:
            with open(item_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f)
        
        # 타임스탬프 생성
        timestamp = datetime.now().isoformat()
        
        # 인덱스 항목 생성
        item = {
            "id": memory_id,
            "image_path": str(image_path.relative_to(self.memory_dir)),
            "mask_path": str(mask_path.relative_to(self.memory_dir)),
            "features_path": str(features_path.relative_to(self.memory_dir)),
            "created_at": timestamp
        }
        
        # 패치 특징 경로 추가 (있는 경우)
        if patch_features_path is not None:
            item["patch_features_path"] = str(patch_features_path.relative_to(self.memory_dir))
        
        # 메타데이터 추가 (있는 경우)
        if metadata is not None:
            item["metadata"] = metadata
        
        # 인덱스에 항목 추가
        self.index["items"].append(item)
        self._save_index()
        
        # FAISS 인덱스에 추가
        if self.faiss_index is None:
            # 첫 번째 항목이면 인덱스 초기화
            self.feature_dim = features.shape[0]
            self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
        
        # 특징 정규화 및 추가
        normalized_features = self._normalize_features(features)
        normalized_features = normalized_features.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(normalized_features)
        self.faiss_index.add(normalized_features)
        
        # ID 매핑 업데이트
        self.id_to_index_map[memory_id] = len(self.id_to_index_map)
        
        # 인덱스 저장
        faiss.write_index(self.faiss_index, str(self.faiss_index_path))
        
        return memory_id
    
    def get_most_similar(self, features: np.ndarray, top_k: int = 1, method: str = "global") -> List[Dict]:
        """
        특징 유사성에 기반하여 메모리에서 가장 유사한 항목 찾기
        
        Args:
            features: 쿼리 특징
            top_k: 반환할 유사 항목 수
            method: 유사도 계산 방법 ("global" 또는 "sparse")
            
        Returns:
            가장 유사한 메모리 항목 목록
        """
        if not self.index["items"]:
            return []
        
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print("FAISS index is empty. Fallback to legacy method.")
            return self._get_most_similar_legacy(features, top_k, method)
        
        # 쿼리 특징 정규화
        normalized_features = self._normalize_features(features)
        normalized_features = normalized_features.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(normalized_features)
        
        print(f"Normalized query feature norm: {np.linalg.norm(normalized_features):.6f}")
        
        # FAISS search
        k = min(top_k, self.faiss_index.ntotal)
        distances, indices = self.faiss_index.search(normalized_features, k)
        
        # Result conversion
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            distance = distances[0][i]
            
            # Find item ID from FAISS index
            item_id = None
            for id, index in self.id_to_index_map.items():
                if index == idx:
                    item_id = id
                    break
            
            if item_id is not None:
                item = self.get_item(item_id)
                # Convert distance to similarity (smaller distance means higher similarity)
                similarity = 1.0 / (1.0 + distance)
                print(f"Item ID {item['id']} similarity: {similarity:.6f}, distance: {distance:.6f}")
                results.append({"similarity": similarity, "item": item})
        
        # Sort results by similarity (descending)
        results.sort(reverse=True, key=lambda x: x["similarity"])
        
        return results
    
    def _get_most_similar_legacy(self, features: np.ndarray, top_k: int = 1, method: str = "global") -> List[Dict]:
        """Fallback to legacy method when FAISS index is not available"""
        # Normalize query feature
        normalized_features = self._normalize_features(features)
        print(f"Normalized query feature norm: {np.linalg.norm(normalized_features):.6f}")
        
        similarities = []
        
        for item in self.index["items"]:
            try:
                # Load item feature
                item_features_path = self.memory_dir / item["features_path"]
                item_features = np.load(str(item_features_path))
                
                # Normalize item feature
                normalized_item_features = self._normalize_features(item_features)
                
                # Calculate similarity (default: cosine similarity)
                if method == "global":
                    similarity = self._cosine_similarity(normalized_features, normalized_item_features)
                else:
                    # Simple fallback for sparse matching
                    similarity = self._cosine_similarity(normalized_features, normalized_item_features)
                
                print(f"Item ID {item['id']} similarity: {similarity:.6f}, feature norm: {np.linalg.norm(item_features):.6f}")
                
                similarities.append((similarity, item))
            except Exception as e:
                print(f"Error processing item ID {item['id']}: {e}")
                continue
        
        # Sort similarities by similarity (descending)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # Return top_k most similar items
        return [{"similarity": sim, "item": item} for sim, item in similarities[:top_k]]
    
    def get_most_similar_sparse(self, patch_features: np.ndarray, mask: Optional[np.ndarray] = None, 
                               grid_size: Optional[Tuple[int, int]] = None, top_k: int = 1, 
                               match_threshold: float = 0.8, match_background: bool = True) -> List[Dict]:
        """
        Get most similar memory items using sparse patch features
        
        Args:
            patch_features: Query patch features (N x D)
            mask: Optional mask for foreground/background separation
            grid_size: Feature grid size
            top_k: Number of top items to return
            match_threshold: Minimum similarity for matching
            match_background: Whether to match background areas
            
        Returns:
            List of similar items with similarity scores
        """
        if not self.index["items"]:
            return []
        
        # 쿼리 패치 특징 정규화
        query_features = patch_features.copy()
        
        # 결과 리스트
        similar_items = []
        
        # 마스크가 있는 경우 전경/배경 분리
        if mask is not None and grid_size is not None:
            # 마스크를 그리드 크기로 조정
            grid_h, grid_w = grid_size
            mask_resized = cv2.resize(
                mask.astype(np.uint8),
                (grid_w, grid_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            
            # 전경/배경 인덱스 가져오기
            fg_indices = np.where(mask_resized.flatten())[0]
            bg_indices = np.where(~mask_resized.flatten())[0]
            
            # 전경/배경 특징만큼 자르기
            fg_query_features = query_features[fg_indices] if len(fg_indices) > 0 else None
            bg_query_features = query_features[bg_indices] if len(bg_indices) > 0 and match_background else None
        else:
            fg_query_features = query_features
            bg_query_features = None
        
        # 각 메모리 항목과 비교
        for item in self.index["items"]:
            try:
                item_id = item["id"]
                
                # 패치 특징이 있는지 확인
                if "patch_features_path" not in item:
                    continue
                
                # 패치 특징 로드
                patch_features_path = self.memory_dir / item["patch_features_path"]
                
                # 절대 경로로 변환 (존재하지 않는 경우를 위한 예외 처리)
                patch_features_path = os.path.abspath(patch_features_path)
                
                # 파일이 존재하는지 확인 (더 명확한 에러 메시지)
                if not os.path.exists(patch_features_path):
                    print(f"패치 특징 파일이 존재하지 않습니다: {patch_features_path}")
                    continue
                
                try:
                    # 패치 특징 로드 시도
                    item_patch_features = np.load(str(patch_features_path))
                except Exception as e:
                    print(f"Error processing item ID {item_id}: {e}")
                    continue
                
                # 패치 정보 로드
                item_grid_size = None
                
                # 패치 정보 경로 구성
                patch_info_path = self.memory_dir / Path(item["patch_features_path"]).parent / "patch_info.json"
                
                # 정보 파일 존재 확인
                if os.path.exists(patch_info_path):
                    try:
                        with open(patch_info_path, 'r') as f:
                            patch_info = json.load(f)
                            item_grid_size = tuple(patch_info["grid_size"])
                    except Exception as e:
                        print(f"패치 정보를 읽는 중 오류: {e}")
                
                # 마스크 로드 (전경/배경 분리용)
                item_mask = None
                if mask is not None and grid_size is not None and item_grid_size is not None:
                    try:
                        # 마스크 파일 경로
                        mask_path = self.memory_dir / item["mask_path"]
                        
                        # 절대 경로로 변환
                        mask_path = os.path.abspath(mask_path)
                        
                        # 마스크 로드
                        if os.path.exists(mask_path):
                            item_mask = np.array(Image.open(mask_path))
                            
                            # 다중 채널 마스크 처리
                            if len(item_mask.shape) > 2:
                                item_mask = item_mask[:, :, 0]
                            
                            # 이진 마스크로 변환
                            item_mask = item_mask > 0
                            
                            # 그리드 크기로 조정
                            item_mask_resized = cv2.resize(
                                item_mask.astype(np.uint8),
                                (item_grid_size[1], item_grid_size[0]),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                            
                            # 전경/배경 인덱스 가져오기
                            item_fg_indices = np.where(item_mask_resized.flatten())[0]
                            item_bg_indices = np.where(~item_mask_resized.flatten())[0]
                            
                            # 유효성 검사
                            if len(item_fg_indices) == 0:
                                print(f"항목 {item_id} 마스크에 전경이 없습니다.")
                        else:
                            print(f"마스크 파일을 찾을 수 없습니다: {mask_path}")
                            continue
                            
                    except Exception as e:
                        print(f"마스크 처리 중 오류: {e}")
                        continue
                
                # 전경/배경 분리 매칭
                fg_similarity = 0.0
                bg_similarity = 0.0
                
                # 전경 매칭
                if fg_query_features is not None and item_mask is not None:
                    try:
                        # 아이템 전경 특징
                        item_fg_features = item_patch_features[item_fg_indices]
                        
                        if len(item_fg_features) > 0 and len(fg_query_features) > 0:
                            # 모든 쿼리 전경 특징에 대해 각 아이템 전경 특징의 최대 유사도 계산
                            similarities = np.zeros(len(fg_query_features))
                            for i, query_feat in enumerate(fg_query_features):
                                # 정규화
                                query_feat_norm = query_feat / np.linalg.norm(query_feat)
                                item_features_norm = item_fg_features / np.linalg.norm(item_fg_features, axis=1, keepdims=True)
                                
                                # 코사인 유사도 계산
                                cosine_sims = np.dot(item_features_norm, query_feat_norm)
                                similarities[i] = np.max(cosine_sims)
                            
                            # 매치율 계산 (임계값 이상 유사한 특징의 비율)
                            fg_match_ratio = np.mean(similarities >= match_threshold)
                            fg_similarity = np.mean(similarities) * fg_match_ratio
                            
                            print(f"Item ID {item_id} foreground similarity: {fg_similarity:.6f}, match ratio: {fg_match_ratio:.6f}")
                    except Exception as e:
                        print(f"전경 매칭 중 오류: {e}")
                        fg_similarity = 0.0
                
                # 배경 매칭 (활성화된 경우)
                if bg_query_features is not None and item_mask is not None and match_background:
                    try:
                        # 아이템 배경 특징
                        item_bg_features = item_patch_features[item_bg_indices]
                        
                        if len(item_bg_features) > 0 and len(bg_query_features) > 0:
                            # 모든 쿼리 배경 특징에 대해 각 아이템 배경 특징의 최대 유사도 계산
                            similarities = np.zeros(len(bg_query_features))
                            for i, query_feat in enumerate(bg_query_features):
                                # 정규화
                                query_feat_norm = query_feat / np.linalg.norm(query_feat)
                                item_features_norm = item_bg_features / np.linalg.norm(item_bg_features, axis=1, keepdims=True)
                                
                                # 코사인 유사도 계산
                                cosine_sims = np.dot(item_features_norm, query_feat_norm)
                                similarities[i] = np.max(cosine_sims)
                            
                            # 매치율 계산 (임계값 이상 유사한 특징의 비율)
                            bg_match_ratio = np.mean(similarities >= match_threshold)
                            bg_similarity = np.mean(similarities) * bg_match_ratio
                            
                            print(f"Item ID {item_id} background similarity: {bg_similarity:.6f}, match ratio: {bg_match_ratio:.6f}")
                    except Exception as e:
                        print(f"배경 매칭 중 오류: {e}")
                        bg_similarity = 0.0
                
                # 최종 유사도 계산 (전경 70%, 배경 30% 가중치)
                if mask is not None and item_mask is not None:
                    if match_background:
                        final_similarity = 0.7 * fg_similarity + 0.3 * bg_similarity
                    else:
                        final_similarity = fg_similarity
                else:
                    # 전체 특징 매칭
                    final_similarity = self._compare_patch_features(query_features, item_patch_features)
                
                print(f"Item ID {item_id} final similarity: {final_similarity:.6f}")
                
                # 결과 리스트에 추가
                if final_similarity > 0:
                    similar_items.append({
                        "item": item,
                        "similarity": float(final_similarity),
                        "has_patch_features": True
                    })
            except Exception as e:
                import traceback
                print(f"Error processing item ID {item['id']}: {e}")
                traceback.print_exc()
        
        # 유사도로 정렬
        similar_items.sort(key=lambda x: x["similarity"], reverse=True)
        
        # top_k 반환
        return similar_items[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        # Vector normalization (NaN prevention)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Prevent division by zero
        if norm_a == 0 or norm_b == 0:
            print("Warning: 0 norm vector detected. Returning similarity 0")
            return 0.0
        
        # Calculate cosine similarity
        similarity = np.dot(a, b) / (norm_a * norm_b)
        
        # NaN or infinite value handling
        if np.isnan(similarity) or np.isinf(similarity):
            print(f"Warning: {similarity} value detected in similarity calculation. Returning 0")
            return 0.0
            
        return similarity
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Feature vector normalization"""
        # Check if feature vector is empty
        if features.size == 0:
            return features
            
        # L2 normalization
        norm = np.linalg.norm(features)
        if norm > 0:
            return features / norm
        return features
    
    def get_item(self, item_id: int) -> Dict:
        """Search memory item by ID"""
        for item in self.index["items"]:
            if item["id"] == item_id:
                return item
        raise ValueError(f"Item ID {item_id} not found")
    
    def load_item_data(self, item_id: int) -> Dict:
        """Load all data of a memory item"""
        item = self.get_item(item_id)
        
        # Load image
        image_path = self.memory_dir / item["image_path"]
        image = np.array(Image.open(str(image_path)))
        
        # Load mask
        mask_path = self.memory_dir / item["mask_path"]
        mask = np.array(Image.open(str(mask_path)))
        
        # 특징 로드
        features_path = self.memory_dir / item["features_path"]
        features = np.load(str(features_path))
        
        result = {
            "item": item,
            "image": image,
            "mask": mask,
            "features": features
        }
        
        # 패치 특징 로드 (있는 경우)
        if "patch_features_path" in item:
            patch_features_path = self.memory_dir / item["patch_features_path"]
            if os.path.exists(patch_features_path):
                patch_features = np.load(str(patch_features_path))
                result["patch_features"] = patch_features
                
                # 패치 정보 로드
                patch_info_path = self.memory_dir / Path(item["patch_features_path"]).parent / "patch_info.json"
                if patch_info_path.exists():
                    with open(patch_info_path, 'r') as f:
                        patch_info = json.load(f)
                    result["grid_size"] = tuple(patch_info["grid_size"])
                    result["resize_scale"] = patch_info["resize_scale"]
        
        return result
    
    def get_all_items(self) -> List[Dict]:
        """모든 메모리 항목 가져오기"""
        return self.index["items"]
    
    def _compare_patch_features(self, query_features: np.ndarray, item_features: np.ndarray) -> float:
        """
        두 패치 피처 세트 간의 유사도를 계산합니다.
        
        Args:
            query_features: 쿼리 이미지의 패치 피처
            item_features: 메모리 항목의 패치 피처
            
        Returns:
            유사도 점수 (0~1 범위)
        """
        if len(query_features) == 0 or len(item_features) == 0:
            return 0.0
        
        try:
            # 쿼리 피처와 항목 피처 중 더 적은 수를 기준으로 샘플링
            max_features = min(len(query_features), len(item_features), 100)  # 최대 100개
            
            if len(query_features) > max_features:
                # 랜덤 샘플링
                indices = np.random.choice(len(query_features), max_features, replace=False)
                sampled_query_features = query_features[indices]
            else:
                sampled_query_features = query_features
                
            if len(item_features) > max_features:
                # 랜덤 샘플링
                indices = np.random.choice(len(item_features), max_features, replace=False)
                sampled_item_features = item_features[indices]
            else:
                sampled_item_features = item_features
            
            # 피처 정규화
            normalized_query = np.zeros_like(sampled_query_features)
            normalized_item = np.zeros_like(sampled_item_features)
            
            for i in range(len(normalized_query)):
                norm = np.linalg.norm(sampled_query_features[i])
                if norm > 0:
                    normalized_query[i] = sampled_query_features[i] / norm
                    
            for i in range(len(normalized_item)):
                norm = np.linalg.norm(sampled_item_features[i])
                if norm > 0:
                    normalized_item[i] = sampled_item_features[i] / norm
            
            # 유사도 행렬 계산
            similarity_matrix = np.matmul(normalized_query, normalized_item.T)
            
            # 각 쿼리 피처에 대해 가장 유사한 항목 피처의 유사도 추출
            best_similarities = np.max(similarity_matrix, axis=1)
            
            # 평균 유사도 계산
            mean_similarity = np.mean(best_similarities)
            
            return float(mean_similarity)
            
        except Exception as e:
            print(f"패치 피처 비교 중 오류: {e}")
            return 0.0