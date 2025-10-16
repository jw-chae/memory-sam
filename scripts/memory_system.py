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
from scripts.memory_repository import MemoryRepository
from scripts.logger import get_logger

class MemorySystem:
    """Memory system to store and retrieve image-mask pairs based on feature similarity"""
    
    def __init__(self, memory_dir: str = "memory"):
        """
        Initialize memory system
        
        Args:
            memory_dir (str): Directory to store memory items
        """
        self.repo = MemoryRepository(memory_dir)
        self.memory_dir = self.repo.memory_dir
        self.index = self.repo.load_index()
        self.log = get_logger("MemorySystem")
        
        # Initialize FAISS index
        self.faiss_index_path = self.memory_dir / "faiss_index.bin"
        self.feature_dim = None
        self.faiss_index = None
        self.id_to_index_map = {}  # Map memory ID to FAISS index
        
        # Clean up orphaned index entries
        self.validate_and_clean_index()
        
        # Build FAISS index from existing items
        self._build_faiss_index()
    
    def _build_faiss_index(self):
        """Build FAISS index from existing memory items"""
        if not self.index["items"]:
            return
        
        # Determine feature dimension from first item
        first_item = self.index["items"][0]
        if "features_path" not in first_item:
            return  # No features available
        first_features_path = self.memory_dir / first_item["features_path"]
        if first_features_path.exists():
            first_features = np.load(str(first_features_path), allow_pickle=True)
            self.feature_dim = first_features.shape[0]
            
            # Create FAISS index
            self.faiss_index = faiss.IndexFlatL2(self.feature_dim)
            
            # Add all items
            for idx, item in enumerate(self.index["items"]):
                try:
                    if "features_path" not in item:
                        continue  # Skip items without features
                    features_path = self.memory_dir / item["features_path"]
                    
                    # Check if file exists before loading
                    if not features_path.exists():
                        print(f"Warning: Features file not found for item ID {item['id']}, skipping")
                        continue
                    
                    features = np.load(str(features_path), allow_pickle=True)
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
        self.repo.save_index(self.index)
    
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
        
        # 항목 디렉토리 생성 및 저장 (Repository 사용)
        item_dir = self.repo.create_item_dir(memory_id)
        image_rel = self.repo.save_image(item_dir, image)
        mask_rel = self.repo.save_mask(item_dir, mask)
        features_rel = self.repo.save_features(item_dir, features)  # Can be None
        patch_features_rel = self.repo.save_patch_features(item_dir, patch_features, grid_size, resize_scale)
        
        # 메타데이터 저장 (있는 경우)
        if metadata is not None:
            with open(item_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f)
        
        # 타임스탬프 생성
        timestamp = datetime.now().isoformat()
        
        # 인덱스 항목 생성
        item = {
            "id": memory_id,
            "image_path": image_rel,
            "mask_path": mask_rel,
            "created_at": timestamp
        }
        
        # features_path 추가 (있는 경우만)
        if features_rel is not None:
            item["features_path"] = features_rel
        
        # 패치 특징 경로 추가 (있는 경우)
        if patch_features_rel is not None:
            item["patch_features_path"] = patch_features_rel
        
        # 메타데이터 추가 (있는 경우)
        if metadata is not None:
            item["metadata"] = metadata
        
        # 인덱스에 항목 추가
        self.index["items"].append(item)
        self._save_index()
        
        # FAISS 인덱스에 추가 (features가 제공된 경우만)
        if features is not None:
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
    
    def get_most_similar_sparse(self, query_patch_features: np.ndarray, grid_size: Tuple[int, int], 
                               mask: Optional[np.ndarray] = None, top_k: int = 1, 
                               match_background: bool = True) -> List[Dict]:
        """
        Get most similar memory items using sparse patch features, simplified for robustness.
        """
        if not self.index["items"]:
            return []

        similar_items = []

        for item in self.index["items"]:
            try:
                if "patch_features_path" not in item:
                    continue

                item_data = self.load_item_data(item["id"])
                item_patch_features = item_data.get("patch_features")
                item_mask = item_data.get("mask")

                if item_patch_features is None:
                    continue
                
                # Reshape features from (C, H, W) to (H*W, C)
                query_features_flat = query_patch_features.reshape(query_patch_features.shape[0], -1).T
                item_features_flat = item_patch_features.reshape(item_patch_features.shape[0], -1).T

                fg_sim = 0.0
                bg_sim = 0.0

                # Foreground matching
                if mask is not None and item_mask is not None:
                    query_fg_features = self._get_masked_features(query_features_flat, mask, grid_size, invert=False)
                    item_fg_features = self._get_masked_features(item_features_flat, item_mask, item_data.get("grid_size"), invert=False)
                    if query_fg_features is not None and item_fg_features is not None:
                        fg_sim = self._calculate_feature_similarity(query_fg_features, item_fg_features)
                
                # Background matching
                if match_background and mask is not None and item_mask is not None:
                    query_bg_features = self._get_masked_features(query_features_flat, mask, grid_size, invert=True)
                    item_bg_features = self._get_masked_features(item_features_flat, item_mask, item_data.get("grid_size"), invert=True)
                    if query_bg_features is not None and item_bg_features is not None:
                        bg_sim = self._calculate_feature_similarity(query_bg_features, item_bg_features)

                # If no masks, compare all features
                if mask is None or item_mask is None:
                    fg_sim = self._calculate_feature_similarity(query_features_flat, item_features_flat)
                
                # Final similarity score
                final_similarity = fg_sim + (bg_sim * 0.2) # Give less weight to background

                if final_similarity > 0:
                    similar_items.append({
                        "item": item,
                        "similarity": float(final_similarity),
                    })

            except Exception as e:
                import traceback
                print(f"Error processing item ID {item['id']} for sparse matching: {e}")
                traceback.print_exc()

        similar_items.sort(key=lambda x: x["similarity"], reverse=True)
        return similar_items[:top_k]

    def _get_masked_features(self, features_flat, mask, grid_size, invert=False):
        if grid_size is None: return None
        
        # Ensure mask is single channel
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        mask_resized = cv2.resize(mask.astype(np.uint8), (grid_size[1], grid_size[0])).astype(bool)
        
        if invert:
            mask_resized = ~mask_resized
            
        indices = np.where(mask_resized.flatten())[0]
        
        if len(indices) == 0 or len(indices) > len(features_flat): return None

        return features_flat[indices]

    def _calculate_feature_similarity(self, features1, features2):
        if features1 is None or features2 is None or len(features1) == 0 or len(features2) == 0:
            return 0.0

        # Ensure arrays are C-contiguous and float32 for Faiss/sklearn
        f1 = np.ascontiguousarray(features1, dtype=np.float32)
        f2 = np.ascontiguousarray(features2, dtype=np.float32)

        # Normalize features
        faiss.normalize_L2(f1)
        faiss.normalize_L2(f2)

        # Use NearestNeighbors to find average distance (as a measure of similarity)
        # We find the 2 nearest neighbors to avoid matching a point with itself if sets are identical
        k = min(2, len(f2))
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(f2)
        distances, _ = nbrs.kneighbors(f1)
        
        avg_distance = np.mean(distances[:, 0]) # Use the distance to the 1st nearest neighbor
        
        # Convert distance to similarity (e.g., exponential decay)
        similarity = np.exp(-avg_distance * 5.0)
        return similarity
    
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
        
        result = {
            "item": item,
            "image": image,
            "mask": mask,
        }
        
        # 특징 로드 (있는 경우만)
        if "features_path" in item:
            features_path = self.memory_dir / item["features_path"]
            try:
                features = np.load(str(features_path), allow_pickle=True)
                result["features"] = features
            except Exception as e:
                print(f"Warning: Failed to load features for item {item_id}: {e}")
                # features 없이 계속 진행
        
        # 패치 특징 로드 (있는 경우)
        if "patch_features_path" in item:
            patch_features_path = self.memory_dir / item["patch_features_path"]
            if os.path.exists(patch_features_path):
                try:
                    patch_features = np.load(str(patch_features_path), allow_pickle=True)
                    result["patch_features"] = patch_features
                    
                    # 패치 정보 로드
                    patch_info_path = self.memory_dir / Path(item["patch_features_path"]).parent / "patch_info.json"
                    if patch_info_path.exists():
                        with open(patch_info_path, 'r') as f:
                            patch_info = json.load(f)
                        result["grid_size"] = tuple(patch_info["grid_size"])
                        result["resize_scale"] = patch_info["resize_scale"]
                except Exception as e:
                    print(f"Warning: Failed to load patch features for item {item_id}: {e}")
        
        return result
    
    def get_all_items(self) -> List[Dict]:
        """모든 메모리 항목 가져오기"""
        return self.index["items"]
    
    def validate_and_clean_index(self) -> int:
        """
        인덱스를 검증하고 실제로 존재하지 않는 항목을 제거
        
        Returns:
            제거된 항목의 수
        """
        items_to_remove = []
        
        for item in self.index["items"]:
            item_dir = self.memory_dir / f"item_{item['id']}"
            image_path = self.memory_dir / item["image_path"]
            
            # 이미지 파일이나 디렉토리가 존재하지 않으면 제거 대상에 추가
            if not item_dir.exists() or not image_path.exists():
                items_to_remove.append(item['id'])
                print(f"Found orphaned index entry for item {item['id']} (files missing)")
        
        # 인덱스에서 제거
        removed_count = 0
        for item_id in items_to_remove:
            for idx, item in enumerate(self.index["items"]):
                if item["id"] == item_id:
                    self.index["items"].pop(idx)
                    removed_count += 1
                    
                    # FAISS 인덱스에서도 제거
                    if item_id in self.id_to_index_map:
                        self.id_to_index_map.pop(item_id)
                    break
        
        if removed_count > 0:
            self._save_index()
            # Don't rebuild FAISS index here to avoid errors
            # It will be rebuilt when needed
            print(f"Cleaned {removed_count} orphaned entries from index")
        
        return removed_count
    
    def delete_memory(self, item_id: int) -> None:
        """
        메모리 항목 삭제
        
        Args:
            item_id: 삭제할 항목의 ID
        """
        import shutil
        
        # Find item in index
        item = None
        item_index = None
        for idx, i in enumerate(self.index["items"]):
            if i["id"] == item_id:
                item = i
                item_index = idx
                break
        
        if item is None:
            raise ValueError(f"Item ID {item_id} not found")
        
        # Delete from filesystem
        item_dir = self.memory_dir / f"item_{item_id}"
        if item_dir.exists():
            try:
                shutil.rmtree(item_dir)
            except Exception as e:
                print(f"Warning: Failed to delete item directory {item_id}: {e}")
        
        # Remove from index
        self.index["items"].pop(item_index)
        self._save_index()
        
        # Remove from FAISS index and rebuild
        if item_id in self.id_to_index_map:
            self.id_to_index_map.pop(item_id)
        
        # Always rebuild FAISS index after deletion
        try:
            self._rebuild_faiss_index()
        except Exception as e:
            print(f"Warning: Failed to rebuild FAISS index: {e}")
        
        print(f"Deleted memory item ID {item_id}")
    
    def delete_all_memory(self) -> None:
        """모든 메모리 항목 삭제"""
        import shutil
        
        # Delete all item directories
        for item in self.index["items"]:
            item_dir = self.memory_dir / f"item_{item['id']}"
            if item_dir.exists():
                try:
                    shutil.rmtree(item_dir)
                except Exception as e:
                    print(f"Warning: Failed to delete item directory {item['id']}: {e}")
        
        # Reset index
        self.index["items"] = []
        self.index["next_id"] = 0
        self._save_index()
        
        # Reset FAISS index
        self.faiss_index = None
        self.id_to_index_map = {}
        try:
            if self.faiss_index_path.exists():
                self.faiss_index_path.unlink()
        except Exception as e:
            print(f"Warning: Failed to delete FAISS index: {e}")
        
        print("Deleted all memory items")
    
    def _rebuild_faiss_index(self):
        """Rebuild FAISS index from scratch"""
        if not self.index["items"]:
            self.faiss_index = None
            self.id_to_index_map = {}
            if self.faiss_index_path.exists():
                self.faiss_index_path.unlink()
            return
        
        # Reset and rebuild
        self.faiss_index = None
        self.id_to_index_map = {}
        self._build_faiss_index()
    
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
            # 피처 차원 확인 및 디버깅 정보 출력
            print(f"Query features shape: {query_features.shape}, Item features shape: {item_features.shape}")
            
            # 차원이 다른 경우 처리
            if query_features.shape != item_features.shape:
                print(f"Warning: Feature shape mismatch. Query: {query_features.shape}, Item: {item_features.shape}")
                # 더 작은 차원에 맞춰 조정
                min_shape = tuple(min(q, i) for q, i in zip(query_features.shape, item_features.shape))
                
                if len(query_features.shape) == 3:  # (C, H, W) 형태
                    query_features = query_features[:min_shape[0], :min_shape[1], :min_shape[2]]
                    item_features = item_features[:min_shape[0], :min_shape[1], :min_shape[2]]
                elif len(query_features.shape) == 2:  # (N, C) 형태
                    query_features = query_features[:min_shape[0], :min_shape[1]]
                    item_features = item_features[:min_shape[0], :min_shape[1]]
                
                print(f"Adjusted shapes - Query: {query_features.shape}, Item: {item_features.shape}")
            
            # 패치 피처를 1D로 평탄화 (3D인 경우)
            if len(query_features.shape) == 3:
                # (C, H, W) -> (H*W, C)
                C, H, W = query_features.shape
                query_flat = query_features.reshape(C, H*W).T  # (H*W, C)
                item_flat = item_features.reshape(C, H*W).T    # (H*W, C)
            else:
                # 이미 2D인 경우
                query_flat = query_features
                item_flat = item_features
            
            # 샘플링으로 계산량 줄이기 (결정적 간격 샘플링)
            max_features = min(len(query_flat), len(item_flat), 100)

            def uniform_sample(arr, k):
                if len(arr) <= k:
                    return arr
                # 균등 간격 인덱스 선택 (결정적)
                idx = np.linspace(0, len(arr) - 1, num=k, dtype=int)
                return arr[idx]

            sampled_query = uniform_sample(query_flat, max_features)
            sampled_item = uniform_sample(item_flat, max_features)
            
            # 피처 정규화
            normalized_query = np.zeros_like(sampled_query)
            normalized_item = np.zeros_like(sampled_item)
            
            for i in range(len(normalized_query)):
                norm = np.linalg.norm(sampled_query[i])
                if norm > 0:
                    normalized_query[i] = sampled_query[i] / norm
                    
            for i in range(len(normalized_item)):
                norm = np.linalg.norm(sampled_item[i])
                if norm > 0:
                    normalized_item[i] = sampled_item[i] / norm
            
            # 차원 확인 후 유사도 행렬 계산
            if normalized_query.shape[1] != normalized_item.shape[1]:
                print(f"Error: Feature dimension mismatch after normalization. Query: {normalized_query.shape}, Item: {normalized_item.shape}")
                return 0.0
            
            # 유사도 행렬 계산
            similarity_matrix = np.matmul(normalized_query, normalized_item.T)
            
            # 각 쿼리 피처에 대해 가장 유사한 항목 피처의 유사도 추출
            best_similarities = np.max(similarity_matrix, axis=1)
            
            # 평균 유사도 계산
            mean_similarity = np.mean(best_similarities)
            
            return float(mean_similarity)
            
        except Exception as e:
            import traceback
            print(f"패치 피처 비교 중 오류: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return 0.0