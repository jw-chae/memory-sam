import json
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

class MemorySystem:
    """Memory system to store and retrieve image-mask pairs based on feature similarity"""
    
    def __init__(self, memory_dir: str = "memory"):
        """
        메모리 시스템 초기화
        
        Args:
            memory_dir (str): 메모리 항목을 저장할 디렉토리
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True, parents=True)
        
        # 인덱스 파일
        self.index_path = self.memory_dir / "index.json"
        
        # 인덱스 로드 또는 초기화
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {
                "items": [],
                "next_id": 0
            }
            self._save_index()
    
    def _save_index(self):
        """인덱스를 디스크에 저장"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def add_memory(self, 
                  image: np.ndarray, 
                  mask: np.ndarray, 
                  features: np.ndarray,
                  metadata: Dict = None) -> int:
        """
        새 이미지-마스크 쌍을 메모리에 추가
        
        Args:
            image: 원본 이미지 (numpy 배열)
            mask: 세그멘테이션 마스크 (numpy 배열)
            features: DINOv2 특징 (numpy 배열)
            metadata: 선택적 메타데이터
            
        Returns:
            저장된 메모리 항목의 ID
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
        
        # 인덱스에 추가
        item_data = {
            "id": memory_id,
            "image_path": str(image_path.relative_to(self.memory_dir)),
            "mask_path": str(mask_path.relative_to(self.memory_dir)),
            "features_path": str(features_path.relative_to(self.memory_dir)),
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.index["items"].append(item_data)
        self._save_index()
        
        return memory_id
    
    def get_most_similar(self, features: np.ndarray, top_k: int = 1) -> List[Dict]:
        """
        특징 유사성에 기반하여 메모리에서 가장 유사한 항목 찾기
        
        Args:
            features: 쿼리 특징
            top_k: 반환할 유사 항목 수
            
        Returns:
            가장 유사한 메모리 항목 목록
        """
        if not self.index["items"]:
            return []
        
        similarities = []
        
        for item in self.index["items"]:
            # 항목 특징 로드
            item_features_path = self.memory_dir / item["features_path"]
            item_features = np.load(str(item_features_path))
            
            # 코사인 유사도 계산
            similarity = self._cosine_similarity(features, item_features)
            similarities.append((similarity, item))
        
        # 유사도로 정렬 (높은 것부터)
        similarities.sort(reverse=True, key=lambda x: x[0])
        
        # top_k 가장 유사한 항목 반환
        return [{"similarity": sim, "item": item} for sim, item in similarities[:top_k]]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """두 벡터 간의 코사인 유사도 계산"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get_item(self, item_id: int) -> Dict:
        """ID로 메모리 항목 검색"""
        for item in self.index["items"]:
            if item["id"] == item_id:
                return item
        raise ValueError(f"ID가 {item_id}인 항목을 찾을 수 없습니다")
    
    def load_item_data(self, item_id: int) -> Dict:
        """메모리 항목의 모든 데이터 로드"""
        item = self.get_item(item_id)
        
        # 이미지 로드
        image_path = self.memory_dir / item["image_path"]
        image = np.array(Image.open(str(image_path)))
        
        # 마스크 로드
        mask_path = self.memory_dir / item["mask_path"]
        mask = np.array(Image.open(str(mask_path)))
        
        # 특징 로드
        features_path = self.memory_dir / item["features_path"]
        features = np.load(str(features_path))
        
        return {
            "item": item,
            "image": image,
            "mask": mask,
            "features": features
        }
    
    def get_all_items(self) -> List[Dict]:
        """모든 메모리 항목 가져오기"""
        return self.index["items"]