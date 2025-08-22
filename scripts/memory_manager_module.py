import os
import numpy as np
from PIL import Image
import gradio as gr
from typing import List, Dict, Tuple, Any, Optional, Union
from scripts.memory_sam_predictor import MemorySAMPredictor
from scripts.memory_ui_utils import visualize_mask
from pathlib import Path

class MemoryManagerModule:
    """메모리 관리 기능을 처리하는 모듈"""
    
    def __init__(self, memory_sam_predictor: MemorySAMPredictor):
        """
        메모리 관리 모듈 초기화
        
        Args:
            memory_sam_predictor: 메모리 SAM 예측기 인스턴스
        """
        self.memory_sam = memory_sam_predictor
    
    def view_memory(self) -> Tuple[List[str], str]:
        """
        메모리 항목 보기
        
        Returns:
            (갤러리 아이템 목록, 메모리 정보) 튜플
        """
        try:
            memory_items = self.memory_sam.memory.index["items"]
            if not memory_items:
                return [], "메모리가 비어 있습니다."
                
            # 갤러리 아이템: 이미지 경로만 포함된 리스트
            gallery_items = []
            for item in memory_items:
                img_path = self.memory_sam.memory.memory_dir / item["image_path"]
                gallery_items.append(str(img_path))  # 문자열만 반환
            
            memory_info = f"메모리에 {len(memory_items)}개 항목이 있습니다.\n"
            for item in memory_items[:5]:
                memory_info += f"ID: {item['id']}, 타임스탬프: {item['timestamp']}\n"
            if len(memory_items) > 5:
                memory_info += f"... 그리고 {len(memory_items) - 5}개의 항목이 더 있습니다."
                
            return gallery_items, memory_info
        except Exception as e:
            import traceback
            traceback.print_exc()
            return [], f"메모리 보기 오류: {str(e)}"
    
    def load_memory_display(self) -> Tuple[List[str], str]:
        """
        메모리 디스플레이 로드
        
        Returns:
            (갤러리 아이템 목록, 메모리 통계) 튜플
        """
        try:
            memory_items = self.memory_sam.memory.index["items"]
            if not memory_items:
                return [], "메모리가 비어 있습니다."
                
            # 갤러리 아이템: 이미지 경로만 포함된 리스트 (튜플이 아님)
            gallery_items = []
            for item in memory_items:
                img_path = self.memory_sam.memory.memory_dir / item["image_path"]
                gallery_items.append(str(img_path))  # 문자열만 반환 (튜플 형식 삭제)
            
            memory_stats = f"총 항목 수: {len(memory_items)}\n"
            memory_stats += f"메모리 디렉토리: {self.memory_sam.memory.memory_dir}\n"
            memory_stats += f"다음 ID: {self.memory_sam.memory.index['next_id']}"
            
            return gallery_items, memory_stats
        except Exception as e:
            import traceback
            traceback.print_exc()
            return [], f"메모리 디스플레이 로드 오류: {str(e)}"
    
    def display_memory_item(self, idx) -> Tuple[Optional[np.ndarray], Dict]:
        """
        선택된 메모리 항목 표시
        
        Args:
            idx: 선택한 갤러리 항목의 인덱스
            
        Returns:
            (시각화된 이미지, 항목 정보) 튜플
        """
        try:
            # 인덱스가 정수인지 확인
            if isinstance(idx, int):
                index = idx
            else:
                # evt 객체의 경우
                if hasattr(idx, 'index'):
                    index = idx.index
                else:
                    # 다른 경우 기본값
                    index = 0
                    
            memory_items = self.memory_sam.memory.index["items"]
            if index < 0 or index >= len(memory_items):
                print(f"유효하지 않은 인덱스: {index}, 총 항목 수: {len(memory_items)}")
                return None, {"error": "유효하지 않은 항목 인덱스"}
                
            selected_item = memory_items[index]
            print(f"메모리 항목 로드 중 (ID: {selected_item['id']}, 인덱스: {index})")
            
            # 이미지 로드
            img_path = self.memory_sam.memory.memory_dir / selected_item["image_path"]
            print(f"이미지 로드 중: {img_path}")
            image = np.array(Image.open(str(img_path)))
            
            # 마스크 로드
            mask_path = self.memory_sam.memory.memory_dir / selected_item["mask_path"]
            print(f"마스크 로드 중: {mask_path}")
            mask = np.array(Image.open(str(mask_path)))
            
            if mask.ndim == 2:
                mask_viz = visualize_mask(image, mask > 0)
            else:
                mask_viz = image
                
            # 항목 정보 수집
            item_info = {
                "id": selected_item["id"],
                "timestamp": selected_item["timestamp"],
                "metadata": selected_item["metadata"]
            }
            
            # 패치 특징 정보 추가 (있는 경우)
            if "patch_features_path" in selected_item:
                item_info["has_patch_features"] = True
                
                # 패치 정보 경로 확인
                patch_info_path = self.memory_sam.memory.memory_dir / selected_item["patch_features_path"].replace("patch_features.npy", "patch_info.json")
                if os.path.exists(patch_info_path):
                    import json
                    with open(patch_info_path, 'r') as f:
                        patch_info = json.load(f)
                    item_info["grid_size"] = patch_info.get("grid_size")
                    item_info["resize_scale"] = patch_info.get("resize_scale")
            
            return mask_viz, item_info
        except Exception as e:
            import traceback
            print(f"메모리 항목 표시 오류: {str(e)}")
            traceback.print_exc()
            return None, {"error": str(e)}
    
    def delete_memory_item(self, item_id: int) -> str:
        """
        메모리 항목 삭제
        
        Args:
            item_id: 삭제할 항목 ID
            
        Returns:
            작업 결과 메시지
        """
        try:
            # 메모리 항목 찾기
            item_index = None
            for i, item in enumerate(self.memory_sam.memory.index["items"]):
                if item["id"] == item_id:
                    item_index = i
                    break
            
            if item_index is None:
                return f"항목 ID {item_id}를 찾을 수 없습니다."
            
            # 항목 정보 가져오기
            item = self.memory_sam.memory.index["items"][item_index]
            
            # 파일 삭제
            files_to_delete = []
            if "image_path" in item:
                files_to_delete.append(self.memory_sam.memory.memory_dir / item["image_path"])
            if "mask_path" in item:
                files_to_delete.append(self.memory_sam.memory.memory_dir / item["mask_path"])
            if "features_path" in item:
                files_to_delete.append(self.memory_sam.memory.memory_dir / item["features_path"])
            if "patch_features_path" in item:
                files_to_delete.append(self.memory_sam.memory.memory_dir / item["patch_features_path"])
                # 패치 정보 파일도 삭제
                patch_info_path = str(self.memory_sam.memory.memory_dir / item["patch_features_path"]).replace("patch_features.npy", "patch_info.json")
                files_to_delete.append(Path(patch_info_path))
            
            # 파일 삭제
            for file_path in files_to_delete:
                if file_path.exists():
                    file_path.unlink()
            
            # 인덱스에서 항목 제거
            self.memory_sam.memory.index["items"].pop(item_index)
            
            # 인덱스 저장
            self.memory_sam.memory._save_index()
            
            return f"항목 ID {item_id}가 삭제되었습니다."
        except Exception as e:
            return f"항목 삭제 중 오류: {str(e)}"
    
    def clear_memory(self) -> str:
        """
        메모리 초기화 (모든 항목 삭제)
        
        Returns:
            작업 결과 메시지
        """
        try:
            # 메모리 항목이 없으면 바로 반환
            if not self.memory_sam.memory.index["items"]:
                return "메모리가 이미 비어 있습니다."
            
            # 모든 항목 ID 수집
            item_ids = [item["id"] for item in self.memory_sam.memory.index["items"]]
            
            # 각 항목 삭제
            for item_id in item_ids:
                self.delete_memory_item(item_id)
            
            # 인덱스 초기화 (items 비우고 next_id 리셋)
            self.memory_sam.memory.index["items"] = []
            if "next_id" in self.memory_sam.memory.index:
                self.memory_sam.memory.index["next_id"] = 0
            self.memory_sam.memory._save_index()
            
            # 비어있는 item_* 폴더 및 잔여 파일 정리
            import shutil
            memory_dir = self.memory_sam.memory.memory_dir
            try:
                for entry in memory_dir.iterdir():
                    # item_* 폴더 정리
                    if entry.is_dir() and entry.name.startswith("item_"):
                        shutil.rmtree(entry, ignore_errors=True)
                    # 인덱스/FAISS 외의 루트 잔여 파일 정리
                    elif entry.is_file() and entry.name not in {"index.json", "faiss_index.bin"}:
                        try:
                            entry.unlink()
                        except Exception:
                            pass
            except Exception:
                # 디렉토리 정리는 베스트 에포트
                pass

            # FAISS 인덱스 초기화/삭제
            faiss_index_path = self.memory_sam.memory.faiss_index_path
            try:
                if hasattr(self.memory_sam.memory, 'faiss_index') and self.memory_sam.memory.faiss_index is not None and self.memory_sam.memory.feature_dim:
                    # 특징 차원 유지하여 빈 인덱스로 재생성
                    import faiss
                    self.memory_sam.memory.faiss_index = faiss.IndexFlatL2(int(self.memory_sam.memory.feature_dim))
                    self.memory_sam.memory.id_to_index_map = {}
                    # 덮어쓰기 저장
                    faiss.write_index(self.memory_sam.memory.faiss_index, str(faiss_index_path))
                else:
                    # 차원 정보가 없거나 인덱스가 없으면 파일 자체 삭제
                    if faiss_index_path.exists():
                        faiss_index_path.unlink()
                    self.memory_sam.memory.faiss_index = None
                    self.memory_sam.memory.id_to_index_map = {}
            except Exception:
                # 베스트 에포트로 무시
                pass
            
            return f"메모리가 초기화되었습니다. {len(item_ids)}개 항목이 삭제되었습니다."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"메모리 초기화 중 오류: {str(e)}"
    
    def visualize_memory_matches(self, item_id: int, image: np.ndarray) -> Optional[np.ndarray]:
        """
        메모리 항목과 주어진 이미지 간의 특징 매칭 시각화
        
        Args:
            item_id: 메모리 항목 ID
            image: 비교할 이미지
            
        Returns:
            시각화된 매칭 이미지
        """
        try:
            # 메모리 항목 로드
            item_data = self.memory_sam.memory.load_item_data(item_id)
            
            # 스파스 매칭이 활성화된 경우 시각화
            if self.memory_sam.use_sparse_matching:
                # 현재 이미지의 패치 특징 추출
                patch_features, grid_size, resize_scale = self.memory_sam.extract_patch_features(image)
                
                # 메모리 항목에 패치 특징이 있는 경우
                if "patch_features" in item_data and "grid_size" in item_data and "resize_scale" in item_data:
                    # 매칭 시각화
                    match_vis = self.memory_sam.visualize_sparse_matches(
                        item_data["image"], 
                        image,
                        item_data["mask"],
                        self.memory_sam.current_mask if hasattr(self.memory_sam, "current_mask") else None
                    )
                    return match_vis
            
            return None
        except Exception as e:
            print(f"Match visualization error: {str(e)}")
            return None