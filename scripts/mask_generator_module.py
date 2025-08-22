import os
import numpy as np
import cv2
from PIL import Image
import gradio as gr
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from scripts.memory_sam_predictor import MemorySAMPredictor
from scripts.memory_ui_utils import draw_points_on_image, visualize_mask

class MaskGeneratorModule:
    """마스크 생성 기능을 처리하는 모듈"""
    
    def __init__(self, memory_sam_predictor: MemorySAMPredictor):
        """
        마스크 생성 모듈 초기화
        
        Args:
            memory_sam_predictor: 메모리 SAM 예측기 인스턴스
        """
        self.memory_sam = memory_sam_predictor
        
        # 상태 변수
        self.current_points = []         # 선택한 포인트 목록
        self.current_point_labels = []   # 포인트 레이블 목록 (1: 전경, 0: 배경)
        self.current_point_type = "전경 (객체)"  # 기본 포인트 타입
        self.current_mask = None         # 생성된 마스크
        self.current_mask_vis = None     # 시각화용 마스크
        self.reference_image = None
        self.reference_image_path = None
    
    def toggle_controls(self, prompt_type: str) -> Tuple[Any, Any, Any]:
        """
        프롬프트 타입에 따라 컨트롤 표시 전환 (포인트 전용으로 수정)
        
        Args:
            prompt_type: 프롬프트 타입 (항상 'points'가 될 것으로 예상)
            
        Returns:
            컨트롤 가시성 업데이트 튜플
        """
        # 박스 관련 UI가 없으므로 항상 포인트 컨트롤만 보이도록 설정
        # 실제로는 이 함수가 호출될 때 prompt_type이 항상 'points'만 오도록 상위에서 제어해야 함
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    
    def handle_point_selection_state(self, img, points, labels, evt: gr.SelectData):
        """
        이미지 클릭 이벤트 처리 (포인트 추가)
        
        Args:
            img: 입력 이미지 (512x512로 리사이징된 이미지)
            points: 현재 포인트 목록 (512x512 기준으로 정규화됨)
            labels: 현재 레이블 목록
            evt: 클릭 이벤트 데이터 (None이면 전달된 points/labels 사용)
            
        Returns:
            (업데이트된 이미지, 포인트 목록, 레이블 목록) 튜플
        """
        print(f"Received img: {type(img)}, points: {points}, labels: {labels}, evt: {evt}")
        if img is None:
            return img, points, labels
        
        # img는 이미 numpy 배열로 전달됨
        img_np = img
        h, w = img_np.shape[:2]  # 항상 512x512
        
        # 이벤트가 있는 경우, 클릭 좌표 추출 및 포인트/레이블 추가
        if evt is not None:
            x, y = evt.index
            print(f"클릭 좌표: ({x}, {y})")
                
            # 현재 포인트 타입에 따라
            label = 1 if self.current_point_type == "전경 (객체)" else 0
            points.append([x, y])
            labels.append(label)
            print(f"추가된 포인트: {points[-1]}, 레이블: {label}")
        else:
            # 이벤트가 없는 경우 전달된 points/labels 사용 (이미 512x512 기준으로 정규화됨)
            if points and len(points) > 0:
                print(f"기존 포인트 사용 (마지막 포인트: {points[-1]})")
            else:
                print("포인트가 없습니다.")
                return img_np, points, labels

        # SAM 예측기로 마스크 생성
        mask = None
        if len(points) > 0:
            try:
                points_array = np.array(points)
                labels_array = np.array(labels)
                predictor = self.memory_sam.predictor
                predictor.set_image(img_np)
                masks, scores, _ = predictor.predict(
                    point_coords=points_array,
                    point_labels=labels_array,
                    multimask_output=True
                )
                if len(scores) > 0:
                    best_idx = np.argmax(scores)
                    mask = masks[best_idx]
                    self.current_mask = mask
                    mask_vis = (mask * 255).astype(np.uint8)
                    self.current_mask_vis = mask_vis
                    print(f"마스크 생성 완료, 최고 점수: {scores[best_idx]:.4f}")
                else:
                    print("마스크 생성 실패: 점수가 없습니다")
            except Exception as e:
                print(f"마스크 생성 중 오류: {e}")
                import traceback
                traceback.print_exc()
                
        # 내부 상태 업데이트
        self.current_points = points
        self.current_point_labels = labels
                
        result_img = draw_points_on_image(img_np, points, labels, mask)
        return result_img, points, labels
    
    def set_point_type(self, point_type: str) -> str:
        """
        포인트 타입 설정 (전경 또는 배경)
        
        Args:
            point_type: 설정할 포인트 타입
            
        Returns:
            상태 메시지
        """
        self.current_point_type = point_type
        return f"포인트 타입: {point_type}"
    
    def clear_all_points(self) -> str:
        """
        모든 포인트 초기화
        
        Returns:
            상태 메시지
        """
        try:
            print("포인트 및 마스크 초기화 중...")
            # 확실하게 모든 상태 변수 초기화
            self.current_points = []
            self.current_point_labels = []
            self.current_mask = None
            self.current_mask_vis = None
            
            # 현재 프로세서 상태도 초기화
            if hasattr(self.memory_sam, 'predictor'):
                if hasattr(self.memory_sam.predictor, 'reset_image'):
                    self.memory_sam.predictor.reset_image()
                    print("예측기 이미지 초기화 완료")
            
            return "포인트 초기화 완료. 새 포인트를 클릭하여 마스크를 생성하세요."
        except Exception as e:
            import traceback
            print(f"포인트 초기화 중 오류: {e}")
            traceback.print_exc()
            return f"포인트 초기화 중 오류: {e}"
    
    def update_image_dimensions(self, img):
        """
        이미지 크기 정보 업데이트
        
        Args:
            img: 입력 이미지
            
        Returns:
            이미지 크기 정보 문자열
        """
        if img is None:
            return "이미지를 로드하세요"
        try:
            h, w = np.array(img).shape[:2]
            return f"이미지 크기: {w}x{h}px"
        except Exception as e:
            print(f"이미지 크기 업데이트 중 오류: {e}")
            return "이미지 크기를 확인할 수 없습니다"
    
    def save_generated_mask(self, img, mask):
        """
        생성된 마스크 저장
        
        Args:
            img: 입력 이미지
            mask: 생성된 마스크
            
        Returns:
            저장 결과 메시지
        """
        if img is None:
            return "오류: 저장할 이미지가 없습니다."
            
        if mask is None and self.current_mask is None:
            return "오류: 저장할 마스크가 없습니다."
            
        try:
            img_np = np.array(img)
            
            # 512x512로 리사이징
            from ui.image_utils import ImageResizer
            resized_img = ImageResizer.resize_image(img_np, "512x512")
            
            # 마스크 사용 (전달된 마스크 또는 현재 마스크)
            mask_to_use = mask if mask is not None else self.current_mask
            
            import time
            timestamp = int(time.time())
            image_path = os.path.join(self.memory_sam.results_dir, f"image_{timestamp}.png")
            mask_path = os.path.join(self.memory_sam.results_dir, f"mask_{timestamp}.png")
            
            Image.fromarray(resized_img).save(image_path)
            mask_vis = (mask_to_use * 255).astype(np.uint8)
            Image.fromarray(mask_vis).save(mask_path)
            
            return f"마스크 저장 완료: {mask_path}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"마스크 저장 중 오류: {str(e)}"
    
    def save_mask_to_reference(self, image, mask):
        """
        생성된 마스크를 참조 이미지로 저장
        
        Args:
            image: 입력 이미지
            mask: 생성된 마스크
            
        Returns:
            저장 결과 메시지
        """
        if image is None:
            return "오류: 저장할 이미지가 없습니다."
            
        if mask is None and self.current_mask is None:
            return "오류: 저장할 마스크가 없습니다."
            
        try:
            img_np = np.array(image)
            
            # 512x512로 리사이징
            from ui.image_utils import ImageResizer
            resized_img = ImageResizer.resize_image(img_np, "512x512")
            
            # 마스크 사용 (전달된 마스크 또는 현재 마스크)
            mask_to_use = mask if mask is not None else self.current_mask
            
            import time
            timestamp = int(time.time())
            image_path = os.path.join(self.memory_sam.results_dir, f"ref_image_{timestamp}.png")
            mask_path = os.path.join(self.memory_sam.results_dir, f"ref_mask_{timestamp}.png")
            
            Image.fromarray(resized_img).save(image_path)
            mask_vis = (mask_to_use * 255).astype(np.uint8)
            Image.fromarray(mask_vis).save(mask_path)
            
            self.reference_image = resized_img
            self.reference_image_path = image_path
            
            return f"참조 이미지 및 마스크 저장 완료: {image_path}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"참조 이미지 저장 중 오류: {str(e)}"
    
    def save_to_memory_directly(self, image, mask):
        """
        생성된 마스크를 메모리에 직접 저장
        
        Args:
            image: 입력 이미지
            mask: 생성된 마스크
            
        Returns:
            저장 결과 메시지
        """
        if image is None:
            return "오류: 저장할 이미지가 없습니다."
        
        if mask is None and self.current_mask is None:
            return "오류: 저장할 마스크가 없습니다."
            
        try:
            # 원본 이미지 정보 저장
            img_np = np.array(image)
            original_size = img_np.shape[:2]  # (높이, 너비)
            
            # 원본 해상도 그대로 저장/특징 추출
            resized_img = img_np
            print(f"저장용 이미지 리사이징 제거: {original_size[1]}x{original_size[0]} 유지")
            
            # 마스크 사용 (전달된 마스크 또는 현재 마스크)
            mask_to_use = mask if mask is not None else self.current_mask
            
            if mask_to_use is None:
                return "오류: 저장할 마스크가 없습니다."
            
            print(f"저장할 마스크 크기: {mask_to_use.shape}")
            
            # 글로벌 특징 추출
            features = self.memory_sam.extract_features(resized_img)
            
            # 패치 특징 추출 (스파스 매칭이 활성화된 경우)
            patch_features = None
            grid_size = None
            resize_scale = None
            
            if self.memory_sam.use_sparse_matching:
                try:
                    patch_features, grid_size, resize_scale = self.memory_sam.extract_patch_features(resized_img)
                    print(f"패치 특징 추출 완료: 형태 {patch_features.shape}, 그리드 크기 {grid_size}")
                except Exception as e:
                    print(f"패치 특징 추출 실패: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 메모리에 추가
            memory_id = self.memory_sam.memory.add_memory(
                image=resized_img,
                mask=mask_to_use,
                features=features,
                patch_features=patch_features,
                grid_size=grid_size,
                resize_scale=resize_scale,
                metadata={
                    "timestamp": str(datetime.now()),
                    "source": "direct_creation",
                    "original_width": original_size[1],
                    "original_height": original_size[0],
                    "points": self.current_points,
                    "point_labels": self.current_point_labels
                }
            )
            print(f"메모리에 저장 완료: ID {memory_id}, 이미지 크기: {resized_img.shape[1]}x{resized_img.shape[0]}, 마스크 크기: {mask_to_use.shape}, 원본 크기: {original_size[1]}x{original_size[0]}")
            return f"메모리에 ID {memory_id}로 저장되었습니다."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"메모리에 저장 중 오류: {str(e)}"