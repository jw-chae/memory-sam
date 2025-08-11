import os
import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm
import tempfile
import shutil

# UI 모듈 임포트
from ui.image_utils import ImageResizer
from ui.file_utils import FileManager
from ui.progress_tracker import ProgressTracker

class EnhancedSegmentationModule:
    """프로그레스 바와 리사이징 기능이 추가된 향상된 세그멘테이션 모듈"""
    
    def __init__(self, memory_sam_predictor):
        """
        초기화
        
        Args:
            memory_sam_predictor: Memory SAM 예측기 인스턴스
        """
        self.memory_sam = memory_sam_predictor
        self.processed_images = []
        self.current_folder_images = []
        self.temp_dir = None
    
    def process_image_with_progress(
        self, 
        input_files, 
        reference_path=None, 
        use_reference=False, 
        prompt_type="points",
        use_sparse_matching=True, 
        match_background=True,
        skip_clustering=False,
        auto_add_to_memory=False,
        progress_callback=None
    ):
        """
        이미지 또는 폴더를 처리하고 진행 상황을 표시
        
        Args:
            input_files: 입력 파일(들) 또는 폴더
            reference_path: 참조 이미지 경로
            use_reference: 참조 이미지 사용 여부
            prompt_type: 프롬프트 유형
            use_sparse_matching: 스파스 매칭 사용 여부
            match_background: 배경 영역 매칭 여부
            skip_clustering: 클러스터링 건너뛰기 여부
            auto_add_to_memory: 자동 메모리 추가 여부
            progress_callback: 진행 상황 콜백 함수
            
        Returns:
            처리 결과
        """
        # 진행 상황 관리 초기화
        start_time = time.time()
        
        # 이전 처리 결과 초기화
        self.processed_images = []
        
        # 폴더 처리 모드인지 확인
        is_folder_mode = isinstance(input_files, str) and os.path.isdir(input_files)
        is_multiple_files = isinstance(input_files, list) and len(input_files) > 0
        
        # 단일 파일, 다중 파일 또는 폴더 처리
        if is_folder_mode:
            folder_path = input_files
            # 폴더에서 이미지 파일 수집
            image_files = FileManager.collect_image_files(folder_path)
            
            if not image_files:
                if progress_callback:
                    progress_callback(1.0, desc="완료")
                return None, None, [], "폴더에 이미지 파일이 없습니다.", None, None, None, None, "", None, None, None
            
            # 진행 상황 초기화
            if progress_callback:
                progress_callback(0, desc=f"폴더 내 {len(image_files)}개 이미지 처리 중...")
            
            # 임시 디렉토리 생성
            temp_dir_obj, temp_dir = FileManager.create_temp_directory()
            self.temp_dir = temp_dir_obj
            
            # 이미지 파일 임시 디렉토리에 복사
            temp_files = FileManager.copy_files_to_temp(image_files, temp_dir)
            
            # 결과 초기화
            result_img, mask_img, memory_matches = None, None, []
            memory_info = f"폴더 처리 모드: {len(image_files)}개 파일을 임시 디렉토리 {temp_dir}에 복사했습니다."
            
            # 각 이미지 파일 처리
            for i, file_path in enumerate(temp_files):
                if progress_callback:
                    progress_callback((i / len(temp_files)), desc=f"이미지 처리 중 ({i+1}/{len(temp_files)}): {os.path.basename(file_path)}")
                
                try:
                    # 이미지 처리
                    img_result = self._process_single_image(
                        file_path, 
                        reference_path, 
                        use_reference, 
                        prompt_type,
                        use_sparse_matching, 
                        match_background,
                        skip_clustering,
                        auto_add_to_memory
                    )
                    
                    # 첫 번째 결과 저장 (UI에 표시할 용도)
                    if i == 0 and img_result and len(img_result) >= 2:
                        result_img, mask_img = img_result[0], img_result[1]
                        if len(img_result) >= 3:
                            memory_matches = img_result[2]
                except Exception as e:
                    print(f"파일 처리 중 오류: {file_path} - {e}")
            
            # 처리 완료
            if progress_callback:
                progress_callback(1.0, desc="폴더 처리 완료")
            
            # 폴더 처리 결과 정보
            info_text = f"폴더 처리 결과: {len(self.processed_images)}개 이미지 처리됨"
            
            # 첫 번째 항목 선택
            selected_original, selected_mask, selected_overlay, selected_info = None, None, None, ""
            
            if self.processed_images:
                first_item = self.processed_images[0]
                selected_info = f"폴더 처리 결과의 첫 아이템 선택됨: {first_item.get('name', '')}"
                selected_original = first_item.get("original")
                selected_mask = first_item.get("mask")
                selected_overlay = first_item.get("overlay")
            
            # 스파스 매칭 결과
            sparse_matches, img1_points, img2_points = None, None, None
            
            return (
                result_img, mask_img, memory_matches, memory_info,
                self._create_gallery_images(), selected_original, selected_mask, selected_overlay, selected_info,
                sparse_matches, img1_points, img2_points
            )
            
        elif is_multiple_files:
            # 다중 파일 처리
            if progress_callback:
                progress_callback(0, desc=f"{len(input_files)}개 파일 처리 중...")
            
            # 결과 초기화
            result_img, mask_img, memory_matches = None, None, []
            memory_info = f"다중 파일 처리 모드: {len(input_files)}개 파일"
            
            # 각 파일 처리
            for i, file_obj in enumerate(input_files):
                if progress_callback:
                    progress_callback((i / len(input_files)), desc=f"파일 처리 중 ({i+1}/{len(input_files)})")
                
                try:
                    file_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
                    
                    # 이미지 처리
                    img_result = self._process_single_image(
                        file_path, 
                        reference_path, 
                        use_reference, 
                        prompt_type,
                        use_sparse_matching, 
                        match_background,
                        skip_clustering,
                        auto_add_to_memory
                    )
                    
                    # 첫 번째 결과 저장 (UI에 표시할 용도)
                    if i == 0 and img_result and len(img_result) >= 2:
                        result_img, mask_img = img_result[0], img_result[1]
                        if len(img_result) >= 3:
                            memory_matches = img_result[2]
                except Exception as e:
                    print(f"파일 처리 중 오류: {file_path} - {e}")
            
            # 처리 완료
            if progress_callback:
                progress_callback(1.0, desc="파일 처리 완료")
            
            # 다중 파일 처리 결과 정보
            info_text = f"다중 파일 처리 결과: {len(self.processed_images)}개 이미지 처리됨"
            
            # 첫 번째 항목 선택
            selected_original, selected_mask, selected_overlay, selected_info = None, None, None, ""
            
            if self.processed_images:
                first_item = self.processed_images[0]
                selected_info = f"다중 파일 처리 결과의 첫 아이템 선택됨: {first_item.get('name', '')}"
                selected_original = first_item.get("original")
                selected_mask = first_item.get("mask")
                selected_overlay = first_item.get("overlay")
            
            # 스파스 매칭 결과
            sparse_matches, img1_points, img2_points = None, None, None
            
            return (
                result_img, mask_img, memory_matches, memory_info,
                self._create_gallery_images(), selected_original, selected_mask, selected_overlay, selected_info,
                sparse_matches, img1_points, img2_points
            )
            
        else:
            # 단일 파일 처리
            if not input_files or (isinstance(input_files, list) and not input_files):
                if progress_callback:
                    progress_callback(1.0, desc="완료")
                return None, None, [], "입력 파일이 지정되지 않았습니다.", None, None, None, None, "", None, None, None
            
            if progress_callback:
                progress_callback(0, desc="이미지 처리 중...")
            
            try:
                # 단일 파일 경로 가져오기
                file_path = input_files[0].name if isinstance(input_files, list) else input_files
                
                # 이미지 처리
                result = self._process_single_image(
                    file_path, 
                    reference_path, 
                    use_reference, 
                    prompt_type,
                    use_sparse_matching, 
                    match_background,
                    skip_clustering,
                    auto_add_to_memory
                )
                
                if not result:
                    if progress_callback:
                        progress_callback(1.0, desc="처리 실패")
                    return None, None, [], "이미지 처리 실패", None, None, None, None, "", None, None, None
                
                result_img, mask_img, memory_matches, memory_info, sparse_matches, img1_points, img2_points = result
                
                # 처리 완료
                if progress_callback:
                    progress_callback(1.0, desc="처리 완료")
                
                # 갤러리 이미지 및 선택된 이미지 설정
                gallery_images = self._create_gallery_images()
                
                selected_original, selected_mask, selected_overlay, selected_info = None, None, None, ""
                
                if self.processed_images:
                    first_item = self.processed_images[0]
                    selected_info = f"단일 파일 처리 결과: {first_item.get('name', '')}"
                    selected_original = first_item.get("original")
                    selected_mask = first_item.get("mask")
                    selected_overlay = first_item.get("overlay")
                
                return (
                    result_img, mask_img, memory_matches, memory_info,
                    gallery_images, selected_original, selected_mask, selected_overlay, selected_info,
                    sparse_matches, img1_points, img2_points
                )
                
            except Exception as e:
                if progress_callback:
                    progress_callback(1.0, desc="오류 발생")
                print(f"이미지 처리 중 오류: {e}")
                return None, None, [], f"오류: {str(e)}", None, None, None, None, "", None, None, None
    
    def _process_single_image(
        self, 
        file_path, 
        reference_path=None, 
        use_reference=False, 
        prompt_type="points",
        use_sparse_matching=True, 
        match_background=True,
        skip_clustering=False,
        auto_add_to_memory=False
    ):
        """
        단일 이미지 처리
        
        Args:
            file_path: 이미지 파일 경로
            reference_path: 참조 이미지 경로
            use_reference: 참조 이미지 사용 여부
            prompt_type: 프롬프트 유형
            use_sparse_matching: 스파스 매칭 사용 여부
            match_background: 배경 영역 매칭 여부
            skip_clustering: 클러스터링 건너뛰기 여부
            auto_add_to_memory: 자동 메모리 추가 여부
            
        Returns:
            처리 결과
        """
        try:
            # 파일 존재 확인
            if not os.path.exists(file_path):
                print(f"파일을 찾을 수 없음: {file_path}")
                return None
            
            # 처리 시작 시간
            start_time = time.time()
            
            # 이미지 로드
            img = cv2.imread(file_path)
            if img is None:
                print(f"이미지 로드 실패: {file_path}")
                return None
            
            # BGR에서 RGB로 변환
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 원본 이미지 크기 저장
            original_h, original_w = img_rgb.shape[:2]
            
            # 리사이징 적용
            if hasattr(self.memory_sam, 'resize_images') and self.memory_sam.resize_images and \
               hasattr(self.memory_sam, 'resize_scale') and self.memory_sam.resize_scale != 1.0:
                img_rgb = ImageResizer.resize_image(img_rgb, self.memory_sam.resize_scale)
                print(f"이미지 리사이징 적용: {self.memory_sam.resize_scale*100:.0f}%, " 
                      f"{original_w}x{original_h} -> {img_rgb.shape[1]}x{img_rgb.shape[0]}")
            
            # 참조 이미지 처리
            reference_img = None
            reference_mask = None
            
            if use_reference and reference_path:
                # 참조 이미지 로드
                ref_img = cv2.imread(reference_path)
                if ref_img is not None:
                    reference_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
                    
                    # 마스크 파일 경로 (참조 이미지와 동일한 이름, '_mask' 접미사)
                    mask_path = os.path.splitext(reference_path)[0] + "_mask.png"
                    if os.path.exists(mask_path):
                        reference_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Memory SAM으로 이미지 처리
            # 프롬프트 타입에 따라 처리 방식 결정
            file_name = os.path.basename(file_path)
            
            # 스파스 매칭 설정
            self.memory_sam.use_sparse_matching = use_sparse_matching
            
            # 이미지 세그멘테이션 수행
            result_img, mask = self.memory_sam.process_image(
                img_rgb, 
                prompt_type=prompt_type,
                reference_image=reference_img,
                reference_mask=reference_mask,
                match_background=match_background,
                skip_clustering=skip_clustering
            )
            
            # 스파스 매칭 결과 (있는 경우)
            sparse_matches = None
            img1_points = None
            img2_points = None
            
            if hasattr(self.memory_sam, 'sparse_match_visualization') and self.memory_sam.sparse_match_visualization is not None:
                sparse_matches = self.memory_sam.sparse_match_visualization
                
                # 특징점 시각화 (있는 경우)
                if hasattr(self.memory_sam, 'img1_points_vis') and self.memory_sam.img1_points_vis is not None:
                    img1_points = self.memory_sam.img1_points_vis
                    
                if hasattr(self.memory_sam, 'img2_points_vis') and self.memory_sam.img2_points_vis is not None:
                    img2_points = self.memory_sam.img2_points_vis
            
            # 메모리 매칭 결과
            memory_items = []
            if hasattr(self.memory_sam, 'memory_matches') and self.memory_sam.memory_matches:
                for item in self.memory_sam.memory_matches:
                    if 'image_path' in item and os.path.exists(item['image_path']):
                        # 이미지 로드 및 변환
                        img = cv2.imread(item['image_path'])
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        memory_items.append((item['image_path'], img_rgb))
            
            # 처리 시간 계산
            processing_time = time.time() - start_time
            
            # 처리된 이미지 정보 저장
            item_info = {
                "path": file_path,
                "name": file_name,
                "original": img_rgb,
                "mask": mask,
                "overlay": result_img,
                "width": img_rgb.shape[1],
                "height": img_rgb.shape[0],
                "original_width": original_w,
                "original_height": original_h,
                "resize_scale": getattr(self.memory_sam, 'resize_scale', 1.0),
                "processing_time": processing_time
            }
            
            # 처리된 이미지 목록에 추가
            self.processed_images.append(item_info)
            
            # 메모리에 자동 추가 (필요한 경우)
            if auto_add_to_memory:
                self.memory_sam.add_to_memory(img_rgb, mask)
            
            # 출력 정보 생성
            memory_info = f"처리 완료: {file_name}, 크기: {img_rgb.shape[1]}x{img_rgb.shape[0]}, " \
                        f"처리 시간: {processing_time:.2f}초"
            
            return result_img, mask, memory_items, memory_info, sparse_matches, img1_points, img2_points
            
        except Exception as e:
            print(f"이미지 처리 중 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_gallery_images(self):
        """
        처리된 이미지로 갤러리 생성
        
        Returns:
            갤러리 이미지 리스트
        """
        gallery_images = []
        
        for item in self.processed_images:
            if "overlay" in item and item["overlay"] is not None:
                # 갤러리 항목 생성
                gallery_images.append(item["overlay"])
        
        return gallery_images
    
    def handle_result_gallery_select(self, evt, processed_images):
        """
        결과 갤러리 선택 이벤트 처리
        
        Args:
            evt: 선택 이벤트
            processed_images: 처리된 이미지 목록
            
        Returns:
            선택된 항목 정보
        """
        if not processed_images or evt is None:
            return None, None, None, "선택된 이미지가 없습니다."
        
        try:
            # 선택된 이미지 인덱스
            index = evt.index if hasattr(evt, 'index') else 0
            
            if index < 0 or index >= len(processed_images):
                return None, None, None, "유효하지 않은 선택입니다."
            
            # 선택된 이미지 항목
            item = processed_images[index]
            
            # 결과 이미지 가져오기
            original_img = item.get("original")
            mask_img = item.get("mask")
            overlay_img = item.get("overlay")
            
            # 정보 텍스트 생성
            info_text = f"파일: {item.get('path', '알 수 없음')}\n"
            info_text += f"크기: {item.get('width', 0)}x{item.get('height', 0)}\n"
            
            if "original_width" in item and "original_height" in item:
                info_text += f"원본 크기: {item.get('original_width', 0)}x{item.get('original_height', 0)}\n"
            
            if "resize_scale" in item:
                info_text += f"리사이징: {item.get('resize_scale', 1.0):.2f}\n"
            
            if "processing_time" in item:
                info_text += f"처리 시간: {item.get('processing_time', 0):.2f}초\n"
            
            return original_img, mask_img, overlay_img, info_text
            
        except Exception as e:
            print(f"결과 갤러리 선택 처리 중 오류: {e}")
            return None, None, None, f"오류 발생: {str(e)}"