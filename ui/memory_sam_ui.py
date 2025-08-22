import os
import sys
import numpy as np
import torch
import gradio as gr
import tempfile
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
from pathlib import Path
from datetime import datetime
import time
from tqdm import tqdm
import cv2

# 현재 디렉토리 경로를 Python 경로에 추가
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if script_dir not in sys.path:
    sys.path.append(script_dir)

# UI 모듈 임포트
from ui.components import UIComponents
from ui.image_utils import ImageResizer
from ui.file_utils import FileManager
from ui.progress_tracker import ProgressTracker

# Memory SAM 모듈 임포트
from scripts.memory_sam_predictor import MemorySAMPredictor
from scripts.segmentation_module import SegmentationModule
from scripts.mask_generator_module import MaskGeneratorModule
from scripts.memory_manager_module import MemoryManagerModule
from scripts.memory_ui_utils import SparseMatchVisualizer

class MemorySAMUI:
    """UI Class for Enhanced Memory SAM System"""
    
    def __init__(self, 
                model_type: str = "hiera_l", 
                checkpoint_path: str = None,
                dinov3_model: str = "dinov3_vitb16",
                memory_dir: str = "memory", 
                results_dir: str = "results",
                device: str = "cuda",
                use_sparse_matching: bool = True):
        """
        Initialize Memory SAM UI
        
        Args:
            model_type: SAM2 model type to use
            checkpoint_path: Path to checkpoint
            dinov3_model: DINOv3 model name
            memory_dir: Memory directory
            results_dir: Results directory
            device: Device to use
            use_sparse_matching: Whether to use sparse matching
        """
        # Initialize Memory SAM predictor
        self.memory_sam = MemorySAMPredictor(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            dinov3_model=dinov3_model,
            memory_dir=memory_dir,
            results_dir=results_dir,
            device=device,
            use_sparse_matching=use_sparse_matching
        )
        
        # Initialize modules
        self.segmentation_module = SegmentationModule(self.memory_sam)
        self.mask_generator_module = MaskGeneratorModule(self.memory_sam)
        self.memory_manager_module = MemoryManagerModule(self.memory_sam)
        
        # Initialize sparse match visualizer
        self.sparse_match_visualizer = SparseMatchVisualizer(self.memory_sam)
        
        # Set up results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Resizing setting
        self.current_resize_scale = 1.0
        
        # Clustering hyperparameters
        self.similarity_threshold = 0.8
        self.background_weight = 0.3
        self.skip_clustering = False
        self.hybrid_clustering = False
        
        # State variables
        self.processed_images = []
        self.current_folder_images = []
    
    def __del__(self):
        """Destructor: clean up temporary directory"""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
    
    def setup_interface(self):
        """Set up enhanced Gradio interface"""
        with gr.Blocks(title="Memory SAM - Image Segmentation") as interface:
            gr.Markdown("# Memory SAM - Image Segmentation")
            gr.Markdown("Intelligent image segmentation using SAM2 and DINOv2 with a memory system")
            
            with gr.Tabs():
                self._setup_enhanced_segmentation_tab()
                self._setup_mask_generator_tab()
                self._setup_memory_manager_tab()
        
        return interface
    
    def _setup_enhanced_segmentation_tab(self):
        """향상된 세그멘테이션 탭 설정"""
        with gr.TabItem("메모리 기반 세그멘테이션"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 이미지 또는 폴더 선택")
                    
                    # 리사이징 옵션
                    resize_ratio = UIComponents.create_resize_buttons()
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            # 파일 또는 폴더 선택
                            memory_image_input = gr.File(
                                label="입력 이미지 (파일 또는 폴더)",
                                file_count="multiple",
                                file_types=["image"],
                                elem_id="memory_image_input"
                            )
                        with gr.Column(scale=1):
                            # 폴더 선택 버튼과 경로 입력
                            folder_btn, folder_path_input = UIComponents.create_folder_browser(
                                callback=FileManager.browse_directory
                            )
                    
                    prompt_type = gr.Radio(
                        choices=["points", "box"],
                        value="points",
                        label="프롬프트 타입 (모든 이미지에 적용)"
                    )
                    
                    # 스파스 매칭 옵션 추가
                    with gr.Column():
                        use_sparse_matching = gr.Checkbox(
                            label="스파스 매칭 사용", 
                            value=self.memory_sam.use_sparse_matching,
                            info="DINOv2 패치 특징 기반 스파스 매칭을 사용합니다."
                        )
                        match_background = gr.Checkbox(
                            label="배경 영역 매칭", 
                            value=True,
                            info="마스크가 아닌 배경 영역도 매칭합니다. 이는 negative 정보로 사용됩니다."
                        )
                    
                    # 클러스터링 하이퍼파라미터 컨트롤
                    with gr.Accordion("클러스터링 하이퍼파라미터", open=False):
                        similarity_threshold, background_weight, skip_clustering, hybrid_clustering, max_positive_points, max_negative_points, use_positive_kmeans, positive_kmeans_clusters = \
                            UIComponents.create_clustering_controls()
                    
                    with gr.Accordion("참조 이미지 (선택 사항)", open=False):
                        reference_image = gr.Image(label="참조 이미지", type="filepath")
                        use_reference = gr.Checkbox(label="참조 이미지 사용", value=False)
                    
                    # 진행 상황 표시
                    progress = UIComponents.create_progress_bar()
                    
                    process_btn = gr.Button("이미지/폴더 처리", variant="primary", elem_id="process_btn")
                    
                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.TabItem("세그멘테이션 결과"):
                            result_image = gr.Image(label="세그멘테이션 결과")
                        with gr.TabItem("마스크"):
                            mask_image = gr.Image(label="세그멘테이션 마스크")
                        with gr.TabItem("메모리 결과"):
                            memory_gallery = gr.Gallery(
                                label="메모리의 유사 이미지", 
                                columns=3,
                                rows=1,
                                height=300
                            )
                    
                    memory_info = gr.Textbox(
                        value="",
                        label="메모리 정보",
                        interactive=False
                    )
                    
                    with gr.Row():
                        save_btn = gr.Button("메모리에 저장")
                        view_memory_btn = gr.Button("메모리 보기")
            
            # 결과 보기 섹션
            with gr.Accordion("🖼️ 처리 결과 보기", open=True) as results_accordion:
                gr.Markdown("### 처리된 이미지 갤러리")
                gr.Markdown("아래 갤러리에서 이미지를 클릭하여 상세 결과를 확인하세요.")
                
                with gr.Row():
                    result_gallery = UIComponents.create_gallery(
                        label="처리된 이미지 결과", 
                        columns=3
                    )
                
                # 결과 탭
                tabs, selected_original, selected_mask, selected_overlay = \
                    UIComponents.create_result_display()
                
                result_info = gr.Textbox(
                    value="",
                    label="결과 정보",
                    interactive=False,
                    elem_id="result_info"
                )
            
            # 처리 결과 상태 저장
            processed_images_state = gr.State([])
            
            # 스파스 매칭 시각화 섹션 추가
            with gr.Accordion("스파스 매칭 시각화", open=False):
                # 탭을 제거하고 단일 이미지로 변경
                sparse_match_vis = gr.Image(
                    label="현재 설정에 따른 스파스 매칭 시각화",
                    type="numpy",
                    interactive=False
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        img1_points = gr.Image(
                            label="메모리 이미지 특징점",
                            type="numpy",
                            interactive=False
                        )
                    with gr.Column(scale=1):
                        img2_points = gr.Image(
                            label="현재 이미지 특징점",
                            type="numpy",
                            interactive=False
                        )
            
            # 이벤트 핸들러 연결
            resize_ratio.change(
                fn=self.handle_resize_change,
                inputs=[resize_ratio],
                outputs=[]
            )
            
            # 배경 매칭 상태 동기화
            def _update_match_background(v):
                self.match_background = bool(v)
                if hasattr(self, 'memory_sam'):
                    self.memory_sam.match_background = bool(v)
                return

            match_background.change(
                fn=_update_match_background,
                inputs=[match_background],
                outputs=[]
            )

            # 전경 KMeans 옵션 동기화
            def _update_positive_kmeans(use_km, n_clusters):
                setattr(self.memory_sam, 'use_positive_kmeans', bool(use_km))
                setattr(self.memory_sam, 'positive_kmeans_clusters', int(n_clusters))
                return

            use_positive_kmeans.change(
                fn=_update_positive_kmeans,
                inputs=[use_positive_kmeans, positive_kmeans_clusters],
                outputs=[]
            )

            positive_kmeans_clusters.change(
                fn=_update_positive_kmeans,
                inputs=[use_positive_kmeans, positive_kmeans_clusters],
                outputs=[]
            )

            # 하이퍼파라미터 변경 이벤트
            similarity_threshold.change(
                fn=self.update_similarity_threshold,
                inputs=[similarity_threshold],
                outputs=[]
            )
            
            background_weight.change(
                fn=self.update_background_weight,
                inputs=[background_weight],
                outputs=[]
            )
            
            # 최대 포인트 수 변경 이벤트 추가
            max_positive_points.change(
                fn=self.update_max_positive_points,
                inputs=[max_positive_points],
                outputs=[]
            )
            
            max_negative_points.change(
                fn=self.update_max_negative_points,
                inputs=[max_negative_points],
                outputs=[]
            )
            
            # 클러스터링 옵션 변경 시 시각화 업데이트
            def update_visualization_on_clustering_change(skip, hybrid):
                """클러스터링 옵션 변경 시 시각화 업데이트"""
                if not hasattr(self.memory_sam, 'current_image') or self.memory_sam.current_image is None:
                    return None, None, None
                
                try:
                    # 현재 설정 저장
                    self.skip_clustering = skip
                    self.hybrid_clustering = hybrid
                    
                    # memory_sam에도 설정 적용
                    self.memory_sam.skip_clustering = skip
                    self.memory_sam.hybrid_clustering = hybrid
                    
                    # 메모리 항목이 있는지 확인
                    if not hasattr(self.memory_sam, 'similar_items') or not self.memory_sam.similar_items:
                        return None, None, None
                    
                    # 첫 번째 메모리 항목 가져오기
                    best_item = self.memory_sam.similar_items[0]["item"]
                    item_data = self.memory_sam.memory.load_item_data(best_item["id"])
                    
                    if "image" not in item_data or "mask" not in item_data:
                        return None, None, None
                    
                    memory_image = item_data["image"]
                    memory_mask = item_data["mask"]
                    
                    # 시각화 생성
                    if not self.memory_sam.current_mask is None:
                        sparse_vis, img1_vis, img2_vis = self.memory_sam.visualize_sparse_matches(
                            memory_image, 
                            self.memory_sam.current_image, 
                            memory_mask, 
                            self.memory_sam.current_mask,
                            skip_clustering=skip,
                            hybrid_clustering=hybrid,
                            match_background=self.match_background if hasattr(self, 'match_background') else True
                        )
                        
                        # 저장 경로 생성
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        clustering_mode = "skip" if skip else "hybrid" if hybrid else "standard"
                        save_path = str(self.results_dir / f"sparse_match_{timestamp}_{clustering_mode}.png")
                        
                        # 결과 저장
                        cv2.imwrite(save_path, cv2.cvtColor(sparse_vis, cv2.COLOR_RGB2BGR))
                        
                        # 현재 처리된 이미지 정보도 업데이트 (중요: 갤러리에 저장된 이미지도 업데이트)
                        if hasattr(self, 'processed_images') and self.processed_images:
                            for img_data in self.processed_images:
                                if 'sparse_match_visualization' in img_data:
                                    img_data['sparse_match_visualization'] = sparse_vis
                                if 'img1_points' in img_data:
                                    img_data['img1_points'] = img1_vis
                                if 'img2_points' in img_data:
                                    img_data['img2_points'] = img2_vis
                        
                        # 세그멘테이션 모듈의 저장된 결과도 업데이트
                        if hasattr(self.segmentation_module, 'last_result') and self.segmentation_module.last_result:
                            if 'sparse_match_visualization' in self.segmentation_module.last_result:
                                self.segmentation_module.last_result['sparse_match_visualization'] = sparse_vis
                            if 'img1_points' in self.segmentation_module.last_result:
                                self.segmentation_module.last_result['img1_points'] = img1_vis
                            if 'img2_points' in self.segmentation_module.last_result:
                                self.segmentation_module.last_result['img2_points'] = img2_vis
                        
                        return sparse_vis, img1_vis, img2_vis
                    
                    return None, None, None
                    
                except Exception as e:
                    import traceback
                    print(f"시각화 업데이트 중 오류: {e}")
                    traceback.print_exc()
                    return None, None, None
            
            # 클러스터링 옵션 변경 이벤트 연결
            skip_clustering.change(
                fn=update_visualization_on_clustering_change,
                inputs=[skip_clustering, hybrid_clustering],
                outputs=[sparse_match_vis, img1_points, img2_points]
            )
            
            hybrid_clustering.change(
                fn=update_visualization_on_clustering_change,
                inputs=[skip_clustering, hybrid_clustering],
                outputs=[sparse_match_vis, img1_points, img2_points]
            )
            
            # 이미지 처리 및 상태 업데이트 래퍼 함수
            def process_image_with_progress(
                files, folder_path, reference_path, use_reference, prompt_type, 
                use_sparse_matching, match_background, skip_clustering, state, progress=gr.Progress()
            ):
                try:
                    result = self.process_image_and_update_state(
                        files, folder_path, reference_path, use_reference, prompt_type, 
                        use_sparse_matching, match_background, skip_clustering, state, progress
                    )
                    
                    # 결과가 정상적으로 반환된 경우
                    if isinstance(result, tuple) and len(result) >= 12 and result[0] is not None:
                        return result
                    else:
                        # 오류 발생 시 기본값 반환
                        print("예상치 못한 결과 형식:", type(result), "길이:", len(result) if hasattr(result, "__len__") else "N/A")
                        return tuple([None] * 12 + [[]])
                except Exception as e:
                    import traceback
                    print(f"process_image_with_progress 오류: {e}")
                    traceback.print_exc()
                    return tuple([None] * 12 + [[]])
            
            # 처리 버튼 이벤트 연결
            process_btn.click(
                fn=process_image_with_progress,
                inputs=[
                    memory_image_input, folder_path_input, 
                    reference_image, use_reference, prompt_type, 
                    use_sparse_matching, match_background, skip_clustering, processed_images_state
                ],
                outputs=[
                    result_image, mask_image, memory_gallery, memory_info,
                    result_gallery, selected_original, selected_mask, selected_overlay, result_info,
                    sparse_match_vis, img1_points, img2_points,
                    processed_images_state
                ]
            )
            
            # 결과 갤러리 선택 이벤트 핸들러
            result_gallery.select(
                fn=self.handle_result_gallery_select,
                inputs=[result_gallery, processed_images_state],
                outputs=[selected_original, selected_mask, selected_overlay, result_info]
            )
            
            # 버튼 이벤트 연결
            save_btn.click(
                fn=self.segmentation_module.save_to_memory,
                inputs=[],
                outputs=[memory_info]
            )
            
            view_memory_btn.click(
                fn=self.memory_manager_module.view_memory,
                inputs=[],
                outputs=[memory_gallery, memory_info]
            )
    
    def handle_resize_change(self, resize_option: str):
        """리사이징 옵션 변경 처리"""
        # UI의 문자열 선택을 백엔드가 이해할 수 있는 값으로 변환
        if resize_option == "512x512 고정":
            self.current_resize_scale = "512x512"
        else: # "원본 이미지" 또는 다른 경우
            self.current_resize_scale = 1.0
        
        print(f"이미지 리사이징 옵션 변경: '{resize_option}' -> internal value: {self.current_resize_scale}")
    
    def update_similarity_threshold(self, threshold: float):
        """유사도 임계값 업데이트"""
        self.similarity_threshold = threshold
        print(f"유사도 임계값이 {threshold}로 설정되었습니다.")
    
    def update_background_weight(self, weight: float):
        """배경 가중치 업데이트"""
        self.background_weight = weight
        print(f"배경 가중치가 {weight}로 설정되었습니다.")
    
    def update_max_positive_points(self, max_points: int):
        """최대 전경 포인트 수 업데이트"""
        self.memory_sam.max_positive_points = max_points
        print(f"최대 전경 포인트 수가 {max_points}로 설정되었습니다.")
    
    def update_max_negative_points(self, max_points: int):
        """최대 배경 포인트 수 업데이트"""
        self.memory_sam.max_negative_points = max_points
        print(f"최대 배경 포인트 수가 {max_points}로 설정되었습니다.")
    
    def process_image_and_update_state(
        self, files, folder_path, reference_path, use_reference, prompt_type, 
        use_sparse_matching, match_background, skip_clustering, state, progress=gr.Progress()
    ):
        """이미지 처리 및 상태 업데이트 메서드"""
        # 파일 또는 폴더 경로 처리
        input_files = files
        if folder_path and not input_files:
            input_files = folder_path
        
        # 진행 상황 추적 설정
        total_files = 1
        if isinstance(input_files, str) and os.path.isdir(input_files):
            # 폴더 내 이미지 파일 수 계산
            image_files = FileManager.collect_image_files(input_files)
            total_files = len(image_files)
        elif isinstance(input_files, list):
            total_files = len(input_files)
        
        # 리사이징 설정 적용
        self.memory_sam.resize_images = True
        self.memory_sam.resize_scale = self.current_resize_scale
        
        # 클러스터링 하이퍼파라미터 설정
        self.memory_sam.similarity_threshold = self.similarity_threshold
        self.memory_sam.background_weight = self.background_weight
        self.memory_sam.skip_clustering = skip_clustering  # UI의 값으로 설정
        self.memory_sam.hybrid_clustering = self.hybrid_clustering  # 내부 값으로 설정
        
        # 내부 변수도 업데이트 (클러스터링 설정 동기화)
        self.skip_clustering = skip_clustering
        
        # 진행 상황 표시 초기화
        progress(0, desc="이미지 처리 준비 중...")
        
        # 이미지 처리 실행
        try:
            results = self.segmentation_module.process_image(
                input_files, 
                reference_path if use_reference else None, 
                prompt_type,
                use_sparse_matching, 
                match_background,
                skip_clustering,
                auto_add_to_memory=False
            )
            
            # 마지막 결과 저장 (중요: 후속 시각화 업데이트를 위해)
            if isinstance(results, tuple) and len(results) >= 1 and results[0] is not None:
                # 스파스 매칭 시각화 결과를 segmentation_module에 저장
                sparse_match_vis = results[9] if len(results) > 9 else None
                img1_points = results[10] if len(results) > 10 else None
                img2_points = results[11] if len(results) > 11 else None
                
                # 마지막 결과 저장
                self.segmentation_module.last_result = {
                    'image': results[0],
                    'mask': results[1],
                    'sparse_match_visualization': sparse_match_vis,
                    'img1_points': img1_points,
                    'img2_points': img2_points
                }
            
            # 결과가 정상적으로 반환된 경우
            if isinstance(results, tuple) and len(results) >= 12 and results[0] is not None:
                # 처리된 이미지 목록 저장
                results_list = list(results)
                
                # 클러스터링 설정 저장 (중요: 일관성 유지를 위해)
                processed_images = self.segmentation_module.processed_images
                if processed_images:
                    for img_data in processed_images:
                        img_data['skip_clustering'] = skip_clustering
                        img_data['hybrid_clustering'] = self.hybrid_clustering
                
                results_list.append(processed_images)
                return tuple(results_list)
            
            # 오류 발생 시
            if isinstance(results, tuple):
                results_list = list(results)
                results_list.append([])
                return tuple(results_list)
            else:
                # 결과가 튜플이 아닌 경우 처리
                return [None] * 12 + [[]]
            
        except Exception as e:
            import traceback
            print(f"이미지 처리 중 오류 발생: {e}")
            traceback.print_exc()
            # 오류 발생 시 빈 결과 반환
            return tuple([None] * 12 + [[]])
    
    def handle_result_gallery_select(self, evt, processed_images):
        """결과 갤러리 선택 이벤트 처리"""
        if not processed_images or evt is None:
            return None, None, None, "선택된 이미지가 없습니다."
        
        try:
            # 선택된 이미지 인덱스
            index = evt.index if hasattr(evt, 'index') else 0
            
            if index < 0 or index >= len(processed_images):
                return None, None, None, "유효하지 않은 선택입니다."
            
            # 선택된 이미지 항목
            item = processed_images[index]
            
            # 클러스터링 설정 적용 (중요: 이미지 선택 시 원래 처리에 사용된 설정 복원)
            if 'skip_clustering' in item:
                self.memory_sam.skip_clustering = item['skip_clustering']
                self.skip_clustering = item['skip_clustering']
            
            if 'hybrid_clustering' in item:
                self.memory_sam.hybrid_clustering = item['hybrid_clustering']
                self.hybrid_clustering = item['hybrid_clustering']
            
            # 결과 이미지 가져오기
            original_img = item.get("input")
            mask_img = item.get("mask")
            overlay_img = item.get("overlay")
            
            # 정보 텍스트 생성
            info_text = f"파일: {item.get('path', '알 수 없음')}\n"
            info_text += f"크기: {item.get('width', 0)}x{item.get('height', 0)}\n"
            info_text += f"리사이징: {item.get('resize_scale', 1.0):.2f}\n"
            
            if "processing_time" in item:
                info_text += f"처리 시간: {item.get('processing_time', 0):.2f}초\n"
            
            # 클러스터링 설정 정보 추가
            info_text += f"클러스터링: {'건너뛰기' if item.get('skip_clustering', False) else '적용'}\n"
            info_text += f"하이브리드 클러스터링: {'적용' if item.get('hybrid_clustering', False) else '미적용'}\n"
            
            return original_img, mask_img, overlay_img, info_text
            
        except Exception as e:
            print(f"결과 갤러리 선택 처리 중 오류: {e}")
            return None, None, None, f"오류 발생: {str(e)}"
    
    def _setup_mask_generator_tab(self):
        """마스크 생성 탭 설정"""
        with gr.TabItem("마스크 생성"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 리사이징 옵션
                    mask_resize_ratio = UIComponents.create_resize_buttons()
                    
                    # 입력 이미지 (클릭 이벤트 전달용으로 type="numpy" 설정)
                    mask_creator_image = gr.Image(
                        label="입력 이미지 - 클릭하여 포인트 추가", 
                        type="numpy", 
                        height=450,
                        interactive=True
                    )
                    
                    # mask_prompt_type 라디오 버튼에서 "box" 옵션 제거
                    mask_prompt_type = gr.Radio(
                        choices=["points"], # "box" 옵션 제거
                        value="points",
                        label="프롬프트 타입"
                    )
                    
                    status_msg = gr.Textbox(
                        value="이미지를 로드하고 클릭하여 포인트를 추가하세요.",
                        label="상태",
                        interactive=False
                    )
                    
                    # 내부 상태 변수
                    box_coords = gr.State(None)
                    current_mask = gr.State(None)
                    current_points_state = gr.State([])
                    current_labels_state = gr.State([])
                    
                    # 포인트 컨트롤
                    with gr.Row(visible=True) as points_controls:
                        gr.Markdown("**포인트 타입 선택**")
                        pos_point_btn = gr.Button("전경 포인트 (객체)", variant="primary")
                        neg_point_btn = gr.Button("배경 포인트", variant="secondary")
                        clear_points_btn = gr.Button("모든 포인트 지우기")
                    
                    # 박스 컨트롤 전체 삭제
                    # with gr.Row(visible=False) as box_controls:
                    #     clear_box_btn = gr.Button("상자 지우기")
                    
                    # 박스 입력 컨트롤 전체 삭제
                    # with gr.Row(visible=False) as box_input_controls:
                    #     image_dimensions = gr.Textbox(
                    #         label="이미지 정보",
                    #         value="이미지를 로드하세요",
                    #         interactive=False
                    #     )
                    #     with gr.Column():
                    #         gr.Markdown("**박스 좌표 입력**")
                    #         with gr.Row():
                    #             box_x1 = gr.Number(label="X1", value=50, precision=0)
                    #             box_y1 = gr.Number(label="Y1", value=50, precision=0)
                    #         with gr.Row():
                    #             box_x2 = gr.Number(label="X2", value=200, precision=0)
                    #             box_y2 = gr.Number(label="Y2", value=200, precision=0)
                    #         apply_box_btn = gr.Button("박스 적용", variant="primary")
                    
                    with gr.Row():
                        save_mask_btn = gr.Button("마스크 저장")
                        save_as_reference_btn = gr.Button("참조 이미지로 저장")
                        save_to_memory_btn = gr.Button("메모리에 저장")
                    
                    save_mask_info = gr.Textbox(
                        value="",
                        label="저장 결과",
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    segmentation_result = gr.Image(
                        label="세그멘테이션 결과 (포인트와 마스크 시각화)", 
                        type="numpy",
                        height=450
                    )
                    
                    mask_output = gr.Image(
                        label="마스크 (저장용)", 
                        type="numpy",
                        height=200,
                        visible=True
                    )
                    
                    current_status = gr.Textbox(
                        value="포인트 타입: 전경 (객체)",
                        label="현재 포인트 타입",
                        interactive=False
                    )
            
            # 이벤트 핸들러 연결
            mask_resize_ratio.change(
                fn=self.handle_resize_change,
                inputs=[mask_resize_ratio],
                outputs=[]
            )
            
            # mask_prompt_type 변경 시 box 관련 UI 제어 로직 삭제
            # mask_prompt_type.change(
            #     fn=self.mask_generator_module.toggle_controls,
            #     inputs=[mask_prompt_type],
            #     outputs=[points_controls, box_controls, box_input_controls]
            # )
            
            # 이미지 클릭 이벤트 처리
            mask_creator_image.select(
                fn=self.handle_mask_image_click,
                inputs=[mask_creator_image, current_points_state, current_labels_state],
                outputs=[segmentation_result, current_points_state, current_labels_state]
            )
            
            # 마스크 출력 업데이트
            def update_mask_output():
                if hasattr(self.mask_generator_module, 'current_mask_vis') and self.mask_generator_module.current_mask_vis is not None:
                    return self.mask_generator_module.current_mask_vis
                return None
            
            segmentation_result.change(
                fn=update_mask_output,
                inputs=[],
                outputs=[mask_output]
            )
            
            # 포인트 타입 버튼 이벤트
            pos_point_btn.click(
                fn=lambda: self.mask_generator_module.set_point_type("전경 (객체)"),
                outputs=[current_status]
            )
            
            neg_point_btn.click(
                fn=lambda: self.mask_generator_module.set_point_type("배경"),
                outputs=[current_status]
            )
            
            # 초기화 버튼 이벤트
            def reset_images():
                return None, None
            
            clear_points_btn.click(
                fn=self.mask_generator_module.clear_all_points,
                outputs=[status_msg]
            )
            
            clear_points_btn.click(
                fn=reset_images,
                outputs=[segmentation_result, mask_output]
            )
            
            # 저장 버튼 이벤트
            save_mask_btn.click(
                fn=self.mask_generator_module.save_generated_mask,
                inputs=[mask_creator_image, mask_output],
                outputs=[save_mask_info]
            )
            
            save_as_reference_btn.click(
                fn=self.mask_generator_module.save_mask_to_reference,
                inputs=[mask_creator_image, mask_output],
                outputs=[save_mask_info]
            )
            
            save_to_memory_btn.click(
                fn=self.mask_generator_module.save_to_memory_directly,
                inputs=[mask_creator_image, mask_output],
                outputs=[save_mask_info]
            )
    
    def handle_mask_image_click(self, image, points, labels, evt: gr.SelectData):
        """마스크 이미지 클릭 이벤트 처리"""
        print(f"클릭 이벤트 감지: {evt.index}, 위치: ({evt.index[0]}, {evt.index[1]})")
        
        # 이미지가 있는지 확인
        if image is None:
            print("이미지가 없습니다. 클릭 이벤트 무시")
            # 출력 형식 유지: (segmentation_result, current_points_state, current_labels_state)
            return None, points, labels
            
        # 원본 이미지 크기 저장
        original_height, original_width = image.shape[:2]
        print(f"원본 이미지 크기: {original_width}x{original_height}")
        
        # 원본 크기 그대로 사용
        processed_image = image
        print(f"처리용 이미지 크기: {processed_image.shape[1]}x{processed_image.shape[0]}")
        
        # 클릭 시 Gradio 인터페이스가 전달하는 좌표를 분석
        x_click, y_click = evt.index
        
        # 원본 좌표계를 그대로 사용
        norm_x = int(x_click)
        norm_y = int(y_click)
        norm_x = max(0, min(norm_x, original_width-1))
        norm_y = max(0, min(norm_y, original_height-1))
        print(f"클릭 좌표 변환: 원본 좌표({x_click}, {y_click}) -> 원본 좌표 유지: ({norm_x}, {norm_y})")
        
        # 새 포인트 생성 (원본 좌표계)
        norm_point = [norm_x, norm_y]
        norm_label = 1 if self.mask_generator_module.current_point_type == "전경 (객체)" else 0
        
        # 내부 상태에 포인트와 레이블 추가
        # 기존에 갖고 있던 state 사용하기
        current_points = self.mask_generator_module.current_points.copy() if hasattr(self.mask_generator_module, "current_points") and self.mask_generator_module.current_points else []
        current_labels = self.mask_generator_module.current_point_labels.copy() if hasattr(self.mask_generator_module, "current_point_labels") and self.mask_generator_module.current_point_labels else []
        
        # 현재 포인트 추가
        current_points.append(norm_point)
        current_labels.append(norm_label)
        
        # 마스크 생성
        try:
            # SAM 예측기에 직접 이미지 설정
            predictor = self.memory_sam.predictor
            predictor.set_image(processed_image)
            
            # 포인트 마스크 생성
            points_array = np.array(current_points)
            labels_array = np.array(current_labels)
            
            masks, scores, _ = predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=True
            )
            
            # 최고 점수 마스크 선택
            if len(scores) > 0:
                best_idx = np.argmax(scores)
                mask = masks[best_idx]
                
                # 마스크 시각화
                mask_vis = (mask * 255).astype(np.uint8)
                
                # 내부 상태 업데이트
                self.mask_generator_module.current_mask = mask
                self.mask_generator_module.current_mask_vis = mask_vis
                
                # 포인트들을 이미지에 시각화
                from scripts.memory_ui_utils import draw_points_on_image
                result_img = draw_points_on_image(processed_image, current_points, current_labels, mask)
                
                print(f"마스크 생성 완료, 최고 점수: {scores[best_idx]:.4f}")
                
                # 내부 상태 업데이트
                self.mask_generator_module.current_points = current_points
                self.mask_generator_module.current_point_labels = current_labels
                
                return result_img, current_points, current_labels
            else:
                print("마스크 생성 실패: 점수가 없습니다")
                return processed_image, current_points, current_labels
                
        except Exception as e:
            import traceback
            print(f"마스크 생성 중 오류: {e}")
            traceback.print_exc()
            return processed_image, current_points, current_labels
    
    def _setup_memory_manager_tab(self):
        """메모리 관리 탭 설정"""
        with gr.TabItem("메모리 관리"):
            with gr.Row():
                with gr.Column(scale=1):
                    # 초기 메모리 데이터 로드
                    initial_memory, initial_stats = self.memory_manager_module.load_memory_display()
                    
                    # Gallery 컴포넌트 생성 시 초기값 설정
                    memory_display = gr.Gallery(
                        label="메모리 항목",
                        value=initial_memory,  # 초기값을 바로 설정
                        columns=4,
                        rows=2,
                        height=400,
                        elem_id="memory_display"
                    )
                    
                    # Textbox 컴포넌트 생성 시 초기값 설정
                    memory_stats = gr.Textbox(
                        value=initial_stats,  # 초기값을 바로 설정
                        label="메모리 통계",
                        interactive=False,
                        elem_id="memory_stats"
                    )
                    
                    refresh_memory_btn = gr.Button(value="메모리 새로고침", elem_id="refresh_memory_btn")
                
                with gr.Column(scale=1):
                    with gr.Tabs():
                        with gr.TabItem("메모리 항목"):
                            selected_memory_image = gr.Image(label="선택된 메모리 항목", elem_id="selected_memory_image")
                            selected_memory_info = gr.JSON(label="항목 정보", elem_id="selected_memory_info")
                            
                            # 선택된 항목 ID 저장
                            selected_item_id = gr.State(None)
                            
                            with gr.Row():
                                delete_item_btn = gr.Button("선택 항목 삭제", variant="secondary", elem_id="delete_item_btn")
                                item_delete_result = gr.Textbox(
                                    value="",
                                    label="항목 삭제 결과",
                                    interactive=False,
                                    elem_id="item_delete_result"
                                )
                        
                        with gr.TabItem("특징 매칭 시각화"):
                            match_visualization = gr.Image(
                                label="특징 매칭 시각화",
                                type="numpy",
                                height=500,
                                elem_id="match_visualization"
                            )
                            match_info = gr.Textbox(
                                value="메모리 항목을 선택하면 특징 매칭이 표시됩니다.",
                                label="매칭 정보",
                                interactive=False,
                                elem_id="match_info"
                            )
                            
                            # 스파스 매칭 시각화 섹션
                            with gr.Accordion("스파스 매칭 시각화", open=False):
                                sparse_match_vis = gr.Image(
                                    label="현재 설정에 따른 스파스 매칭 시각화",
                                    type="numpy",
                                    interactive=False
                                )
                                
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        img1_points = gr.Image(
                                            label="메모리 이미지 특징점",
                                            type="numpy",
                                            interactive=False
                                        )
                                    with gr.Column(scale=1):
                                        img2_points = gr.Image(
                                            label="현재 이미지 특징점",
                                            type="numpy",
                                            interactive=False
                                        )
                    
                    with gr.Row():
                        delete_memory_btn = gr.Button("메모리 초기화", variant="stop", elem_id="delete_memory_btn")
                        delete_result = gr.Textbox(
                            value="",
                            label="작업 결과",
                            interactive=False,
                            elem_id="delete_result"
                        )
            
            # 메모리 새로고침 버튼 이벤트 설정
            refresh_memory_btn.click(
                fn=self.memory_manager_module.load_memory_display,
                inputs=[],
                outputs=[memory_display, memory_stats]
            )
            
            # 메모리 초기화 버튼 이벤트 설정
            delete_memory_btn.click(
                fn=self.memory_manager_module.clear_memory,
                inputs=[],
                outputs=[delete_result]
            ).then(
                fn=self.memory_manager_module.load_memory_display,
                inputs=[],
                outputs=[memory_display, memory_stats]
            )

            # 메모리 항목 선택 이벤트 설정
            def handle_memory_select(evt: gr.SelectData):
                # 이벤트에서 인덱스 가져오기
                idx = evt.index
                
                # 선택한 인덱스로 메모리 항목 표시
                image, info = self.memory_manager_module.display_memory_item(idx)
                match_vis = None
                match_info_text = "메모리 항목이 선택되었습니다."
                
                # 항목 ID 저장
                item_id = info.get("id") if info else None
                
                # 스파스 매칭 시각화 초기화
                sparse_vis = None
                img1_vis = None
                img2_vis = None
                
                if self.memory_sam.current_image is not None and self.memory_sam.use_sparse_matching:
                    try:
                        if item_id is not None and info.get("has_patch_features", False):
                            # 메모리 항목 데이터 로드
                            item_data = self.memory_sam.memory.load_item_data(item_id)
                            
                            if "patch_features" in item_data:
                                # 스파스 매칭 시각화 생성
                                sparse_vis, img1_vis, img2_vis = self.memory_sam.visualize_sparse_matches(
                                    item_data["image"], 
                                    self.memory_sam.current_image,
                                    item_data.get("mask"),
                                    self.memory_sam.current_mask
                                )
                                match_info_text = f"ID {item_id}와 현재 이미지 간의 특징 매칭 시각화"
                            else:
                                match_info_text = "이 메모리 항목에는 패치 특징이 없습니다."
                        else:
                            match_info_text = "이 메모리 항목에는 스파스 매칭을 위한 패치 특징이 없습니다."
                    except Exception as e:
                        match_info_text = f"특징 매칭 시각화 오류: {str(e)}"
                        import traceback
                        traceback.print_exc()
                else:
                    match_info_text = "스파스 매칭이 비활성화되었거나 현재 이미지가 없습니다."
                
                return image, info, sparse_vis, match_info_text, item_id, sparse_vis, img1_vis, img2_vis

            memory_display.select(
                fn=handle_memory_select,
                inputs=[],
                outputs=[selected_memory_image, selected_memory_info, match_visualization, match_info, selected_item_id, sparse_match_vis, img1_points, img2_points]
            )
            
            # 선택 항목 삭제 버튼 이벤트 설정
            def delete_selected_item(item_id):
                if item_id is None:
                    return "삭제할 항목이 선택되지 않았습니다."
                
                result = self.memory_manager_module.delete_memory_item(item_id)
                return result
            
            delete_item_btn.click(
                fn=delete_selected_item,
                inputs=[selected_item_id],
                outputs=[item_delete_result]
            ).then(
                fn=self.memory_manager_module.load_memory_display,
                inputs=[],
                outputs=[memory_display, memory_stats]
            )