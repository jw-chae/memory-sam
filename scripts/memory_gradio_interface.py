import os
import sys
import numpy as np
import torch
import gradio as gr
import tempfile
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path

from scripts.memory_sam_predictor import MemorySAMPredictor
from scripts.segmentation_module import SegmentationModule
from scripts.mask_generator_module import MaskGeneratorModule
from scripts.memory_manager_module import MemoryManagerModule
from scripts.memory_ui_utils import browse_directory

class MemoryGradioInterface:
    """Memory SAM 시스템을 위한 Gradio 인터페이스"""
    
    def __init__(self, 
                memory_sam_predictor=None, 
                model_type="hiera_l", 
                checkpoint_path=None, 
                dinov2_model="facebook/dinov2-base",
                dinov2_matching_repo="facebookresearch/dinov2",
                dinov2_matching_model="dinov2_vitb14",
                memory_dir="memory", 
                results_dir="results",
                use_sparse_matching=True):
        """
        Memory SAM 이미지 세그멘테이션을 위한 Gradio 인터페이스
        
        Args:
            memory_sam_predictor: Memory SAM 예측기 인스턴스 (None이면 새로 생성)
            model_type: 사용할 모델 타입
            checkpoint_path: 체크포인트 경로
            dinov2_model: DINOv2 모델 이름
            dinov2_matching_repo: DINOv2 스파스 매칭용 리포지토리
            dinov2_matching_model: DINOv2 스파스 매칭용 모델
            memory_dir: 메모리 디렉토리
            results_dir: 결과 디렉토리
            use_sparse_matching: 스파스 매칭 사용 여부
        """
        # 초기화: MemorySAMPredictor 인스턴스
        if memory_sam_predictor is None:
            self.memory_sam = MemorySAMPredictor(
                model_type=model_type,
                checkpoint_path=checkpoint_path,
                dinov2_model=dinov2_model,
                dinov2_matching_repo=dinov2_matching_repo,
                dinov2_matching_model=dinov2_matching_model,
                memory_dir=memory_dir,
                results_dir=results_dir,
                use_sparse_matching=use_sparse_matching
            )
        else:
            self.memory_sam = memory_sam_predictor
        
        # 모듈 초기화
        self.segmentation_module = SegmentationModule(self.memory_sam)
        self.mask_generator_module = MaskGeneratorModule(self.memory_sam)
        self.memory_manager_module = MemoryManagerModule(self.memory_sam)
        
        # 결과 디렉토리 설정
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # 임시 디렉토리
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # 아코디언 상태 초기화
        self._accordion_state = False
    
    def _update_accordion_state(self, open_state: bool):
        """결과 아코디언 상태 업데이트"""
        self._accordion_state = open_state
    
    def get_accordion_state(self):
        """아코디언 상태 반환"""
        if hasattr(self, '_accordion_state'):
            return gr.update(open=self._accordion_state)
        return gr.update(open=False)
    
    def __del__(self):
        """소멸자: 임시 디렉토리 정리"""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()

    def setup_interface(self):
        """Gradio 인터페이스 설정"""
        with gr.Blocks(title="Memory SAM - 이미지 세그멘테이션") as interface:
            gr.Markdown("# Memory SAM - 이미지 세그멘테이션")
            gr.Markdown("SAM2와 DINOv2 기반의 메모리 시스템을 활용한 지능형 이미지 세그멘테이션")
            
            with gr.Tabs():
                self._setup_segmentation_tab()
                self._setup_mask_generator_tab()
                self._setup_memory_manager_tab()
        
        return interface
    
    def _setup_segmentation_tab(self):
        """세그멘테이션 탭 설정"""
        with gr.TabItem("메모리 기반 세그멘테이션"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 이미지 또는 폴더 선택")
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
                            # 폴더 경로 직접 입력
                            folder_path_input = gr.Textbox(
                                label="폴더 경로 직접 입력 (선택적)",
                                placeholder="/path/to/folder",
                                elem_id="folder_path_input"
                            )
                            folder_browse_btn = gr.Button("폴더 찾아보기", elem_id="folder_browse_btn")
                    
                    gr.Markdown("프롬프트 타입: 포인트 기반")
                    
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
                        skip_clustering = gr.Checkbox(
                            label="클러스터링 건너뛰기", 
                            value=False,
                            info="클러스터링 없이 모든 스파스 매칭을 시각화합니다."
                        )
                    
                    with gr.Accordion("참조 이미지 (선택 사항)", open=False):
                        reference_image = gr.Image(label="참조 이미지", type="filepath")
                        use_reference = gr.Checkbox(label="참조 이미지 사용", value=False)
                    
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
                    result_gallery = gr.Gallery(
                        label="처리된 이미지 결과", 
                        columns=3,
                        rows=2,
                        height=400,
                        elem_id="result_gallery"
                    )
                
                # 탭으로 결과 유형 선택
                with gr.Tabs(elem_id="result_tabs"):
                    with gr.TabItem("원본"):
                        selected_original = gr.Image(
                            label="선택된 원본 이미지", 
                            interactive=False,
                            elem_id="selected_original"
                        )
                    with gr.TabItem("마스크", elem_id="mask_tab"):
                        selected_mask = gr.Image(
                            label="선택된 마스크", 
                            interactive=False,
                            elem_id="selected_mask"
                        )
                    with gr.TabItem("오버레이"):
                        selected_overlay = gr.Image(
                            label="선택된 오버레이", 
                            interactive=False,
                            elem_id="selected_overlay"
                        )
                
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
                with gr.Row():
                    sparse_match_vis = gr.Image(
                        label="스파스 매칭 시각화",
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
            folder_browse_btn.click(
                fn=browse_directory,
                inputs=[],
                outputs=[folder_path_input]
            )
            
            # 이미지 처리 및 상태 업데이트 래퍼 함수
            def process_image_and_update_state(files, folder_path, reference_path, use_reference, 
                                            use_sparse_matching, match_background, skip_clustering, state):
                # 파일 또는 폴더 경로 처리
                input_data = self.segmentation_module.prepare_input_data(files, folder_path)
                if not input_data:
                    return state, None, None, [], "이미지를 선택하거나 폴더 경로를 입력하세요.", [], None, None, None, "입력 없음", None, None, None, gr.update(open=self._accordion_state)

                # prompt_type을 "points"로 고정
                prompt_type_fixed = "points"

                (seg_vis, mask_vis, memory_gallery_items, memory_info_text, 
                 result_gallery_items, selected_original, selected_mask, selected_overlay, result_info_text,
                 sparse_match_vis, img1_points, img2_points) = self.segmentation_module.process_image(
                    files=files, 
                    folder_path=folder_path,
                    reference_path=reference_path,
                    use_reference=use_reference,
                    prompt_type=prompt_type_fixed,
                    use_sparse_matching=use_sparse_matching,
                    match_background=match_background,
                    skip_clustering=skip_clustering,
                    auto_add_to_memory=state.get("auto_add_to_memory", False) 
                )
                
                # 결과가 정상적으로 반환된 경우
                if len(result_gallery_items) >= 12 and result_gallery_items[0] is not None:  # 12개 결과 (스파스 매칭 시각화 포함)
                    # 결과 아코디언 열기 상태 설정
                    self._update_accordion_state(True)
                    # 처리된 이미지 목록 저장
                    return result_gallery_items + (self.segmentation_module.processed_images,)
                
                # 오류 발생 시
                self._update_accordion_state(False)
                return result_gallery_items + ([],)  # 빈 처리 이미지 목록 반환
            
            # 처리 버튼 이벤트 연결
            process_btn.click(
                fn=process_image_and_update_state,
                inputs=[
                    memory_image_input, folder_path_input, reference_image, use_reference,
                    use_sparse_matching, match_background, skip_clustering, processed_images_state
                ],
                outputs=[
                    result_image, mask_image, memory_gallery, memory_info,
                    result_gallery, selected_original, selected_mask, selected_overlay, result_info,
                    sparse_match_vis, img1_points, img2_points, processed_images_state
                ]
            ).then(
                fn=self.get_accordion_state,
                inputs=[],
                outputs=[results_accordion]
            )
            
            # 결과 갤러리 선택 이벤트 핸들러
            result_gallery.select(
                fn=self.segmentation_module.handle_result_gallery_select,
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
    
    def _setup_mask_generator_tab(self):
        """마스크 직접 생성 탭 설정"""
        with gr.TabItem("마스크 직접 생성"):
            gr.Markdown("이미지에 포인트를 클릭하여 마스크를 생성합니다.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 1. 이미지 업로드")
                    mg_image_input = gr.Image(label="입력 이미지", type="numpy", tool=None, elem_id="mg_image_input")
                    
                    gr.Markdown("### 2. 포인트 타입 선택")
                    mg_point_type_radio = gr.Radio(
                        choices=["전경 (객체)", "배경 (제외)"], 
                        value="전경 (객체)", 
                        label="포인트 타입",
                        elem_id="mg_point_type_radio"
                    )
                    
                    mg_clear_points_btn = gr.Button("모든 포인트 초기화", elem_id="mg_clear_points_btn")

                    gr.Markdown("### 3. 결과 저장")
                    with gr.Row():
                        mg_save_mask_btn = gr.Button("생성된 마스크 저장", elem_id="mg_save_mask_btn")
                        mg_save_to_ref_btn = gr.Button("참조로 저장", elem_id="mg_save_to_ref_btn")
                        mg_save_to_mem_btn = gr.Button("메모리에 저장", elem_id="mg_save_to_mem_btn")
                    
                    mg_status_text = gr.Textbox(label="상태", interactive=False, elem_id="mg_status_text")

                with gr.Column(scale=1):
                    gr.Markdown("### 마스크 생성 결과")
                    mg_image_display = gr.ImageEditor(
                        label="이미지 (클릭하여 포인트 추가)", 
                        type="numpy",
                        elem_id="mg_image_display"
                    ) 
                    mg_mask_output = gr.Image(label="생성된 마스크", type="numpy", elem_id="mg_mask_output")

            # 상태 저장을 위한 Gradio State 변수들
            mg_points_state = gr.State([])
            mg_labels_state = gr.State([])

            # --- 이벤트 핸들러 ---
            
            mg_point_type_radio.change(
                fn=self.mask_generator_module.set_point_type,
                inputs=[mg_point_type_radio],
                outputs=[mg_status_text]
            )

            # 이미지 클릭(선택) 이벤트 처리
            mg_image_display.select(
                self.mask_generator_module.handle_point_selection_state,
                inputs=[mg_image_input, mg_points_state, mg_labels_state],
                outputs=[mg_image_display, mg_points_state, mg_labels_state]
            ).then(
                fn=update_mask_output,
                inputs=None,
                outputs=[mg_mask_output]
            )

            mg_clear_points_btn.click(
                fn=self.mask_generator_module.clear_all_points,
                inputs=None,
                outputs=[mg_status_text]
            ).then(
                fn=reset_images,
                inputs=None,
                outputs=[mg_image_display, mg_mask_output, mg_points_state, mg_labels_state]
            )
            
            # 이미지 업로드 시 MaskGeneratorModule 상태 초기화 및 디스플레이 업데이트
            mg_image_input.upload(
                fn=self.mask_generator_module.clear_all_points,
                inputs=None,
                outputs=[mg_status_text]
            ).then(
                fn=lambda img: (img, [], [], img, None),
                inputs=[mg_image_input],
                outputs=[mg_image_display, mg_points_state, mg_labels_state, mg_mask_output]
            )

            def update_mask_output():
                return self.mask_generator_module.current_mask_vis

            def reset_images():
                return None, None, [], []
            
            # 저장 버튼들의 이벤트 핸들러는 MaskGeneratorModule의 내부 상태(current_image, current_mask)를 사용하므로,
            # 해당 함수들이 박스에 의존하지 않는다면 그대로 사용 가능.
            # MaskGeneratorModule의 save_xxx 함수들은 image와 mask를 인자로 받음.
            # UI에서 현재 이미지(mg_image_input)와 생성된 마스크(mg_mask_output 또는 내부 current_mask)를 전달해야 함.

            mg_save_mask_btn.click(
                fn=self.mask_generator_module.save_generated_mask,
                inputs=[mg_image_input, mg_mask_output],
                outputs=[mg_status_text]
            )
            mg_save_to_ref_btn.click(
                fn=self.mask_generator_module.save_mask_to_reference,
                inputs=[mg_image_input, mg_mask_output],
                outputs=[mg_status_text]
            )
            mg_save_to_mem_btn.click(
                fn=self.mask_generator_module.save_to_memory_directly,
                inputs=[mg_image_input, mg_mask_output],
                outputs=[mg_status_text]
            )
    
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
                image, info = self.memory_manager_module.display_memory_item(evt.index)
                match_vis = None
                match_info_text = "메모리 항목이 선택되었습니다."
                
                # 항목 ID 저장
                item_id = info.get("id") if info else None
                
                if self.memory_sam.current_image is not None and self.memory_sam.use_sparse_matching:
                    try:
                        if item_id is not None and info.get("has_patch_features", False):
                            match_vis = self.memory_manager_module.visualize_memory_matches(
                                item_id,
                                self.memory_sam.current_image
                            )
                            match_info_text = f"ID {item_id}와 현재 이미지 간의 특징 매칭 시각화" if match_vis is not None else "특징 매칭을 시각화할 수 없습니다."
                        else:
                            match_info_text = "이 메모리 항목에는 스파스 매칭을 위한 패치 특징이 없습니다."
                    except Exception as e:
                        match_info_text = f"특징 매칭 시각화 오류: {str(e)}"
                else:
                    match_info_text = "스파스 매칭이 비활성화되었거나 현재 이미지가 없습니다."
                
                return image, info, match_vis, match_info_text, item_id

            memory_display.select(
                fn=handle_memory_select,
                inputs=[],
                outputs=[selected_memory_image, selected_memory_info, match_visualization, match_info, selected_item_id]
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



if __name__ == "__main__":
    interface = MemoryGradioInterface(use_sparse_matching=True).setup_interface()
    interface.launch(share=True)