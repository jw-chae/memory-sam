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
from scripts.memory_ui_utils import SparseMatchVisualizer, draw_points_on_image, prepare_input_data

class MemorySAMUI:
    """UI Class for Enhanced Memory SAM System"""
    
    def __init__(self, 
                model_type: str = "hiera_l", 
                checkpoint_path: str = None,
                dinov3_model: str = "dinov3_vitb16",
                memory_dir: str = "memory", 
                results_dir: str = "results",
                device: str = "cuda"):
        """
        Initialize Memory SAM UI
        """
        # Initialize Memory SAM predictor
        self.memory_sam = MemorySAMPredictor(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            dinov3_model=dinov3_model,
            memory_dir=memory_dir,
            results_dir=results_dir,
            device=device
        )
        
        # Initialize sparse match visualizer
        self.sparse_match_visualizer = SparseMatchVisualizer(self.memory_sam)
        
        # Set up results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Temporary directory
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # State variables
        self.processed_images_data = [] # Store full data for gallery selection
        self.current_point_type = "전경 (객체)"

    
    def __del__(self):
        """Destructor: clean up temporary directory"""
        if hasattr(self, 'temp_dir'):
            self.temp_dir.cleanup()
    
    def setup_interface(self):
        """Set up enhanced Gradio interface"""
        with gr.Blocks(title="Memory SAM - Image Segmentation") as interface:
            gr.Markdown("# Memory SAM - Image Segmentation")
            gr.Markdown("Intelligent image segmentation using SAM2 and DINOv3 with a memory system")
            
            with gr.Tabs():
                with gr.TabItem("메모리 기반 세그멘테이션"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 이미지 또는 폴더 선택")
                            with gr.Row():
                                memory_image_input = gr.File(label="입력 이미지", file_count="multiple", file_types=["image"])
                            with gr.Row():
                                folder_path_input = gr.Textbox(label="폴더 경로", placeholder="/path/to/images", scale=3)
                                browse_folder_btn = gr.Button("폴더 찾기", variant="secondary", scale=1)
                            gr.Markdown("*💡 **이미지 처리**: 여러 이미지 파일을 선택하여 처리 | **폴더 처리**: 폴더 경로 입력 또는 폴더 찾기 버튼으로 선택*")
                            with gr.Row():
                                process_images_btn = gr.Button("이미지 처리", variant="primary")
                                process_folder_btn = gr.Button("폴더 처리", variant="secondary")
                            gr.Markdown("*🔘 **이미지 처리**: 선택된 이미지 파일들을 처리 | **폴더 처리**: 선택된 폴더 내 모든 이미지를 일괄 처리*")
                            
                            gr.Markdown("### 처리 옵션")
                            match_background = gr.Checkbox(label="배경 매칭 포함", value=False)
                            use_kmeans = gr.Checkbox(label="K-Means 클러스터링 사용", value=True)
                            kmeans_fg_clusters = gr.Number(label="전경 K-Means 클러스터 수", value=10, precision=0)
                            gr.Markdown("*📋 **배경 매칭**: 배경 영역도 매칭하여 더 정확한 세그멘테이션 수행*")
                            gr.Markdown("*📋 **K-Means 클러스터링**: 특징점을 클러스터링하여 대표점만 선택*")
                            gr.Markdown("*📋 **전경 클러스터 수**: 전경 영역에서 선택할 대표 특징점 수*")

                        with gr.Column(scale=2):
                            result_gallery = gr.Gallery(label="처리된 이미지 결과", columns=4, height=300)
                            with gr.Tabs():
                                with gr.TabItem("결과 (오버레이)"):
                                    selected_overlay = gr.Image(label="선택된 이미지")
                                with gr.TabItem("원본"):
                                    selected_original = gr.Image(label="선택된 이미지 (원본)")
                                with gr.TabItem("마스크"):
                                    selected_mask = gr.Image(label="선택된 마스크")
                            result_info = gr.Textbox(label="처리 결과 정보", interactive=False, lines=3)
                            with gr.Accordion("가장 유사한 메모리 Top 5", open=True):
                                top5_memory_gallery = gr.Gallery(label="유사도 순 메모리 항목", columns=5, rows=1, height="auto", object_fit="contain")
                            with gr.Accordion("스파스 매칭 시각화", open=True):
                                gr.Markdown("**전체 매칭 결과 (좌: 메모리 이미지, 우: 현재 이미지)**")
                                sparse_match_vis = gr.Image(label="스파스 매칭 시각화", type="numpy", interactive=False, height=400)
                                gr.Markdown("**개별 이미지 특징점 분석**")
                                with gr.Row():
                                    with gr.Column(scale=1):
                                        gr.Markdown("**메모리 이미지 특징점**")
                                        img1_points = gr.Image(label="메모리 이미지 특징점", type="numpy", interactive=False, height=300)
                                    with gr.Column(scale=1):
                                        gr.Markdown("**현재 이미지 특징점**")
                                        img2_points = gr.Image(label="현재 이미지 특징점", type="numpy", interactive=False, height=300)

                self._setup_mask_generator_tab()
                self._setup_memory_manager_tab(interface)
        
            # --- State and Event Handlers for Segmentation Tab ---
            processed_images_state = gr.State([])

            # 이미지 처리 버튼
            process_images_btn.click(
                fn=self.process_images_and_update_ui,
                inputs=[memory_image_input, match_background, use_kmeans, kmeans_fg_clusters],
                outputs=[
                    result_gallery, selected_overlay, selected_original, selected_mask, 
                    result_info, top5_memory_gallery, sparse_match_vis, 
                    img1_points, img2_points, processed_images_state
                ]
            )
            
            # 폴더 찾기 버튼
            browse_folder_btn.click(
                fn=self.browse_folder,
                outputs=[folder_path_input]
            )
            
            # 폴더 처리 버튼
            process_folder_btn.click(
                fn=self.process_folder_and_update_ui,
                inputs=[folder_path_input, match_background, use_kmeans, kmeans_fg_clusters],
                outputs=[
                    result_gallery, selected_overlay, selected_original, selected_mask, 
                    result_info, top5_memory_gallery, sparse_match_vis, 
                    img1_points, img2_points, processed_images_state
                ]
            )

            result_gallery.select(
                fn=self.handle_result_gallery_select,
                inputs=[processed_images_state],
                outputs=[
                    selected_overlay, selected_original, selected_mask, result_info, 
                    top5_memory_gallery, sparse_match_vis, img1_points, img2_points
                ]
            )
            
            for control, name in [(match_background, "배경 매칭 포함"), (use_kmeans, "K-Means 클러스터링")]:
                control.change(
                    fn=lambda value, n=name: gr.Info(f"{n} 설정이 {'활성' if value else '비활성'}화 되었습니다. '처리' 버튼을 눌러 적용하세요."),
                    inputs=[control], outputs=[]
                )
        
        return interface
    
    def process_images_and_update_ui(self, files, match_bg, use_kmeans, kmeans_k, progress=gr.Progress()):
        if not files:
            gr.Info("처리할 파일을 업로드하세요.")
            return [None] * 9 + [[]]

        file_paths = [f.name for f in files] if isinstance(files, list) else [files.name]
        all_results_data = []
        gallery_images = []

        progress(0, desc="이미지 처리 시작...")
        for i, file_path in enumerate(file_paths):
            progress((i + 1) / len(file_paths), desc=f"{os.path.basename(file_path)} 처리 중...")
            
            self.memory_sam.use_kmeans_fg = use_kmeans
            self.memory_sam.kmeans_fg_clusters = int(kmeans_k)
            self.memory_sam.skip_clustering = not use_kmeans

            results = self.memory_sam.process_image(
                image_path=file_path,
                match_background=match_bg,
            )
            if "error" in results:
                gr.Warning(f"{os.path.basename(file_path)} 처리 실패: {results['error']}")
                continue
            all_results_data.append(results)
            gallery_images.append(results.get("visualization"))

        if not all_results_data:
            gr.Info("모든 이미지 처리에 실패했습니다.")
            return [None] * 9 + [[]]

        first_res = all_results_data[0]
        ref_gallery, _ = self._get_top5_gallery_data(first_res)

        # 결과 정보에 스파스 매칭 정보 추가
        sparse_info = ""
        if first_res.get("sparse_match_visualization") is not None:
            sparse_info = f"\n✅ 스파스 매칭 시각화 생성됨"
            if first_res.get("img1_points") is not None:
                sparse_info += f"\n✅ 메모리 이미지 특징점 분석 완료"
            if first_res.get("img2_points") is not None:
                sparse_info += f"\n✅ 현재 이미지 특징점 분석 완료"
        else:
            sparse_info = "\n❌ 스파스 매칭 시각화 생성 실패"
        
        result_info_text = f"총 {len(all_results_data)}개 처리됨. 첫 결과 표시.{sparse_info}"
        
        return (
            gallery_images, first_res.get("visualization"), first_res.get("image"), 
            first_res.get("mask"), result_info_text,
            ref_gallery, first_res.get("sparse_match_visualization"),
            first_res.get("img1_points"), first_res.get("img2_points"), all_results_data 
        )
    
    def browse_folder(self):
        """폴더 찾기 함수 - 시스템 파일 브라우저 열기"""
        try:
            import tkinter as tk
            from tkinter import filedialog
            
            # Tkinter 루트 윈도우 생성 (숨김)
            root = tk.Tk()
            root.withdraw()  # 윈도우 숨기기
            
            # 폴더 선택 다이얼로그 열기
            folder_path = filedialog.askdirectory(
                title="처리할 이미지가 있는 폴더를 선택하세요",
                initialdir=os.path.expanduser("~")  # 홈 디렉토리에서 시작
            )
            
            root.destroy()  # Tkinter 루트 윈도우 정리
            
            if folder_path:
                return folder_path
            else:
                return ""
                
        except Exception as e:
            print(f"폴더 찾기 오류: {e}")
            return ""
    
    def process_folder_and_update_ui(self, folder_path, match_bg, use_kmeans, kmeans_k, progress=gr.Progress()):
        """폴더 내 모든 이미지 처리"""
        if not folder_path:
            gr.Info("폴더 경로를 입력하거나 폴더 찾기 버튼을 클릭해주세요.")
            return [None] * 9 + [[]]
        
        if not os.path.exists(folder_path):
            gr.Info("입력된 폴더가 존재하지 않습니다.")
            return [None] * 9 + [[]]
        
        # 폴더 내 이미지 파일 찾기
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        try:
            for file in os.listdir(folder_path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(folder_path, file))
        except Exception as e:
            gr.Warning(f"폴더 읽기 실패: {e}")
            return [None] * 9 + [[]]
        
        if not image_files:
            gr.Info("폴더에 이미지 파일이 없습니다.")
            return [None] * 9 + [[]]
        
        folder_name = os.path.basename(folder_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_result_dir = Path(self.results_dir) / f"{folder_name}_{timestamp}"
        main_result_dir.mkdir(exist_ok=True, parents=True)
        
        gr.Info(f"폴더 '{folder_name}'에서 {len(image_files)}개의 이미지를 찾았습니다.")
        print(f"[DEBUG] 폴더 처리 시작: {folder_path}")
        print(f"[DEBUG] 폴더명: {folder_name}")
        print(f"[DEBUG] 타임스탬프: {timestamp}")
        print(f"[DEBUG] 발견된 이미지: {len(image_files)}개")
        print(f"[DEBUG] 메인 결과 폴더 생성: {main_result_dir}")
        for i, img_path in enumerate(image_files[:5]):  # 처음 5개만 출력
            print(f"[DEBUG] {i+1}: {os.path.basename(img_path)}")
        if len(image_files) > 5:
            print(f"[DEBUG] ... 및 {len(image_files) - 5}개 더")
        
        all_results_data = []
        gallery_images = []

        progress(0, desc="폴더 이미지 처리 시작...")
        for i, file_path in enumerate(image_files):
            progress((i + 1) / len(image_files), desc=f"{os.path.basename(file_path)} 처리 중...")
            
            self.memory_sam.use_kmeans_fg = use_kmeans
            self.memory_sam.kmeans_fg_clusters = int(kmeans_k)
            self.memory_sam.skip_clustering = not use_kmeans

            results = self.memory_sam.process_image(
                image_path=file_path,
                match_background=match_bg,
            )
            if "error" in results:
                gr.Warning(f"{os.path.basename(file_path)} 처리 실패: {results['error']}")
                continue
            
            # 각 이미지 결과를 개별적으로 저장 (메인 폴더 내에)
            try:
                self._save_individual_results(results, main_result_dir, i)
                print(f"[DEBUG] {os.path.basename(file_path)} 결과 저장 완료")
            except Exception as e:
                print(f"[ERROR] {os.path.basename(file_path)} 결과 저장 실패: {e}")
            
            all_results_data.append(results)
            gallery_images.append(results.get("visualization"))

        if not all_results_data:
            gr.Info("모든 이미지 처리에 실패했습니다.")
            return [None] * 9 + [[]]

        first_res = all_results_data[0]
        ref_gallery, _ = self._get_top5_gallery_data(first_res)

        # 결과 정보에 폴더 처리 정보 추가
        sparse_info = ""
        if first_res.get("sparse_match_visualization") is not None:
            sparse_info = f"\n✅ 스파스 매칭 시각화 생성됨"
            if first_res.get("img1_points") is not None:
                sparse_info += f"\n✅ 메모리 이미지 특징점 분석 완료"
            if first_res.get("img2_points") is not None:
                sparse_info += f"\n✅ 현재 이미지 특징점 분석 완료"
        else:
            sparse_info = "\n❌ 스파스 매칭 시각화 생성 실패"
        
        result_info_text = f"폴더 '{folder_name}'에서 {len(all_results_data)}개 처리됨. 결과 저장 위치: {folder_name}_{timestamp}/{sparse_info}"
        
        return (
            gallery_images, first_res.get("visualization"), first_res.get("image"), 
            first_res.get("mask"), result_info_text,
            ref_gallery, first_res.get("sparse_match_visualization"),
            first_res.get("img1_points"), first_res.get("img2_points"), all_results_data 
        )
    
    def _save_individual_results(self, results, main_result_dir, index):
        """개별 이미지 결과를 저장하는 함수"""
        try:
            from PIL import Image
            import cv2
            
            # 이미지 파일명 추출
            image_filename = os.path.basename(results.get("image_path", f"image_{index}"))
            image_stem = os.path.splitext(image_filename)[0]
            
            # 각 이미지별 하위폴더 생성 (메인 폴더 내에)
            image_result_dir = main_result_dir / image_stem
            image_result_dir.mkdir(exist_ok=True, parents=True)
            
            # 1. 원본 이미지 저장
            if results.get("image") is not None:
                Image.fromarray(results["image"]).save(str(image_result_dir / "input.png"))
                print(f"[DEBUG] 원본 이미지 저장: {image_result_dir}/input.png")
            
            # 2. 마스크 저장
            if results.get("mask") is not None:
                mask_img = (results["mask"] * 255).astype(np.uint8)
                Image.fromarray(mask_img).save(str(image_result_dir / "mask.png"))
                print(f"[DEBUG] 마스크 저장: {image_result_dir}/mask.png")
            
            # 3. 시각화 결과 저장 (오버레이)
            if results.get("visualization") is not None:
                Image.fromarray(results["visualization"]).save(str(image_result_dir / "overlay.png"))
                print(f"[DEBUG] 시각화 결과 저장: {image_result_dir}/overlay.png")
            
            # 3-1. 세그멘테이션 결과 저장 (segment.png)
            if results.get("mask") is not None and results.get("image") is not None:
                # 마스크를 원본 이미지에 오버레이하여 segment.png 생성
                mask_img = results["mask"].astype(np.uint8) * 255
                segment_img = self._create_segment_image(results["image"], mask_img)
                Image.fromarray(segment_img).save(str(image_result_dir / "segment.png"))
                print(f"[DEBUG] 세그멘테이션 결과 저장: {image_result_dir}/segment.png")
            
            # 4. 스파스 매칭 시각화 저장
            if results.get("sparse_match_visualization") is not None:
                cv2.imwrite(str(image_result_dir / "sparse_matches.png"), 
                           cv2.cvtColor(results["sparse_match_visualization"], cv2.COLOR_RGB2BGR))
                print(f"[DEBUG] 스파스 매칭 저장: {image_result_dir}/sparse_matches.png")
            
            # 5. 개별 특징점 이미지 저장
            if results.get("img1_points") is not None:
                cv2.imwrite(str(image_result_dir / "img1_points.png"), 
                           cv2.cvtColor(results["img1_points"], cv2.COLOR_RGB2BGR))
                print(f"[DEBUG] 이미지1 특징점 저장: {image_result_dir}/img1_points.png")
            
            if results.get("img2_points") is not None:
                cv2.imwrite(str(image_result_dir / "img2_points.png"), 
                           cv2.cvtColor(results["img2_points"], cv2.COLOR_RGB2BGR))
                print(f"[DEBUG] 이미지2 특징점 저장: {image_result_dir}/img2_points.png")
            
            # 6. 메타데이터 저장
            metadata = {
                "image_path": results.get("image_path"),
                "score": results.get("score"),
                "timestamp": timestamp,
                "index": index
            }
            
            import json
            with open(str(image_result_dir / "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            print(f"[DEBUG] 메타데이터 저장: {image_result_dir}/metadata.json")
            
            print(f"[DEBUG] {image_stem} 모든 결과 저장 완료: {main_result_dir.name}/{image_stem}/")
            
        except Exception as e:
            print(f"[DEBUG] 결과 저장 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def _create_segment_image(self, image, mask):
        """세그멘테이션 결과 이미지 생성 (segment.png용)"""
        try:
            import cv2
            
            # 이미지와 마스크 크기 맞추기
            if image.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            # 마스크를 3채널로 확장 (이미지와 같은 채널 수)
            if len(mask.shape) == 2:
                mask_3ch = np.stack([mask, mask, mask], axis=2)
            else:
                mask_3ch = mask
            
            # 마스크 영역을 파란색으로 표시
            segment_img = image.copy()
            blue_color = np.array([255, 0, 0], dtype=np.uint8)  # BGR 형식
            
            # 마스크가 있는 영역을 파란색으로 채우기
            segment_img[mask_3ch[:, :, 0] > 0] = blue_color
            
            # 마스크 경계선을 흰색으로 표시
            kernel = np.ones((3, 3), np.uint8)
            mask_eroded = cv2.erode(mask, kernel, iterations=1)
            boundary = mask - mask_eroded
            
            if len(boundary.shape) == 2:
                boundary_3ch = np.stack([boundary, boundary, boundary], axis=2)
            else:
                boundary_3ch = boundary
            
            segment_img[boundary_3ch[:, :, 0] > 0] = [255, 255, 255]  # 흰색 경계선
            
            print(f"[DEBUG] 세그멘테이션 이미지 생성 완료: {segment_img.shape}")
            return segment_img
            
        except Exception as e:
            print(f"[ERROR] 세그멘테이션 이미지 생성 실패: {e}")
            # 오류 시 원본 이미지 반환
            return image

    def _get_top5_gallery_data(self, item_data):
        top5_gallery = []
        info = "메모리 매칭 없음"
        if "gallery_items" in item_data and item_data["gallery_items"]:
            for item in item_data["gallery_items"]:
                caption = f"ID: {item['id']}\nSim: {item['similarity']:.4f}"
                top5_gallery.append((item['image'], caption))
            info = f"Path: {item_data.get('image_path')}"
        return top5_gallery, info

    def handle_result_gallery_select(self, processed_data, evt: gr.SelectData):
        selected_item = processed_data[evt.index]
        top5_gallery, info = self._get_top5_gallery_data(selected_item)
        
        # 선택된 결과에 대한 상세 정보 생성
        selected_info = f"선택된 결과: {os.path.basename(selected_item.get('image_path', 'Unknown'))}"
        
        # 스파스 매칭 정보 추가
        if selected_item.get("sparse_match_visualization") is not None:
            selected_info += f"\n✅ 스파스 매칭 시각화 있음"
            if selected_item.get("img1_points") is not None:
                selected_info += f"\n✅ 메모리 이미지 특징점 분석 완료"
            if selected_item.get("img2_points") is not None:
                selected_info += f"\n✅ 현재 이미지 특징점 분석 완료"
        else:
            selected_info += f"\n❌ 스파스 매칭 시각화 없음"
        
        # 세그멘테이션 점수 정보 추가
        if "score" in selected_item:
            selected_info += f"\n🎯 세그멘테이션 점수: {selected_item['score']:.4f}"
        
        return (
            selected_item.get("visualization"), selected_item.get("image"), 
            selected_item.get("mask"), selected_info, top5_gallery,
            selected_item.get("sparse_match_visualization"),
            selected_item.get("img1_points"), selected_item.get("img2_points")
        )
    
    def _setup_mask_generator_tab(self):
        """마스크 생성 탭 설정"""
        with gr.TabItem("마스크 생성"):
            with gr.Row():
                with gr.Column(scale=1):
                    mask_creator_image = gr.Image(label="이미지-클릭하여 포인트 추가", type="numpy", interactive=True)
                    with gr.Row():
                        pos_point_btn = gr.Button("전경 포인트 (객체)", variant="primary")
                        neg_point_btn = gr.Button("배경 포인트", variant="secondary")
                        clear_points_btn = gr.Button("모든 포인트 지우기")
                    
                    gr.Markdown("### 3. 결과 저장")
                    save_to_memory_btn = gr.Button("메모리에 저장")
                    save_status_text = gr.Textbox(label="저장 결과", interactive=False)

                    status_msg = gr.Textbox(label="상태", value="포인트 타입: 전경 (객체)", interactive=False)
                
                with gr.Column(scale=1):
                    segmentation_result = gr.Image(label="세그멘테이션 결과", type="numpy")

            current_points_state = gr.State([])
            current_labels_state = gr.State([])
            current_mask_state = gr.State(None)

            pos_point_btn.click(lambda: "전경 (객체)", outputs=[]).then(
                lambda: self.set_point_type("전경 (객체)"), outputs=[status_msg]
            )
            neg_point_btn.click(lambda: "배경", outputs=[]).then(
                lambda: self.set_point_type("배경"), outputs=[status_msg]
            )

            mask_creator_image.select(
                fn=self.handle_mask_image_click,
                inputs=[mask_creator_image, current_points_state, current_labels_state],
                outputs=[segmentation_result, current_points_state, current_labels_state, current_mask_state]
            )
            
            def clear_points():
                self.current_points = []
                self.current_point_labels = []
                return None, [], [], None, "포인트를 모두 초기화했습니다."
            
            clear_points_btn.click(
                fn=clear_points,
                outputs=[segmentation_result, current_points_state, current_labels_state, current_mask_state, status_msg]
            )
            
            save_to_memory_btn.click(
                fn=self.save_mask_to_memory,
                inputs=[mask_creator_image, current_mask_state],
                outputs=[save_status_text]
            )
            
    def set_point_type(self, ptype):
        self.current_point_type = ptype
        return f"포인트 타입: {ptype}"
    
    def handle_mask_image_click(self, image, points, labels, evt: gr.SelectData):
        if image is None:
            return None, points, labels, None
        
        x, y = evt.index
        label = 1 if self.current_point_type == "전경 (객체)" else 0
        
        points.append([x, y])
        labels.append(label)

        predictor = self.memory_sam.sam_predictor
        predictor.set_image(image)
        
        points_np = np.array(points)
        labels_np = np.array(labels)
        points_tensor = torch.as_tensor([points_np], dtype=torch.float, device=self.memory_sam.device)
        labels_tensor = torch.as_tensor([labels_np], dtype=torch.int, device=self.memory_sam.device)

        masks, scores, _ = predictor.predict(
            point_coords=points_tensor,
            point_labels=labels_tensor,
            multimask_output=True,
        )
        
        scores_for_image = scores[0] if scores.ndim > 1 else scores
        scores_tensor = torch.from_numpy(np.atleast_1d(scores_for_image)).to(self.memory_sam.device)
        mask = masks[0, torch.argmax(scores_tensor)].cpu().numpy()
        
        result_img = draw_points_on_image(image, points, labels, mask)
        
        return result_img, points, labels, mask

    def save_mask_to_memory(self, image, mask):
        if image is None or mask is None:
            return "저장할 이미지나 마스크가 없습니다."

        try:
            # 1. Extract features
            patch_features, _, _ = self.memory_sam.feature_extractor.extract_patch_features(image)
            
            # 2. Add to memory
            memory_id = self.memory_sam.memory.add_memory(
                image=image,
                mask=mask,
                features=None, # Global features are not needed for sparse matching memory
                patch_features=patch_features,
            )
            
            success_msg = f"성공적으로 메모리에 저장되었습니다. (ID: {memory_id})"
            print(success_msg)
            return success_msg
        except Exception as e:
            error_msg = f"메모리 저장 중 오류 발생: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
    
    def _setup_memory_manager_tab(self, interface: gr.Blocks):
        """메모리 관리 탭 설정"""
        with gr.TabItem("메모리 관리"):
            with gr.Row():
                with gr.Column(scale=1):
                    memory_display = gr.Gallery(label="메모리 항목", columns=4, rows=2, height=400)
                    memory_stats = gr.Textbox(label="메모리 통계", interactive=False)
                    refresh_memory_btn = gr.Button(value="메모리 새로고침")
                
                with gr.Column(scale=1):
                    selected_memory_image = gr.Image(label="선택된 메모리 항목")
                    selected_memory_info = gr.JSON(label="항목 정보")
                    selected_item_id = gr.State(None)
                    delete_item_btn = gr.Button("선택 항목 삭제", variant="secondary")
                    item_delete_result = gr.Textbox(label="항목 삭제 결과", interactive=False)
            
            def load_memory_display():
                item_metas = self.memory_sam.memory.get_all_items()
                gallery = []
                for meta in item_metas:
                    try:
                        item_id = meta['id']
                        # For gallery, we only need the thumbnail, not all data.
                        # This assumes MemoryRepository has a way to get the image directly.
                        image_path = self.memory_sam.memory.repo.memory_dir / meta['image_path']
                        gallery.append((str(image_path), f"ID: {item_id}"))
                    except Exception as e:
                        print(f"Error loading memory item {meta.get('id', 'N/A')} for gallery: {e}")
                stats = f"총 {len(item_metas)}개의 항목이 메모리에 저장되어 있습니다."
                return gallery, stats

            def display_memory_item(evt: gr.SelectData):
                caption = evt.value
                item_id_str = caption.split(': ')[1]
                item_id = int(item_id_str)
                item_data = self.memory_sam.memory.load_item_data(item_id)
                # We show the full image here, and metadata.
                return item_data.get('image'), item_data, item_id

            def delete_memory_item(item_id):
                if item_id is None:
                    return "삭제할 항목을 먼저 선택하세요."
                try:
                    self.memory_sam.memory.delete_memory(item_id)
                    success_msg = f"항목 ID {item_id}가 삭제되었습니다."
                    print(success_msg)
                    return success_msg
                except Exception as e:
                    error_msg = f"항목 삭제 중 오류 발생: {e}"
                    print(error_msg)
                    return error_msg

            refresh_memory_btn.click(
                fn=load_memory_display,
                outputs=[memory_display, memory_stats]
            )

            memory_display.select(
                fn=display_memory_item,
                outputs=[selected_memory_image, selected_memory_info, selected_item_id]
            )

            delete_item_btn.click(
                fn=delete_memory_item,
                inputs=[selected_item_id],
                outputs=[item_delete_result]
            ).then(
                fn=load_memory_display,
                outputs=[memory_display, memory_stats]
            )
            
            # Tab is created, load initial data
            interface.load(fn=load_memory_display, outputs=[memory_display, memory_stats])