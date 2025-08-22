import os
import gradio as gr
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Callable

class UIComponents:
    """Gradio UI 구성요소를 캡슐화하는 클래스"""
    
    @staticmethod
    def create_resize_buttons() -> gr.Radio:
        """이미지 리사이징 옵션을 위한 라디오 버튼 생성"""
        return gr.Radio(
            choices=["원본 이미지", "512x512 고정"],
            value="원본 이미지",
            label="이미지 크기 모드",
            info="원본 이미지 또는 512x512 크기로 처리합니다. 512x512 모드가 더 빠르지만 정확도가 다를 수 있습니다."
        )
    
    @staticmethod
    def create_folder_browser(callback: Callable) -> Tuple[gr.Button, gr.Textbox]:
        """폴더 브라우저 버튼과 텍스트박스 생성"""
        folder_btn = gr.Button("폴더 찾아보기", variant="secondary")
        folder_path = gr.Textbox(
            label="폴더 경로",
            placeholder="/path/to/folder",
            info="이미지 파일을 포함한 폴더 경로"
        )
        
        # 이벤트 연결
        folder_btn.click(fn=callback, outputs=[folder_path])
        
        return folder_btn, folder_path
    
    @staticmethod
    def create_image_browser() -> gr.Image:
        """이미지 브라우저 생성"""
        return gr.Image(
            label="입력 이미지",
            type="numpy",
            height=400
        )
    
    @staticmethod
    def create_clustering_controls() -> Tuple[gr.Slider, gr.Slider, gr.Checkbox, gr.Checkbox, gr.Slider, gr.Slider]:
        """클러스터링 하이퍼파라미터 컨트롤 생성"""
        # 유사도 임계값 슬라이더
        similarity_threshold = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.8,
            step=0.05,
            label="유사도 임계값",
            info="높을수록 더 엄격한 매칭이 이루어집니다"
        )
        
        # 배경 영역 매칭 비율 슬라이더
        background_weight = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.3,
            step=0.05,
            label="배경 영역 가중치",
            info="배경 영역 매칭의 영향력"
        )
        
        # 클러스터링 건너뛰기 체크박스
        skip_clustering = gr.Checkbox(
            label="클러스터링 건너뛰기",
            value=False,
            info="활성화 하면 모든 포인트를 이용해 마스크를 생성합니다"
        )
        
        # 하이브리드 클러스터링 체크박스
        hybrid_clustering = gr.Checkbox(
            label="하이브리드 클러스터링",
            value=False,
            info="전경과 배경 점을 함께 클러스터링하여 같은 영역에 있는 점들을 필터링합니다"
        )
        
        # 최대 positive 포인트 슬라이더 추가
        max_positive_points = gr.Slider(
            minimum=1,
            maximum=20,
            value=10,
            step=1,
            label="최대 전경(positive) 포인트 수",
            info="클러스터링 사용 시 적용되는 최대 전경 포인트 수"
        )
        
        # 최대 negative 포인트 슬라이더 추가
        max_negative_points = gr.Slider(
            minimum=1,
            maximum=20,
            value=5,
            step=1,
            label="최대 배경(negative) 포인트 수",
            info="클러스터링 사용 시 적용되는 최대 배경 포인트 수"
        )
        
        # 전경 KMeans 중심 사용 옵션
        use_positive_kmeans = gr.Checkbox(
            label="전경 KMeans 중심 사용",
            value=False,
            info="전경 포인트 후보에서 특징 기반 KMeans 중심 n개만 사용"
        )
        positive_kmeans_clusters = gr.Slider(
            minimum=1,
            maximum=20,
            value=5,
            step=1,
            label="전경 KMeans 중심 수"
        )

        return similarity_threshold, background_weight, skip_clustering, hybrid_clustering, max_positive_points, max_negative_points, use_positive_kmeans, positive_kmeans_clusters
    
    @staticmethod
    def create_progress_bar() -> gr.Progress:
        """진행 상황 표시 바 생성"""
        return gr.Progress(track_tqdm=True)
    
    @staticmethod
    def create_gallery(label: str = "결과 갤러리", columns: int = 3) -> gr.Gallery:
        """결과 갤러리 생성"""
        return gr.Gallery(
            label=label,
            columns=columns,
            rows=2,
            height=400,
            object_fit="contain"
        )
    
    @staticmethod
    def create_result_display() -> Tuple[gr.Tabs, gr.Image, gr.Image, gr.Image]:
        """결과 표시를 위한 탭 구성 요소 생성"""
        with gr.Tabs() as tabs:
            with gr.Tab("원본"):
                original_img = gr.Image(label="원본 이미지", type="numpy")
            with gr.Tab("마스크"):
                mask_img = gr.Image(label="마스크", type="numpy")
            with gr.Tab("오버레이"):
                overlay_img = gr.Image(label="오버레이", type="numpy")
        
        return tabs, original_img, mask_img, overlay_img