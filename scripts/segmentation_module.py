import os
import numpy as np
import torch
import gradio as gr
import cv2
from PIL import Image
from typing import List, Dict, Tuple, Any, Optional, Union
from pathlib import Path
from datetime import datetime

from scripts.memory_sam_predictor import MemorySAMPredictor
from scripts.memory_ui_utils import visualize_mask, prepare_input_data, SparseMatchVisualizer

class SegmentationModule:
    """Memory-based segmentation processing module"""
    
    def __init__(self, memory_sam_predictor: MemorySAMPredictor):
        """
        Initialize segmentation module
        
        Args:
            memory_sam_predictor: Memory SAM predictor instance
        """
        self.memory_sam = memory_sam_predictor
        self.processed_images = []
        self.last_result = None
        self.sparse_visualizer = SparseMatchVisualizer(memory_sam_predictor)
        
    def process_image(self, 
                     files: Union[List[Any], str, Path], 
                     reference_path: str = None, 
                     prompt_type: str = "points",
                     use_sparse_matching: bool = True,
                     match_background: bool = True,
                     skip_clustering: bool = False,
                     auto_add_to_memory: bool = False) -> Tuple:
        """
        Process image or folder
        
        Args:
            files: File upload or file path
            reference_path: Reference image path
            prompt_type: Prompt type ('points' or 'box')
            use_sparse_matching: Whether to use sparse matching
            match_background: Whether to match background area
            skip_clustering: Whether to skip clustering
            auto_add_to_memory: Whether to automatically add to memory after processing
            
        Returns:
            Processing result tuple (visualized image, mask, gallery items, info, etc.)
        """
        if not files:
            return None, None, [], "Please select an image.", None, None, None, None, "Please select an image.", None, None, None
        
        # 메모리 시스템에 클러스터링 설정 저장
        self.memory_sam.skip_clustering = skip_clustering
        hybrid_clustering = getattr(self.memory_sam, 'hybrid_clustering', False)
        
        try:
            # Prepare input data
            image_path = prepare_input_data(files)
            if image_path is None:
                return None, None, [], "No valid input.", None, None, None, None, "No valid input.", None, None, None
            
            # Process with segmentation
            result = self.memory_sam.process_image(
                image_path=image_path,
                reference_path=reference_path,
                prompt_type=prompt_type,
                use_sparse_matching=use_sparse_matching,
                match_background=match_background,
                skip_clustering=skip_clustering,
                auto_add_to_memory=auto_add_to_memory
            )
            
            # --- 결과 처리 로직 재구성 ---
            
            # 갤러리 및 처리된 이미지 목록 초기화
            gallery_items = []
            result_gallery_items = []
            processed_images_list = []
            
            # 반환된 결과가 폴더 처리 결과인지 확인
            is_folder = result.get("is_folder", False)
            results_list = result.get("results_list") if is_folder else [result]

            # 대표 결과 설정 (첫 번째 이미지의 결과 사용)
            if not results_list:
                 raise ValueError("Processing returned no results.")
            
            representative_result = results_list[0]
            
            # Process basic results for the representative image
            mask_vis = (representative_result["mask"] * 255).astype(np.uint8)
            # 마스크/이미지 크기 불일치를 자동 보정하는 예측기 메서드 사용
            seg_vis = self.memory_sam.visualize_mask(representative_result["image"], representative_result["mask"])
            
            # Prepare memory gallery items from the representative result
            if representative_result.get("similar_items"):
                for item_data in representative_result["similar_items"]:
                    item = item_data["item"]
                    similarity = item_data["similarity"]
                    img_path = self.memory_sam.memory.memory_dir / item["image_path"]
                    gallery_items.append((str(img_path), f"ID: {item['id']}, Sim: {similarity:.4f}"))
            
            # 정보 텍스트 생성 (폴더 또는 단일 이미지에 따라 다르게)
            if is_folder:
                memory_info = f"Processed {result.get('image_count', 0)} images in folder.\n"
                memory_info += f"Showing memory results for the first image: {Path(representative_result.get('image_path')).name}\n"
            else:
                memory_info = ""

            if representative_result.get("similar_items"):
                memory_info += f"Found {len(representative_result['similar_items'])} similar items.\n"
            else:
                memory_info += "No similar items found in memory. Using default prompt.\n"
            memory_info += f"Seg. score: {representative_result['score']:.4f}\n"

            # --- 결과 갤러리 및 processed_images_list 채우기 ---
            
            # 결과 저장 기본 경로 설정
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = Path(self.memory_sam.results_dir) / f"result_{timestamp}"
            base_path.mkdir(exist_ok=True, parents=True)
            
            for res in results_list:
                try:
                    img_path = res.get("image_path")
                    img_filename = Path(img_path).stem if img_path else "unknown"
                    
                    # 이미지 데이터 가져오기
                    original_img = res.get("original_image", res["image"])
                    processed_img = res["image"] # 리사이즈된 이미지
                    mask = res["mask"]

                    # 시각화 및 마스크 이미지 생성 (저장용)
                    mask_to_save = (mask * 255).astype(np.uint8)
                    # 오버레이는 원본 기준, 크기 자동 보정 함수 사용
                    overlay_img = self.memory_sam.visualize_mask(original_img, mask)
                    
                    # 파일 저장 경로 설정
                    img_dir = base_path / img_filename
                    img_dir.mkdir(exist_ok=True)
                    
                    input_save_path = img_dir / "input.png"
                    mask_save_path = img_dir / "mask.png"
                    overlay_save_path = img_dir / "overlay.png"
                    
                    # 파일 저장
                    Image.fromarray(original_img).save(str(input_save_path))
                    Image.fromarray(mask_to_save).save(str(mask_save_path))
                    Image.fromarray(overlay_img).save(str(overlay_save_path))

                    # 결과 갤러리에 추가
                    result_gallery_items.append((str(overlay_save_path), f"{img_filename} (Score: {res['score']:.2f})"))
                    
                    # 처리된 이미지 정보 목록에 추가
                    processed_images_list.append({
                        "filename": img_filename,
                        "path": img_path,
                        "input": str(input_save_path),
                        "mask": str(mask_save_path),
                        "overlay": str(overlay_save_path),
                        "score": float(res['score']),
                        "width": original_img.shape[1],
                        "height": original_img.shape[0],
                        "resize_scale": res.get("actual_resize_scale", 1.0),
                        "processing_time": 0.0, # Placeholder
                        "skip_clustering": skip_clustering,
                        "hybrid_clustering": hybrid_clustering
                    })
                except Exception as e:
                    print(f"Error processing and saving result for {res.get('image_path')}: {e}")
            
            result_info = f"Processed {len(processed_images_list)} image(s).\n"
            result_info += f"Results saved in: {base_path}"

            # Set default selection to first item
            selected_original, selected_mask, selected_overlay = (None, None, None)
            if processed_images_list:
                first_item = processed_images_list[0]
                selected_original = first_item["input"]
                selected_mask = first_item["mask"]
                selected_overlay = first_item["overlay"]
            
            # Store processed images list
            self.processed_images = processed_images_list
            
            # Sparse matching visualization for the representative result
            sparse_match_vis = representative_result.get("sparse_match_visualization")
            img1_points = representative_result.get("img1_points")
            img2_points = representative_result.get("img2_points")
            
            # 마지막 결과 저장
            self.last_result = {
                "image": representative_result["image"],
                "mask": representative_result["mask"],
                "sparse_match_visualization": sparse_match_vis,
                "img1_points": img1_points,
                "img2_points": img2_points,
                "skip_clustering": skip_clustering,
                "hybrid_clustering": hybrid_clustering
            }
            
            return (seg_vis, mask_vis, gallery_items, memory_info, 
                   result_gallery_items, selected_original, selected_mask, selected_overlay, result_info,
                   sparse_match_vis, img1_points, img2_points)
            
        except Exception as e:
            import traceback
            error_msg = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return None, None, [], f"Error: {str(e)}", None, None, None, None, f"Error: {str(e)}", None, None, None
    
    def handle_result_gallery_select(self, evt, processed_images=None):
        """
        Handle result gallery selection
        
        Args:
            evt: Selection event
            processed_images: List of processed images
            
        Returns:
            (original image, mask, overlay, info) tuple
        """
        try:
            if processed_images is None:
                processed_images = self.processed_images
                
            if not processed_images or evt is None:
                return None, None, None, ""
            
            # Get clicked index
            idx = evt.index if hasattr(evt, 'index') else 0
            
            # If there are processed images
            if isinstance(processed_images, list) and len(processed_images) > 0:
                if isinstance(processed_images[0], dict) and idx < len(processed_images):
                    selected_item = processed_images[idx]
                    
                    # 클러스터링 설정 복원 (갤러리 선택 시 해당 이미지의 원래 클러스터링 설정 사용)
                    if 'skip_clustering' in selected_item:
                        self.memory_sam.skip_clustering = selected_item['skip_clustering']
                    
                    if 'hybrid_clustering' in selected_item:
                        self.memory_sam.hybrid_clustering = selected_item['hybrid_clustering']
                    
                    clustering_info = ""
                    clustering_info += f"\n클러스터링: {'건너뛰기' if selected_item.get('skip_clustering', False) else '적용'}"
                    clustering_info += f"\n하이브리드 클러스터링: {'적용' if selected_item.get('hybrid_clustering', False) else '미적용'}"
                    
                    return (
                        selected_item["input"], 
                        selected_item["mask"], 
                        selected_item["overlay"],
                        f"Selected image: {selected_item['filename']}\nScore: {selected_item['score']:.4f}{clustering_info}"
                    )
            
            return None, None, None, "No valid selection"
            
        except Exception as e:
            print(f"Error in result gallery selection handler: {e}")
            return None, None, None, f"Error: {str(e)}"
    
    def save_to_memory(self) -> str:
        """
        Save current result to memory
        
        Returns:
            Status message
        """
        try:
            if not hasattr(self.memory_sam, 'current_image') or self.memory_sam.current_image is None:
                return "No image to save."
                
            if not hasattr(self.memory_sam, 'current_mask') or self.memory_sam.current_mask is None:
                return "No mask to save."
            
            # Add to memory system
            image = self.memory_sam.current_image
            mask = self.memory_sam.current_mask
            features = self.memory_sam.current_features
            
            # Check if patch features are available
            patch_features = getattr(self.memory_sam, 'current_patch_features', None)
            grid_size = getattr(self.memory_sam, 'current_grid_size', None)
            resize_scale = getattr(self.memory_sam, 'current_resize_scale', None)
            
            memory_id = self.memory_sam.memory.add_memory(
                image, mask, features, patch_features, grid_size, resize_scale,
                metadata={"original_path": self.memory_sam.current_image_path}
            )
            
            return f"Saved to memory with ID: {memory_id}"
            
        except Exception as e:
            return f"Error saving to memory: {str(e)}"