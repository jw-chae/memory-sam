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
            
            # Process basic results
            mask_vis = (result["mask"] * 255).astype(np.uint8)
            seg_vis = visualize_mask(result["image"], result["mask"])
            
            # Prepare memory gallery items
            gallery_items = []
            if result["similar_items"]:
                for item_data in result["similar_items"]:
                    item = item_data["item"]
                    similarity = item_data["similarity"]
                    img_path = self.memory_sam.memory.memory_dir / item["image_path"]
                    gallery_items.append((str(img_path), f"ID: {item['id']}, Similarity: {similarity:.4f}"))
            
            # Prepare memory info text
            if result["similar_items"]:
                memory_info = f"Found {len(result['similar_items'])} similar items in memory.\n"
                for i, item_data in enumerate(result["similar_items"]):
                    memory_info += f"Item {i+1}: ID {item_data['item']['id']}, Similarity: {item_data['similarity']:.4f}\n"
                    
                    # Foreground/background matching info (if available)
                    if "foreground" in item_data:
                        fg = item_data["foreground"]
                        memory_info += f"   Foreground - Similarity: {fg['similarity']:.4f}, Match ratio: {fg['match_ratio']:.4f}, Mean distance: {fg['mean_distance']:.4f}\n"
                        
                        if "background" in item_data and item_data["background"] is not None:
                            bg = item_data["background"]
                            memory_info += f"   Background - Similarity: {bg['similarity']:.4f}, Match ratio: {bg['match_ratio']:.4f}, Mean distance: {bg['mean_distance']:.4f}\n"
                    # Previous version compatibility
                    elif "match_ratio" in item_data:
                        memory_info += f"   Match ratio: {item_data['match_ratio']:.4f}, Mean distance: {item_data['mean_distance']:.4f}\n"
            else:
                memory_info = "No similar items found in memory. Using default prompt."
            
            memory_info += f"\nSegmentation score: {result['score']:.4f}\n"
            memory_info += f"Result path: {result['result_path']}"
            
            # Prepare result gallery items
            result_gallery_items = []
            
            # Clean and ensure processed_images list exists
            processed_images_list = []
            
            # For folder processing
            if result.get("is_folder", False):
                # Check if folder image paths are available
                if hasattr(self.memory_sam, 'folder_image_paths') and self.memory_sam.folder_image_paths:
                    folder_paths = self.memory_sam.folder_image_paths
                    print(f"Folder processing: {len(folder_paths)} images found")
                    
                    # Take first image from folder as representative
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    base_path = Path(self.memory_sam.results_dir) / f"result_{timestamp}" / "all_images"
                    base_path.mkdir(exist_ok=True, parents=True)
                    
                    for i, img_path in enumerate(folder_paths):
                        try:
                            # Process each image
                            img_result = self.memory_sam.process_image(
                                image_path=img_path,
                                reference_path=reference_path,
                                prompt_type=prompt_type,
                                use_sparse_matching=use_sparse_matching,
                                match_background=match_background,
                                skip_clustering=skip_clustering
                            )
                            
                            # Get image filename without extension
                            img_filename = Path(img_path).stem
                            
                            # Convert to images
                            input_img = img_result["image"]
                            mask_img = (img_result["mask"] * 255).astype(np.uint8)
                            overlay_img = visualize_mask(input_img, img_result["mask"])
                            
                            # Save images to disk
                            input_save_path = base_path / f"{img_filename}_input.png"
                            mask_save_path = base_path / f"{img_filename}_mask.png"
                            overlay_save_path = base_path / f"{img_filename}_overlay.png"
                            
                            Image.fromarray(input_img).save(str(input_save_path))
                            Image.fromarray(mask_img).save(str(mask_save_path))
                            Image.fromarray(overlay_img).save(str(overlay_save_path))
                            
                            # Add to gallery
                            result_gallery_items.append((str(overlay_save_path), f"{img_filename} (Score: {img_result['score']:.2f})"))
                            
                            # Add to processed images list with clustering settings
                            processed_images_list.append({
                                "filename": img_filename,
                                "path": img_path,
                                "input": str(input_save_path),
                                "mask": str(mask_save_path),
                                "overlay": str(overlay_save_path),
                                "score": float(img_result['score']),
                                "width": input_img.shape[1],
                                "height": input_img.shape[0],
                                "resize_scale": 1.0,
                                "processing_time": 0.0,
                                "skip_clustering": skip_clustering,
                                "hybrid_clustering": hybrid_clustering
                            })
                            
                        except Exception as e:
                            print(f"Error processing image {img_path}: {e}")
                
                result_info = f"Processed {len(processed_images_list)} images.\n"
                result_info += f"Result path: {base_path}"
                
            else:
                # Single file processing
                print("Single image processing result")
                
                # Save the files to disk if they don't exist
                overlay_path = Path(str(result['result_path'])) / "overlay.png"
                input_path = Path(str(result['result_path'])) / "input.png"
                mask_path = Path(str(result['result_path'])) / "mask.png"
                
                if not overlay_path.exists():
                    Image.fromarray(seg_vis).save(str(overlay_path))
                
                if not input_path.exists():
                    Image.fromarray(result["image"]).save(str(input_path))
                
                if not mask_path.exists():
                    Image.fromarray(mask_vis).save(str(mask_path))
                
                # Add to gallery
                result_gallery_items = [
                    (str(input_path), "Input image"),
                    (str(mask_path), "Mask"),
                    (str(overlay_path), "Overlay")
                ]
                
                # Add to processed images list with clustering settings
                processed_images_list.append({
                    "filename": Path(image_path).stem if isinstance(image_path, (str, Path)) else "unknown",
                    "path": str(image_path) if isinstance(image_path, (str, Path)) else "unknown",
                    "input": str(input_path),
                    "mask": str(mask_path),
                    "overlay": str(overlay_path),
                    "score": float(result['score']),
                    "width": result["image"].shape[1],
                    "height": result["image"].shape[0],
                    "resize_scale": 1.0,
                    "processing_time": 0.0,
                    "skip_clustering": skip_clustering,
                    "hybrid_clustering": hybrid_clustering
                })
                
                result_info = f"Processed single image.\n"
                result_info += f"Segmentation score: {result['score']:.4f}\n"
                result_info += f"Result path: {result['result_path']}"
            
            # Set default selection to first item
            selected_original = None
            selected_mask = None
            selected_overlay = None
            
            if processed_images_list:
                first_item = processed_images_list[0]
                selected_original = first_item["input"]
                selected_mask = first_item["mask"]
                selected_overlay = first_item["overlay"]
            
            # Store processed images list
            self.processed_images = processed_images_list
            
            # Sparse matching visualization
            sparse_match_vis = None
            img1_points = None
            img2_points = None
            
            # 스파스 매칭 시각화 생성 (일관된 설정으로)
            if hasattr(self.memory_sam, 'similar_items') and self.memory_sam.similar_items:
                try:
                    # 베스트 매칭 메모리 항목 찾기
                    best_item = self.memory_sam.similar_items[0]["item"]
                    item_data = self.memory_sam.memory.load_item_data(best_item["id"])
                    
                    if "image" in item_data and "mask" in item_data:
                        memory_image = item_data["image"]
                        memory_mask = item_data["mask"]
                        
                        # 현재 클러스터링 설정으로 시각화 생성
                        sparse_match_vis, img1_points, img2_points = self.sparse_visualizer.visualize_matches(
                            memory_image, 
                            result["image"], 
                            memory_mask, 
                            result["mask"],
                            skip_clustering=skip_clustering,
                            hybrid_clustering=hybrid_clustering,
                            save_path=str(Path(result["result_path"]) / f"sparse_match_{skip_clustering}_{hybrid_clustering}.png")
                        )
                        
                        print(f"스파스 매칭 시각화 생성 완료: skip={skip_clustering}, hybrid={hybrid_clustering}")
                        
                        # 결과에 추가
                        result["sparse_match_visualization"] = sparse_match_vis
                        result["img1_points"] = img1_points
                        result["img2_points"] = img2_points
                        
                except Exception as e:
                    print(f"스파스 매칭 시각화 생성 중 오류: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # 직접 결과에서 가져오기
                sparse_match_vis = result.get("sparse_match_visualization", None)
                img1_points = result.get("img1_points", None)
                img2_points = result.get("img2_points", None)
            
            # 처리된 이미지에 스파스 매칭 시각화 추가
            if sparse_match_vis is not None and processed_images_list:
                for img_data in processed_images_list:
                    img_data["sparse_match_visualization"] = sparse_match_vis
                    img_data["img1_points"] = img1_points
                    img_data["img2_points"] = img2_points
            
            # 마지막 결과 저장
            self.last_result = {
                "image": result["image"],
                "mask": result["mask"],
                "sparse_match_visualization": sparse_match_vis,
                "img1_points": img1_points,
                "img2_points": img2_points,
                "skip_clustering": skip_clustering,
                "hybrid_clustering": hybrid_clustering
            }
            
            # Add sparse matching info if available
            if sparse_match_vis is not None:
                memory_info += "\n\nSparse matching visualization created."
                memory_info += f"\nResult path: {result['result_path']}/sparse_matches.png"
                memory_info += f"\nClustering setting: {'Skip clustering (all points)' if skip_clustering else 'Hybrid clustering' if hybrid_clustering else 'Standard clustering'}"
            
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