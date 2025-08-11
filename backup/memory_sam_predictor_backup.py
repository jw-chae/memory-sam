import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple, Union, Dict, Any
from pathlib import Path
from datetime import datetime

# Import DINOv2 for feature extraction
from transformers import AutoImageProcessor, AutoModel

# SAM2 모듈 임포트 - 다양한 경로 시도
possible_paths = [
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # 현재 스크립트의 상위 디렉토리
    '/home/joongwon00/sam2',  # sam2 원본 저장소 경로
    '/home/joongwon00/Memory_SAM',  # Memory_SAM 프로젝트 루트
]

for path in possible_paths:
    if path not in sys.path:
        sys.path.append(path)

# 메인 SAM2 패키지 경로도 설정
sam2_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sam2")
if sam2_path not in sys.path:
    sys.path.insert(0, sam2_path)

# PYTHONPATH 환경 변수 설정
os.environ["PYTHONPATH"] = f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}:/home/joongwon00/sam2"

print(f"Python 경로: {sys.path}")

# Hydra 모듈 초기화 확인
try:
    from hydra.core.global_hydra import GlobalHydra
    from hydra import initialize_config_module
    
    # Hydra가 초기화되지 않았으면 초기화
    if not GlobalHydra.instance().is_initialized():
        print("Hydra 초기화 중...")
        initialize_config_module("sam2", version_base="1.2")
        print("Hydra 초기화 완료")
except Exception as e:
    print(f"Hydra 초기화 확인 중 오류: {e}")

# Import SAM2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2

# Import memory system
from scripts.memory_system import MemorySystem

class MemorySAMPredictor:
    """SAM2 with DINOv2 and a memory system for intelligent segmentation"""
    
    def __init__(self, 
                model_type: str = "hiera_l", 
                checkpoint_path: str = None,
                dinov2_model: str = "facebook/dinov2-base",
                memory_dir: str = "memory",
                results_dir: str = "results",
                device: str = "cuda"):
        """
        Memory SAM 예측기 초기화
        
        Args:
            model_type: 사용할 SAM2 모델 타입 ("hiera_b+", "hiera_l", "hiera_s", "hiera_t")
            checkpoint_path: SAM2 체크포인트 경로
            dinov2_model: DINOv2 모델 이름
            memory_dir: 메모리 시스템 디렉토리
            results_dir: 결과 저장 디렉토리
            device: 사용할 디바이스 ("cuda" 또는 "cpu")
        """
        # Set device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
            # TF32 활성화 (Ampere GPU 이상)
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            self.device = torch.device("cpu")
        
        print(f"사용 디바이스: {self.device}")
        
        # 체크포인트 경로 찾기
        if checkpoint_path is None:
            # 모델 타입에 따른 매핑
            model_name_map = {
                "hiera_b+": ["base_plus", "b+"],
                "hiera_l": ["large", "l"],
                "hiera_s": ["small", "s"],
                "hiera_t": ["tiny", "t"]
            }
            
            # 체크포인트 파일 이름 패턴 목록
            checkpoint_patterns = [
                f"sam2_{model_type}.pt",
                f"sam2.1_{model_type}.pt"
            ]
            
            # 추가 패턴 생성
            for variant in model_name_map[model_type]:
                checkpoint_patterns.extend([
                    f"sam2_hiera_{variant}.pt",
                    f"sam2.1_hiera_{variant}.pt"
                ])
            
            # 체크포인트 디렉토리
            checkpoint_dir = os.path.join('/home/joongwon00/sam2', 'checkpoints')
            
            # 패턴에 맞는 체크포인트 파일 찾기
            for pattern in checkpoint_patterns:
                path = os.path.join(checkpoint_dir, pattern)
                if os.path.exists(path):
                    checkpoint_path = path
                    break
            
            if checkpoint_path is None:
                raise FileNotFoundError(f"체크포인트 파일을 찾을 수 없습니다. 시도한 패턴: {checkpoint_patterns}")
        
        print(f"체크포인트 경로: {checkpoint_path}")
        
        # SAM2 프로젝트 루트 경로
        sam2_root = '/home/joongwon00/sam2'
        
        # 가능한 Hydra 설정 파일 이름
        possible_config_names = [
            # SAM2.1 설정
            f"configs/sam2.1/sam2.1_hiera_{model_type.replace('hiera_', '')}",
            # SAM2 원본 설정
            f"configs/sam2/sam2_hiera_{model_type.replace('hiera_', '')}",
            # 절대 경로
            f"/home/joongwon00/sam2/configs/sam2.1/sam2.1_hiera_{model_type.replace('hiera_', '')}.yaml",
            f"/home/joongwon00/sam2/configs/sam2/sam2_hiera_{model_type.replace('hiera_', '')}.yaml"
        ]
        
        # 각 이름 시도
        config_file = possible_config_names[0]  # 기본값
        
        print(f"시도할 Hydra 설정 파일 목록: {possible_config_names}")
        
        # Initialize SAM2
        all_failed = True
        for i, cfg in enumerate(possible_config_names):
            try:
                print(f"[{i+1}/{len(possible_config_names)}] {cfg} 시도 중...")
                self.sam_model = build_sam2(cfg, checkpoint_path, device=self.device)
                config_file = cfg  # 성공한 설정 기록
                print(f"성공: {cfg}")
                all_failed = False
                break
            except Exception as e:
                print(f"실패: {cfg} - {e}")
        
        if all_failed:
            print(f"모든 설정 파일 시도 실패: {possible_config_names}")
            print(f"Hydra 설정 파일 문제일 수 있습니다. 대체 방법 시도 중...")
            
            # 절대 경로를 직접 찾은 후 환경 변수를 통해 Hydra에게 알림
            config_paths = [
                # SAM2 디렉토리의 configs/sam2.1 경로
                os.path.join('/home/joongwon00/sam2/configs/sam2.1', f'sam2.1_hiera_{model_type.replace("hiera_", "")}.yaml'),
                # SAM2 디렉토리의 configs/sam2 경로
                os.path.join('/home/joongwon00/sam2/configs/sam2', f'sam2_hiera_{model_type.replace("hiera_", "")}.yaml'),
                # SAM2 디렉토리의 sam2/configs/sam2.1 경로
                os.path.join('/home/joongwon00/sam2/sam2/configs/sam2.1', f'sam2.1_hiera_{model_type.replace("hiera_", "")}.yaml'),
                # SAM2 디렉토리의 sam2/configs/sam2 경로
                os.path.join('/home/joongwon00/sam2/sam2/configs/sam2', f'sam2_hiera_{model_type.replace("hiera_", "")}.yaml')
            ]
            
            # 존재하는 설정 파일 찾기
            config_path = None
            for path in config_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다. 시도한 경로: {config_paths}")
            
            print(f"설정 파일 경로: {config_path}")
            
            # config_path를 직접 지정하여 build_sam2 호출 (이 방법은 작동하지 않을 수 있음)
            # 파일이 이미 존재하는 경우, 이를 복사하여 Hydra의 기본 경로에 놓음
            try:
                # 1. 설정 파일 내용 읽기
                with open(config_path, 'r') as f:
                    config_content = f.read()
                
                # 2. configs/sam2_hiera_l.yaml 파일 생성 (Hydra가 찾으려는 위치)
                hydra_config_path = os.path.join('/home/joongwon00/sam2/configs', f'sam2_hiera_{model_type}.yaml')
                os.makedirs(os.path.dirname(hydra_config_path), exist_ok=True)
                
                with open(hydra_config_path, 'w') as f:
                    f.write(config_content)
                
                print(f"설정 파일을 Hydra 기본 경로에 복사: {hydra_config_path}")
                
                # 3. 설정 파일을 직접 지정하지 않고 기본 Hydra 경로 사용 시도
                try:
                    self.sam_model = build_sam2(f"configs/sam2_hiera_{model_type}", checkpoint_path, device=self.device)
                    print("Hydra 기본 경로에서 설정 로드 성공")
                except Exception as e3:
                    print(f"Hydra 기본 경로 시도 실패: {e3}")
                    # 4. 직접 경로 지정 시도
                    try:
                        self.sam_model = build_sam2(config_path, checkpoint_path, device=self.device)
                        print(f"절대 경로에서 설정 로드 성공: {config_path}")
                    except Exception as e4:
                        print(f"모든 시도 실패: {e4}")
                        print("시스템 관리자에게 문의하거나 설정 파일 경로 구조를 확인하세요.")
                        raise e4
            except Exception as e2:
                print(f"설정 파일 처리 실패: {e2}")
                print("시스템 관리자에게 문의하거나 설정 파일 경로 구조를 확인하세요.")
                raise
        self.predictor = SAM2ImagePredictor(self.sam_model)
        
        # Initialize DINOv2
        try:
            print(f"DINOv2 모델 로드 중: {dinov2_model}")
            self.image_processor = AutoImageProcessor.from_pretrained(dinov2_model)
            self.dinov2_model = AutoModel.from_pretrained(dinov2_model).to(self.device)
            print("DINOv2 모델 로드 완료")
        except Exception as e:
            print(f"DINOv2 모델 로드 실패: {e}")
            raise
        
        # Initialize memory system
        self.memory = MemorySystem(memory_dir)
        
        # Create results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # State variables
        self.current_image = None
        self.current_image_path = None
        self.current_mask = None
        self.current_features = None
    
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        DINOv2를 사용하여 이미지에서 특징 추출
        
        Args:
            image: 입력 이미지 (numpy 배열)
            
        Returns:
            추출된 특징 벡터
        """
        # Convert to PIL image if numpy array
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Process image for DINOv2
        inputs = self.image_processor(images=image_pil, return_tensors="pt").to(self.device)
        
        # Extract features
        with torch.no_grad():
            outputs = self.dinov2_model(**inputs)
        
        # Get CLS token features
        features = outputs.last_hidden_state[:, 0].cpu().numpy()
        
        return features[0]  # Return the feature vector
    
    def generate_prompt(self, mask: np.ndarray, prompt_type: str = "points") -> Dict:
        """
        SAM2 프롬프트 생성
        
        Args:
            mask: 마스크 배열
            prompt_type: 프롬프트 유형 ("points" 또는 "box")
            
        Returns:
            프롬프트 데이터가 포함된 사전
        """
        if prompt_type not in ["points", "box"]:
            raise ValueError("prompt_type은 'points' 또는 'box'여야 합니다")
        
        # 마스크 차원 확인
        if mask.ndim > 2:
            # 다중 채널 마스크는 첫 번째 채널만 사용 (예: RGB 또는 RGBA 이미지인 경우)
            print(f"다중 채널 마스크 감지됨. 차원: {mask.shape}")
            if mask.shape[2] >= 3:  # RGB 또는 RGBA
                # 모든 채널이 동일한 값을 가지면 첫 번째 채널만 사용
                mask = mask[:, :, 0]
            elif mask.ndim == 3 and mask.shape[2] == 1:
                # 단일 채널 3D 마스크 (높이, 너비, 1)
                mask = mask[:, :, 0]
        
        print(f"마스크 차원: {mask.shape}, 타입: {mask.dtype}")
        
        if prompt_type == "points":
            # 전경 포인트 생성 (마스크에서 무작위 샘플링)
            where_result = np.where(mask > 0)
            if len(where_result) >= 2:
                fg_y, fg_x = where_result[0], where_result[1]
                if len(fg_y) > 0:
                    # 전경에서 최대 5개 포인트 샘플링
                    num_fg_points = min(5, len(fg_y))
                    fg_indices = np.random.choice(len(fg_y), num_fg_points, replace=False)
                    fg_points = np.array([[fg_x[i], fg_y[i]] for i in fg_indices])
                    fg_labels = np.ones(num_fg_points)
                else:
                    fg_points = np.empty((0, 2))
                    fg_labels = np.empty(0)
            else:
                print(f"전경 마스크 추출 실패: where_result={where_result}")
                fg_points = np.empty((0, 2))
                fg_labels = np.empty(0)
                
            # 배경 포인트 생성 (비마스크에서 무작위 샘플링)
            where_result = np.where(mask == 0)
            if len(where_result) >= 2:
                bg_y, bg_x = where_result[0], where_result[1]
                if len(bg_y) > 0:
                    # 배경에서 최대 5개 포인트 샘플링
                    num_bg_points = min(5, len(bg_y))
                    bg_indices = np.random.choice(len(bg_y), num_bg_points, replace=False)
                    bg_points = np.array([[bg_x[i], bg_y[i]] for i in bg_indices])
                    bg_labels = np.zeros(num_bg_points)
                else:
                    bg_points = np.empty((0, 2))
                    bg_labels = np.empty(0)
            else:
                print(f"배경 마스크 추출 실패: where_result={where_result}")
                bg_points = np.empty((0, 2))
                bg_labels = np.empty(0)
                
            # 전경 및 배경 포인트 결합
            points = np.vstack([fg_points, bg_points]) if len(fg_points) > 0 and len(bg_points) > 0 else (fg_points if len(fg_points) > 0 else bg_points)
            labels = np.concatenate([fg_labels, bg_labels]) if len(fg_labels) > 0 and len(bg_labels) > 0 else (fg_labels if len(fg_labels) > 0 else bg_labels)
            
            return {
                "type": "points",
                "points": points,
                "labels": labels
            }
            
        else:  # box prompt
            # 마스크의 경계 상자 찾기
            where_result = np.where(mask > 0)
            if len(where_result) >= 2:
                y, x = where_result[0], where_result[1]
                if len(y) > 0:
                    x_min, x_max = np.min(x), np.max(x)
                    y_min, y_max = np.min(y), np.max(y)
                    
                    # 상자에 약간의 패딩 추가
                    h, w = mask.shape
                    x_min = max(0, x_min - 5)
                    y_min = max(0, y_min - 5)
                    x_max = min(w - 1, x_max + 5)
                    y_max = min(h - 1, y_max + 5)
                    
                    box = np.array([x_min, y_min, x_max, y_max])
                    return {
                        "type": "box",
                        "box": box
                    }
            
            # 마스크가 비어 있거나 추출 실패 시 이미지 중앙에 작은 상자 반환
            h, w = mask.shape
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 4
            return {
                "type": "box",
                "box": np.array([center_x - size, center_y - size, center_x + size, center_y + size])
            }
    
    def _create_default_prompt(self, image: np.ndarray, prompt_type: str) -> Dict:
        """
        기본 프롬프트 생성 (메모리 또는 참조 없을 때)
        
        Args:
            image: 입력 이미지
            prompt_type: 프롬프트 타입 ('points' 또는 'box')
            
        Returns:
            프롬프트 사전
        """
        # 상자 프롬프트의 경우 중앙 상자 사용
        if prompt_type == "box":
            h, w = image.shape[:2]
            center_x, center_y = w // 2, h // 2
            size = min(w, h) // 4
            box = np.array([
                center_x - size, 
                center_y - size, 
                center_x + size, 
                center_y + size
            ])
            return {"type": "box", "box": box}
        else:
            # 포인트 프롬프트의 경우 중앙 포인트 사용
            h, w = image.shape[:2]
            points = np.array([[w // 2, h // 2]])
            labels = np.array([1])  # 전경 포인트
            return {"type": "points", "points": points, "labels": labels}
    
    def segment_with_sam(self, 
                        image: np.ndarray, 
                        prompt: Dict,
                        multimask_output: bool = True) -> Tuple[np.ndarray, float]:
        """
        주어진 프롬프트로 SAM2를 사용하여 이미지 세그멘테이션
        
        Args:
            image: 입력 이미지
            prompt: generate_prompt()로 생성된 프롬프트 사전
            multimask_output: 여러 마스크 출력 여부
            
        Returns:
            (best_mask, score) 튜플
        """
        # 프레딕터에 이미지 설정
        self.predictor.set_image(image)
        
        # 프롬프트 유형에 따라 마스크 생성
        if prompt["type"] == "points":
            # 포인트 프롬프트
            masks, scores, _ = self.predictor.predict(
                point_coords=prompt["points"],
                point_labels=prompt["labels"],
                multimask_output=multimask_output
            )
        else:
            # 상자 프롬프트
            masks, scores, _ = self.predictor.predict(
                box=prompt["box"][None, :],
                multimask_output=multimask_output
            )
        
        # 최상의 마스크 선택
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        best_score = scores[best_idx]
        
        return best_mask, best_score
    
    def process_image(self, 
                     image_path: str, 
                     reference_path: str = None,
                     prompt_type: str = "points") -> Dict:
        """
        Memory SAM 시스템을 사용하여 이미지 처리
        
        Args:
            image_path: 입력 이미지 경로 (파일 또는 폴더)
            reference_path: 참조 이미지 경로 (선택 사항)
            prompt_type: 생성할 프롬프트 유형 ('points' 또는 'box')
            
        Returns:
            처리 결과가 포함된 사전
        """
        # 이미지 로드 - 파일 또는 폴더 경로 처리
        image_path = Path(image_path)
        
        # 폴더인 경우 첫 번째 이미지만 처리 (반환용)
        if image_path.is_dir():
            # 이미지 확장자
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_files = [f for f in image_path.glob('*') if f.suffix.lower() in valid_extensions]
            
            if not image_files:
                raise ValueError(f"폴더 {image_path}에 이미지 파일이 없습니다.")
            
            # 대표 이미지로 첫 번째 이미지 사용
            rep_image_path = str(image_files[0])
            image = np.array(Image.open(rep_image_path).convert("RGB"))
            self.current_image = image
            self.current_image_path = rep_image_path
            
            # 나중에 추가 처리를 위해 모든 이미지 경로 저장
            self.folder_image_paths = [str(f) for f in image_files]
            self.is_folder_processing = True
        else:
            # 단일 이미지 파일
            image = np.array(Image.open(image_path).convert("RGB"))
            self.current_image = image
            self.current_image_path = str(image_path)
            self.is_folder_processing = False
        
        # DINOv2로 특징 추출
        features = self.extract_features(image)
        self.current_features = features
        
        reference_mask = None
        
        # 참조 이미지가 제공된 경우 사용
        if reference_path:
            print(f"참조 이미지 사용: {reference_path}")
            try:
                reference_img = np.array(Image.open(reference_path).convert("RGB"))
                reference_mask_path = Path(reference_path).with_suffix('.png')
                
                if reference_mask_path.exists():
                    reference_mask = np.array(Image.open(reference_mask_path))
                    print(f"참조 마스크 로드됨: {reference_mask_path}, 형태: {reference_mask.shape}, 타입: {reference_mask.dtype}")
                    
                    # 마스크가 그레이스케일인지 확인
                    if reference_mask.ndim == 2:
                        print("그레이스케일 마스크 감지됨")
                    elif reference_mask.ndim == 3:
                        print(f"색상 마스크 감지됨, 채널 수: {reference_mask.shape[2]}")
                        # 다중 채널을 단일 채널로 변환 (첫 번째 채널 사용)
                        if reference_mask.shape[2] >= 3:
                            reference_mask = reference_mask[:, :, 0]
                            print(f"첫 번째 채널로 변환됨, 새 형태: {reference_mask.shape}")
                else:
                    print(f"경고: 참조 마스크 {reference_mask_path}를 찾을 수 없습니다")
                    reference_mask = None
            except Exception as e:
                print(f"참조 이미지 처리 중 오류: {e}")
                reference_mask = None
        
        # 메모리에서 유사한 이미지 찾기 또는 참조 사용
        if reference_mask is not None:
            # 참조 마스크를 사용하여 프롬프트 생성
            prompt = self.generate_prompt(reference_mask, prompt_type)
            similar_items = []
        else:
            # 유사한 이미지에 대한 메모리 쿼리
            similar_items = self.memory.get_most_similar(features, top_k=3)
            
            if similar_items:
                try:
                    # 가장 일치하는 메모리 항목 가져오기
                    best_item = similar_items[0]["item"]
                    item_data = self.memory.load_item_data(best_item["id"])
                    
                    print(f"메모리 마스크 로드됨: ID {best_item['id']}")
                    if "mask" in item_data:
                        mask_data = item_data["mask"]
                        print(f"메모리 마스크 형태: {mask_data.shape}, 타입: {mask_data.dtype}")
                        
                        # 마스크 정규화
                        if mask_data.dtype != np.bool_:
                            # 이진 마스크로 변환 (0 또는 1)
                            mask_data = (mask_data > 0).astype(np.uint8)
                        
                        # 메모리 마스크에서 프롬프트 생성
                        prompt = self.generate_prompt(mask_data, prompt_type)
                    else:
                        print("메모리 항목에 마스크가 없습니다. 기본 프롬프트 사용")
                        raise KeyError("mask")
                except Exception as e:
                    print(f"메모리 마스크 처리 중 오류: {e}")
                    # 오류 발생 시 기본 프롬프트 사용
                    prompt = self._create_default_prompt(image, prompt_type)
            else:
                # 메모리 항목을 찾을 수 없음, 기본 프롬프트 사용
                prompt = self._create_default_prompt(image, prompt_type)
        
        # SAM2로 세그멘테이션
        mask, score = self.segment_with_sam(image, prompt)
        self.current_mask = mask
        
        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = self.results_dir / f"result_{timestamp}"
        result_path.mkdir(exist_ok=True, parents=True)
        
        # 원본 이미지 저장
        Image.fromarray(image).save(str(result_path / "input.png"))
        
        # 마스크 저장
        mask_img = (mask * 255).astype(np.uint8)
        Image.fromarray(mask_img).save(str(result_path / "mask.png"))
        
        # 시각화 저장
        vis_img = self.visualize_mask(image, mask)
        Image.fromarray(vis_img).save(str(result_path / "visualization.png"))
        
        # 폴더 처리인 경우 모든 이미지 처리 및 저장
        processed_images = []
        if hasattr(self, 'is_folder_processing') and self.is_folder_processing:
            folder_result_path = result_path / "all_images"
            folder_result_path.mkdir(exist_ok=True, parents=True)
            
            for img_path in self.folder_image_paths:
                try:
                    folder_img = np.array(Image.open(img_path).convert("RGB"))
                    folder_img_features = self.extract_features(folder_img)
                    
                    # 첫 번째 이미지와 같은 프롬프트 유형과 설정 사용
                    folder_mask, folder_score = self.segment_with_sam(folder_img, prompt)
                    
                    # 이미지 파일명 추출
                    img_filename = Path(img_path).stem
                    
                    # 결과 저장
                    folder_mask_img = (folder_mask * 255).astype(np.uint8)
                    folder_vis_img = self.visualize_mask(folder_img, folder_mask)
                    
                    Image.fromarray(folder_img).save(str(folder_result_path / f"{img_filename}_input.png"))
                    Image.fromarray(folder_mask_img).save(str(folder_result_path / f"{img_filename}_mask.png"))
                    Image.fromarray(folder_vis_img).save(str(folder_result_path / f"{img_filename}_overlay.png"))
                    
                    processed_images.append({
                        "path": img_path,
                        "filename": img_filename,
                        "score": float(folder_score),
                        "input": str(folder_result_path / f"{img_filename}_input.png"),
                        "mask": str(folder_result_path / f"{img_filename}_mask.png"),
                        "overlay": str(folder_result_path / f"{img_filename}_overlay.png")
                    })
                except Exception as e:
                    print(f"이미지 처리 중 오류 {img_path}: {e}")
        
        # 메모리에 추가
        memory_id = self.memory.add_memory(
            image=image,
            mask=mask,
            features=features,
            metadata={
                "original_path": str(image_path),
                "timestamp": timestamp,
                "score": float(score),
                "is_folder": hasattr(self, 'is_folder_processing') and self.is_folder_processing,
                "result_folder": str(result_path)
            }
        )
        
        return {
            "image": image,
            "mask": mask,
            "score": float(score),
            "memory_id": memory_id,
            "result_path": str(result_path),
            "similar_items": similar_items,
            "features": features,
            "processed_images": processed_images,
            "is_folder": hasattr(self, 'is_folder_processing') and self.is_folder_processing
        }
    
    def visualize_mask(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        이미지에 마스크 오버레이 시각화
        
        Args:
            image: 원본 이미지
            mask: 세그멘테이션 마스크
            alpha: 불투명도
            
        Returns:
            시각화된 이미지
        """
        vis = image.copy()
        
        # 마스크용 색상 오버레이 생성
        color_mask = np.zeros_like(image)
        color_mask[mask > 0] = [30, 144, 255]  # 마스크용 파란색
        
        # 이미지와 마스크 블렌딩
        vis = cv2.addWeighted(vis, 1, color_mask, alpha, 0)
        
        # 윤곽선 그리기
        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
        
        return vis