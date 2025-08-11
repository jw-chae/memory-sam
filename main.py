#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path

# 스크립트 디렉토리를 Python 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="Memory SAM 인터페이스 실행")
    
    parser.add_argument(
        "--interface", 
        type=str, 
        default="enhanced", 
        choices=["memory", "sam2_ui", "enhanced"],
        help="사용할 인터페이스 유형 (memory: Memory SAM, sam2_ui: SAM2 UI, enhanced: 향상된 UI)"
    )
    
    parser.add_argument(
        "--model_type", 
        type=str, 
        default="hiera_l", 
        choices=["hiera_b+", "hiera_l", "hiera_s", "hiera_t"],
        help="사용할 SAM2 모델 유형"
    )
    
    parser.add_argument(
        "--checkpoint_path", 
        type=str, 
        default=None,
        help="SAM2 체크포인트 경로 (기본값: 자동 감지)"
    )
    
    parser.add_argument(
        "--dinov2_model", 
        type=str, 
        default="facebook/dinov2-base",
        help="사용할 DINOv2 모델"
    )
    
    parser.add_argument(
        "--memory_dir", 
        type=str, 
        default="memory",
        help="메모리 시스템 디렉토리"
    )
    
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="results",
        help="결과 저장 디렉토리"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="사용할 디바이스 (cuda, cpu 또는 mps)"
    )
    
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Gradio 인터페이스를 공개 URL로 공유"
    )
    
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_args()
    
    # 디렉토리 경로 설정 (절대 경로로 변환)
    memory_dir = os.path.abspath(args.memory_dir)
    results_dir = os.path.abspath(args.results_dir)
    
    # 디렉토리 생성
    os.makedirs(memory_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 인터페이스 유형에 따라 적절한 인터페이스 로드
    if args.interface == "memory":
        from scripts.memory_sam_predictor import MemorySAMPredictor
        from scripts.memory_gradio_interface import MemoryGradioInterface
        
        print("Memory SAM 인터페이스 초기화 중...")
        
        # Memory SAM 예측기 초기화
        predictor = MemorySAMPredictor(
            model_type=args.model_type,
            checkpoint_path=args.checkpoint_path,
            dinov2_model=args.dinov2_model,
            memory_dir=memory_dir,
            results_dir=results_dir,
            device=args.device
        )
        
        # Memory SAM Gradio 인터페이스 초기화
        interface = MemoryGradioInterface(
            memory_sam_predictor=predictor,
            model_type=args.model_type,
            checkpoint_path=args.checkpoint_path,
            dinov2_model=args.dinov2_model,
            memory_dir=memory_dir,
            results_dir=results_dir
        )
        
        # Gradio 인터페이스 설정 및 실행
        app = interface.setup_interface()
        # 허용된 경로 추가 (메모리 및 결과 디렉토리)
        app.launch(share=args.share, allowed_paths=[memory_dir, results_dir])
    
    elif args.interface == "enhanced":
        from ui.memory_sam_ui import MemorySAMUI
        
        print("향상된 Memory SAM 인터페이스 초기화 중...")
        
        # 향상된 Memory SAM UI 초기화
        memory_ui = MemorySAMUI(
            model_type=args.model_type,
            checkpoint_path=args.checkpoint_path,
            dinov2_model=args.dinov2_model,
            memory_dir=memory_dir,
            results_dir=results_dir,
            device=args.device,
            use_sparse_matching=True
        )
        
        # Gradio 인터페이스 설정 및 실행
        app = memory_ui.setup_interface()
        # 허용된 경로 추가 (메모리 및 결과 디렉토리)
        app.launch(share=args.share, allowed_paths=[memory_dir, results_dir])
        
    else:  # sam2_ui
        print("SAM2 UI 인터페이스 초기화 중...")
        
        # sam2_ui.py 모듈 임포트
        import importlib.util
        import sys
        
        # sam2_ui.py 파일 경로
        sam2_ui_path = os.path.join(script_dir, "sam2_ui.py")
        
        # 체크포인트 경로 설정
        if args.checkpoint_path:
            # 환경 변수로 체크포인트 경로 전달
            os.environ["SAM2_CHECKPOINT"] = args.checkpoint_path
            print(f"체크포인트 경로 설정: {args.checkpoint_path}")
        
        # 모듈 스펙 로드
        spec = importlib.util.spec_from_file_location("sam2_ui", sam2_ui_path)
        sam2_ui = importlib.util.module_from_spec(spec)
        sys.modules["sam2_ui"] = sam2_ui
        spec.loader.exec_module(sam2_ui)
        
        # SAM2 UI의 Gradio 인터페이스 실행 (허용된 경로 추가)
        sam2_ui.demo.launch(share=args.share, allowed_paths=[memory_dir, results_dir])
        
        print("SAM2 UI 인터페이스가 종료되었습니다.")
    
    print("인터페이스가 종료되었습니다.")

if __name__ == "__main__":
    main()