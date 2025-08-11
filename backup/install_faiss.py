#!/usr/bin/env python3
"""
FAISS 설치 스크립트

이 스크립트는 FAISS 라이브러리를 설치하고 필요한 의존성을 확인합니다.
"""

import subprocess
import sys
import os

def check_cuda():
    """CUDA 가용성 확인"""
    try:
        import torch
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return torch.cuda.is_available()
    except ImportError:
        print("PyTorch가 설치되어 있지 않습니다.")
        return False

def install_faiss(cuda_available=False):
    """FAISS 설치"""
    if cuda_available:
        print("CUDA 지원 FAISS 설치 중...")
        package = "faiss-gpu"
    else:
        print("CPU 전용 FAISS 설치 중...")
        package = "faiss-cpu"
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} 설치 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAISS 설치 중 오류 발생: {e}")
        return False

def verify_installation():
    """FAISS 설치 확인"""
    try:
        import faiss
        print(f"FAISS 버전: {faiss.__version__}")
        
        # 간단한 테스트
        d = 64                           # 차원
        nb = 100                         # 데이터베이스 크기
        nq = 10                          # 쿼리 크기
        
        import numpy as np
        xb = np.random.random((nb, d)).astype('float32')
        xq = np.random.random((nq, d)).astype('float32')
        
        index = faiss.IndexFlatL2(d)     # L2 거리 기반 인덱스 생성
        print(f"인덱스 생성됨: {index.is_trained}")
        index.add(xb)                    # 데이터베이스 벡터 추가
        print(f"인덱스 크기: {index.ntotal}")
        
        k = 4                            # 각 쿼리에 대해 찾을 이웃 수
        D, I = index.search(xq, k)       # 실제 검색
        print(f"검색 결과 형태: {D.shape}")
        
        print("FAISS 설치 및 테스트 완료!")
        return True
    except ImportError:
        print("FAISS 설치를 확인할 수 없습니다.")
        return False
    except Exception as e:
        print(f"FAISS 테스트 중 오류 발생: {e}")
        return False

def main():
    print("FAISS 설치 스크립트 시작")
    
    # CUDA 확인
    cuda_available = check_cuda()
    
    # FAISS 설치
    install_success = install_faiss(cuda_available)
    
    if install_success:
        # 설치 확인
        verify_success = verify_installation()
        if verify_success:
            print("\n성공: FAISS가 올바르게 설치되었습니다!")
        else:
            print("\n경고: FAISS 설치가 완료되었지만 확인 중 문제가 발생했습니다.")
    else:
        print("\n오류: FAISS 설치에 실패했습니다.")
    
    print("\n추가 정보:")
    print("- FAISS 문서: https://github.com/facebookresearch/faiss/wiki")
    print("- 문제 해결: https://github.com/facebookresearch/faiss/wiki/Troubleshooting")

if __name__ == "__main__":
    main() 