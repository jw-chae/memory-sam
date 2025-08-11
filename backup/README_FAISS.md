# FAISS를 이용한 효율적인 유사도 검색

이 문서는 SAM2 프로젝트에서 FAISS(Facebook AI Similarity Search)를 사용하여 효율적인 유사도 검색을 구현하는 방법을 설명합니다.

## FAISS란?

FAISS는 Facebook AI Research에서 개발한 라이브러리로, 고차원 벡터의 효율적인 유사도 검색과 클러스터링을 위한 도구입니다. 주요 특징은 다음과 같습니다:

- 대규모 데이터셋에서 빠른 유사도 검색
- GPU 가속 지원
- 다양한 인덱스 타입 제공 (정확도와 속도 간의 트레이드오프)
- 메모리 효율적인 검색 알고리즘

## 설치 방법

FAISS는 CPU 버전과 GPU 버전이 있습니다. GPU 버전은 CUDA가 설치된 환경에서만 사용할 수 있습니다.

### 자동 설치

제공된 설치 스크립트를 사용하여 FAISS를 설치할 수 있습니다:

```bash
python scripts/install_faiss.py
```

이 스크립트는 시스템에 CUDA가 설치되어 있는지 확인하고, 적절한 버전의 FAISS를 설치합니다.

### 수동 설치

pip를 사용하여 직접 설치할 수도 있습니다:

CPU 버전:
```bash
pip install faiss-cpu
```

GPU 버전:
```bash
pip install faiss-gpu
```

## 사용 방법

### 기본 사용법

FAISS를 사용한 기본적인 유사도 검색 과정은 다음과 같습니다:

1. 특징 벡터 추출
2. FAISS 인덱스 생성
3. 인덱스에 벡터 추가
4. 쿼리 벡터로 검색 수행

```python
import faiss
import numpy as np

# 1. 특징 벡터 준비 (예: 임의의 벡터)
d = 64                           # 차원
nb = 100000                      # 데이터베이스 크기
nq = 10                          # 쿼리 크기
xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# 2. 인덱스 생성
index = faiss.IndexFlatL2(d)     # L2 거리 기반 인덱스

# 3. 인덱스에 벡터 추가
index.add(xb)                    # 데이터베이스 벡터 추가

# 4. 검색 수행
k = 4                            # 각 쿼리에 대해 찾을 이웃 수
D, I = index.search(xq, k)       # 실제 검색
# D: 거리 배열, I: 인덱스 배열
```

### 예제 스크립트

프로젝트에는 FAISS를 사용한 이미지 유사도 검색 예제 스크립트가 포함되어 있습니다:

```bash
python scripts/example_faiss_matching.py --dataset ./dataset --query ./query_image.jpg
```

이 스크립트는 다음과 같은 기능을 수행합니다:

1. 데이터셋 폴더에서 모든 이미지 로드
2. DINOv2 모델을 사용하여 각 이미지의 특징 추출
3. FAISS 인덱스 구축 및 저장
4. 쿼리 이미지와 가장 유사한 이미지 검색
5. 결과 시각화

## 프로젝트 통합

SAM2 프로젝트에서는 다음 파일들이 FAISS를 사용하도록 수정되었습니다:

1. `memory_system.py`: 메모리 시스템에서 유사도 검색에 FAISS 사용
2. `dinov2_matcher.py`: DINOv2 특징 매칭에 FAISS 사용

### 성능 비교

FAISS를 사용하면 기존 sklearn의 NearestNeighbors에 비해 다음과 같은 성능 향상을 기대할 수 있습니다:

- 대규모 데이터셋에서 검색 속도 향상 (최대 100배 이상)
- GPU 가속을 통한 추가 성능 향상
- 메모리 사용량 최적화

## 고급 사용법

### 인덱스 타입

FAISS는 다양한 인덱스 타입을 제공합니다:

- `IndexFlatL2`: 정확한 L2 거리 계산 (가장 정확하지만 느림)
- `IndexIVFFlat`: 역 파일 인덱스 (더 빠르지만 약간의 정확도 손실)
- `IndexIVFPQ`: 제품 양자화를 사용한 압축 인덱스 (매우 빠르고 메모리 효율적)
- `IndexHNSW`: 계층적 네비게이션 소형 월드 그래프 (빠른 검색과 좋은 정확도)

예를 들어, IVF 인덱스를 사용하는 방법:

```python
nlist = 100  # 클러스터 수
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)
index.train(xb)  # IVF 인덱스는 훈련이 필요함
index.add(xb)
```

### GPU 가속

FAISS는 GPU 가속을 지원합니다:

```python
res = faiss.StandardGpuResources()  # GPU 리소스 할당
index_flat = faiss.IndexFlatL2(d)   # CPU 인덱스 생성
gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)  # GPU로 변환
```

## 문제 해결

- **메모리 오류**: 대규모 데이터셋을 처리할 때 메모리 오류가 발생하면 배치 처리를 고려하세요.
- **정확도 문제**: 정확도가 중요한 경우 `IndexFlatL2`를 사용하세요.
- **속도 문제**: 속도가 중요한 경우 `IndexIVFPQ`나 `IndexHNSW`를 고려하세요.

## 참고 자료

- [FAISS 공식 문서](https://github.com/facebookresearch/faiss/wiki)
- [FAISS 튜토리얼](https://github.com/facebookresearch/faiss/wiki/Getting-started)
- [FAISS 인덱스 타입](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) 